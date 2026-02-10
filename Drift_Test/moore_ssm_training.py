"""Training script for MooreDotSSM.

Parses SPOT-format traces and trains the SSM from a Moore DOT file.

Trace format: "ap1&!ap2&ap3;ap1&ap2&!ap3;...;cycle{n}"
Each step contains all APs (inputs AND outputs).

Architecture:
    μ_t = h_t ⊗ σ_t                    # Kronecker product
    h_{t+1} = A @ h_t + B @ μ_t        # State update (optionally with tanh)
    y_t = C @ h_{t+1}                  # Output AFTER transition (one-hot symbol)

Output decoding:
    - Model outputs one-hot over 2^|Λ| symbols
    - For BCE training, we convert to binary via learned/fixed projection
    - For accuracy, we use argmax then decode to binary
"""

import argparse
import csv
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from moore_ssm import MooreDotSSM, build_transition_table, parse_moore_dot

# =============================================================================
# TRACE PARSING
# =============================================================================


def parse_spot_step(step_str: str, ap_names: List[str]) -> Dict[str, bool]:
    """Parse a single SPOT step like "g_0&!c_0&r_0" into a dict."""
    values = {ap: False for ap in ap_names}

    for lit in step_str.split("&"):
        lit = lit.strip()
        if not lit:
            continue
        if lit.startswith("!"):
            ap = lit[1:]
            val = False
        else:
            ap = lit
            val = True
        if ap in values:
            values[ap] = val

    return values


def parse_spot_trace(trace_str: str, ap_names: List[str]) -> List[Dict[str, bool]]:
    """Parse a full SPOT trace into list of dicts."""
    trace_str = re.sub(r";?cycle\{.*\}$", "", trace_str.strip())
    steps = trace_str.split(";")
    return [parse_spot_step(s, ap_names) for s in steps if s.strip()]


def load_traces(filename: str, ap_names: List[str]) -> List[List[Dict[str, bool]]]:
    """Load all traces from a file."""
    traces = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                trace = parse_spot_trace(line, ap_names)
                if trace:
                    traces.append(trace)
    return traces


# =============================================================================
# DATA ENCODING
# =============================================================================


def encode_input_onehot(step: Dict[str, bool], input_aps: List[str]) -> torch.Tensor:
    """Encode input APs as one-hot vector."""
    num_inputs = 2 ** len(input_aps)

    idx = 0
    for i, ap in enumerate(input_aps):
        bit_pos = len(input_aps) - 1 - i  # MSB first
        if step.get(ap, False):
            idx |= 1 << bit_pos

    one_hot = torch.zeros(num_inputs)
    one_hot[idx] = 1.0
    return one_hot


def encode_output_binary(step: Dict[str, bool], output_aps: List[str]) -> torch.Tensor:
    """Encode output APs as binary vector (for BCE loss)."""
    return torch.tensor([float(step.get(ap, False)) for ap in output_aps])


def prepare_dataset(
    traces: List[List[Dict[str, bool]]], input_aps: List[str], output_aps: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
    """Prepare dataset from parsed traces.

    Returns:
        X: (num_traces, max_len, 2^num_inputs) - one-hot inputs
        Y: (num_traces, max_len, num_outputs) - binary outputs
        seq_lens: {trace_idx: length}
    """
    num_traces = len(traces)
    max_len = max(len(t) for t in traces)
    num_input_symbols = 2 ** len(input_aps)
    num_outputs = len(output_aps)

    X = torch.zeros(num_traces, max_len, num_input_symbols)
    Y = torch.zeros(num_traces, max_len, num_outputs)
    seq_lens = {}

    for i, trace in enumerate(traces):
        seq_lens[i] = len(trace)
        for t, step in enumerate(trace):
            X[i, t] = encode_input_onehot(step, input_aps)
            Y[i, t] = encode_output_binary(step, output_aps)

    return X, Y, seq_lens


# =============================================================================
# OUTPUT DECODING
# =============================================================================


def build_symbol_to_binary_matrix(num_output_aps: int) -> torch.Tensor:
    """Build matrix to convert one-hot symbol to binary vector.

    D ∈ ℝ^{num_output_aps × 2^num_output_aps}
    D[i, j] = 1 if bit i is set in symbol j

    Usage: binary = D @ one_hot_symbol
    """
    num_symbols = 2**num_output_aps
    D = torch.zeros(num_output_aps, num_symbols)

    for symbol_idx in range(num_symbols):
        for bit_idx in range(num_output_aps):
            bit_pos = num_output_aps - 1 - bit_idx  # MSB first
            if (symbol_idx >> bit_pos) & 1:
                D[bit_idx, symbol_idx] = 1.0

    return D


def onehot_to_binary(y_onehot: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Convert one-hot symbol logits to binary logits.

    Args:
        y_onehot: (B, T, 2^num_outputs) - one-hot symbol logits
        D: (num_outputs, 2^num_outputs) - decoding matrix

    Returns:
        y_binary: (B, T, num_outputs) - binary logits
    """
    # y_onehot @ D.T -> (B, T, num_outputs)
    return y_onehot @ D.T


# =============================================================================
# TRAINING UTILITIES
# =============================================================================


def get_padding_mask(
    seq_lens: Dict[int, int], batch_size: int, max_len: int, device
) -> torch.Tensor:
    """Create mask where True = valid timestep."""
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        length = seq_lens.get(i, max_len)
        mask[i, :length] = True
    return mask


def masked_bce_loss(
    y_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute BCE loss only on valid timesteps."""
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    mask_expanded = mask.unsqueeze(-1).expand_as(y_hat)
    loss_per_elem = criterion(y_hat, y_true)
    masked_loss = loss_per_elem * mask_expanded.float()

    total_loss = masked_loss.sum()
    count = mask_expanded.float().sum()

    return total_loss / count if count > 0 else total_loss


def masked_accuracy(
    y_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor
) -> float:
    """Per-timestep accuracy on valid positions."""
    preds = (y_hat > 0).float()
    correct = (preds == y_true).all(dim=-1)

    valid_correct = (correct & mask).sum().item()
    valid_total = mask.sum().item()

    return valid_correct / valid_total if valid_total > 0 else 0.0


def trace_accuracy(
    y_hat: torch.Tensor, y_true: torch.Tensor, seq_lens: Dict[int, int]
) -> float:
    """Fraction of traces where ALL timesteps are correct."""
    B, T, _ = y_hat.shape
    device = y_hat.device
    mask = get_padding_mask(seq_lens, B, T, device)

    preds = (y_hat > 0).float()
    correct = (preds == y_true).all(dim=-1)

    correct = correct | ~mask  # Ignore padding
    trace_correct = correct.all(dim=-1)

    return trace_correct.float().mean().item()


# =============================================================================
# CSV LOGGING
# =============================================================================


def init_csv_log(csv_path: str) -> None:
    """Initialize CSV file with headers."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "loss",
                "train_step_acc",
                "train_trace_acc",
                "test_step_acc",
                "test_trace_acc",
            ]
        )


def append_csv_log(
    csv_path: str,
    epoch: int,
    loss: float,
    train_step_acc: float,
    train_trace_acc: float,
    test_step_acc: float,
    test_trace_acc: float,
) -> None:
    """Append a row to the CSV log."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                f"{loss:.6f}",
                f"{train_step_acc:.6f}",
                f"{train_trace_acc:.6f}",
                f"{test_step_acc:.6f}",
                f"{test_trace_acc:.6f}",
            ]
        )


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================


def diagnose_first_trace(
    model: MooreDotSSM,
    D: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    traces: List[List[Dict[str, bool]]],
    num_steps: int = 5,
):
    """Print detailed diagnostic for first trace."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: First trace step-by-step")
    print("=" * 70)

    input_aps = model.machine.input_aps
    output_aps = model.machine.output_aps

    print(f"Input APs: {input_aps}")
    print(f"Output APs: {output_aps}")

    trace = traces[0]

    with torch.no_grad():
        y_onehot = model(X[:1])
        y_binary = onehot_to_binary(y_onehot, D)

    print(f"\nFirst {min(num_steps, len(trace))} steps:")
    for t in range(min(num_steps, len(trace))):
        step = trace[t]

        # Extract values
        input_vals = [step.get(ap, False) for ap in input_aps]
        output_vals = [step.get(ap, False) for ap in output_aps]

        input_idx = X[0, t].argmax().item()

        # One-hot symbol prediction
        symbol_idx = y_onehot[0, t].argmax().item()
        symbol_aps = model.output_symbol_to_aps(symbol_idx)

        # Binary prediction
        pred_binary = (y_binary[0, t] > 0).tolist()
        target_binary = (Y[0, t] > 0.5).tolist()

        match = pred_binary == target_binary
        status = "✓" if match else "✗"

        print(f"\nt={t}:")
        print(f"  Raw step: {step}")
        print(f"  Input vals {input_aps}: {input_vals} -> idx={input_idx}")
        print(f"  Output vals {output_aps}: {output_vals}")
        print(f"  Symbol idx: {symbol_idx} -> {symbol_aps}")
        print(f"  Target binary: {target_binary}")
        print(
            f"  Predicted binary: {pred_binary} (raw: {[f'{v:.3f}' for v in y_binary[0, t].tolist()]}) {status}"
        )


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train MooreDotSSM model")
    parser.add_argument("--dot", type=str, required=True, help="Moore machine DOT file")
    parser.add_argument(
        "--data", type=str, required=True, help="Training data file (SPOT format)"
    )
    parser.add_argument(
        "--inputs", type=str, required=True, help="Comma-separated input APs"
    )
    parser.add_argument(
        "--outputs", type=str, required=True, help="Comma-separated output APs"
    )
    parser.add_argument("--epochs", type=int, default=1500, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--epsilon", type=float, default=0.0, help="Noise for zero entries in A, B, C"
    )
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--save", type=str, default=None, help="Save model to path")
    parser.add_argument(
        "--save-csv", type=str, default=None, help="Save training log to CSV path"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify symbolic accuracy"
    )
    parser.add_argument(
        "--diagnose", action="store_true", help="Print detailed diagnostic"
    )
    parser.add_argument(
        "--log-interval", type=int, default=25, help="Logging interval (epochs)"
    )
    parser.add_argument(
        "--use-tanh", action="store_true", help="Use tanh activation on hidden state"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse APs
    input_aps = sorted([x.strip() for x in args.inputs.split(",") if x.strip()])
    output_aps = sorted([x.strip() for x in args.outputs.split(",") if x.strip()])
    all_aps = input_aps + output_aps

    print(f"Input APs: {input_aps}")
    print(f"Output APs: {output_aps}")

    # Load and parse DOT file
    with open(args.dot, "r") as f:
        dot_string = f.read()

    machine = parse_moore_dot(dot_string, input_aps, output_aps)
    transition_table = build_transition_table(machine)

    print(f"\nLoaded Moore machine from {args.dot}")
    print(f"  States: {len(machine.states)}")
    print(f"  Initial state: {machine.initial_state}")

    # For verify-only and diagnose, use epsilon=0 and no tanh
    if args.verify_only or args.diagnose:
        effective_epsilon = 0.0
        use_tanh = False
    else:
        effective_epsilon = args.epsilon
        use_tanh = args.use_tanh

    # Create model
    model = MooreDotSSM(
        machine, transition_table, epsilon=effective_epsilon, use_tanh=use_tanh
    ).to(device)

    # Build symbol-to-binary decoding matrix
    D = build_symbol_to_binary_matrix(len(output_aps)).to(device)

    print(f"\nSSM matrices (epsilon={effective_epsilon}, use_tanh={use_tanh}):")
    print(f"  A: {tuple(model.A.shape)}")
    print(f"  B: {tuple(model.B.shape)}")
    print(f"  C: {tuple(model.C.shape)}")
    print(f"  D (decoder): {tuple(D.shape)}")

    # Load and parse traces
    print(f"\nLoading traces from {args.data}...")
    traces = load_traces(args.data, all_aps)
    print(f"Loaded {len(traces)} traces")

    if len(traces) == 0:
        print("ERROR: No traces loaded!")
        return

    # Show first trace
    print("\nFirst trace (first 3 steps):")
    for t, step in enumerate(traces[0][:3]):
        print(f"  t={t}: {step}")

    # Split train/test
    n_test = max(1, int(len(traces) * args.test_ratio))
    n_train = len(traces) - n_test

    train_traces = traces[:n_train]
    test_traces = traces[n_train:]

    print(f"\nTrain: {len(train_traces)}, Test: {len(test_traces)}")

    # Prepare datasets with binary outputs
    X_train, Y_train, train_seq_lens = prepare_dataset(
        train_traces, input_aps, output_aps
    )
    X_test, Y_test, test_seq_lens = prepare_dataset(test_traces, input_aps, output_aps)

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")

    # Diagnostic mode
    if args.diagnose:
        diagnose_first_trace(model, D, X_train, Y_train, train_traces)
        return

    # Verify-only mode
    if args.verify_only:
        print("\n" + "=" * 70)
        print("VERIFICATION MODE (epsilon=0, use_tanh=False, exact symbolic)")
        print("=" * 70)

        model.eval()
        with torch.no_grad():
            y_train_onehot = model(X_train)
            y_test_onehot = model(X_test)

            y_train_binary = onehot_to_binary(y_train_onehot, D)
            y_test_binary = onehot_to_binary(y_test_onehot, D)

            train_mask = get_padding_mask(
                train_seq_lens, X_train.shape[0], X_train.shape[1], device
            )
            test_mask = get_padding_mask(
                test_seq_lens, X_test.shape[0], X_test.shape[1], device
            )

            train_step = masked_accuracy(y_train_binary, Y_train, train_mask)
            train_trace = trace_accuracy(y_train_binary, Y_train, train_seq_lens)
            test_step = masked_accuracy(y_test_binary, Y_test, test_mask)
            test_trace = trace_accuracy(y_test_binary, Y_test, test_seq_lens)

            print(f"Train Step Acc: {train_step:.4f}")
            print(f"Train Trace Acc: {train_trace:.4f}")
            print(f"Test Step Acc: {test_step:.4f}")
            print(f"Test Trace Acc: {test_trace:.4f}")

            if train_step < 1.0:
                print("\nWARNING: Symbolic accuracy is not perfect!")
                print("Running diagnostic on first trace...")
                diagnose_first_trace(model, D, X_train, Y_train, train_traces)
        return

    # Training mode
    print(f"\nTraining with epsilon={args.epsilon}, use_tanh={use_tanh}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize CSV logging if requested
    csv_path = args.save_csv
    if csv_path:
        init_csv_log(csv_path)
        print(f"CSV logging enabled: {csv_path}")

    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 70)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        y_onehot = model(X_train)
        y_binary = onehot_to_binary(y_onehot, D)
        mask = get_padding_mask(
            train_seq_lens, X_train.shape[0], X_train.shape[1], device
        )

        loss = masked_bce_loss(y_binary, Y_train, mask)
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            model.eval()
            with torch.no_grad():
                train_step_acc = masked_accuracy(y_binary, Y_train, mask)
                train_trace_acc = trace_accuracy(y_binary, Y_train, train_seq_lens)

                if len(test_traces) > 0:
                    y_test_onehot = model(X_test)
                    y_test_binary = onehot_to_binary(y_test_onehot, D)
                    test_mask = get_padding_mask(
                        test_seq_lens, X_test.shape[0], X_test.shape[1], device
                    )
                    test_step_acc = masked_accuracy(y_test_binary, Y_test, test_mask)
                    test_trace_acc = trace_accuracy(
                        y_test_binary, Y_test, test_seq_lens
                    )
                else:
                    test_step_acc = 0.0
                    test_trace_acc = 0.0

            print(
                f"Epoch {epoch:4d} | Loss = {loss.item():.4f} | "
                f"Train Step = {train_step_acc:.4f} | Train Trace = {train_trace_acc:.4f} | "
                f"Test Step = {test_step_acc:.4f} | Test Trace = {test_trace_acc:.4f}"
            )

            # Save to CSV if enabled
            if csv_path:
                append_csv_log(
                    csv_path,
                    epoch,
                    loss.item(),
                    train_step_acc,
                    train_trace_acc,
                    test_step_acc,
                    test_trace_acc,
                )

    # Log final epoch if not already logged
    final_epoch = args.epochs - 1
    if final_epoch % args.log_interval != 0:
        model.eval()
        with torch.no_grad():
            y_onehot = model(X_train)
            y_binary = onehot_to_binary(y_onehot, D)
            mask = get_padding_mask(
                train_seq_lens, X_train.shape[0], X_train.shape[1], device
            )
            loss = masked_bce_loss(y_binary, Y_train, mask)
            train_step_acc = masked_accuracy(y_binary, Y_train, mask)
            train_trace_acc = trace_accuracy(y_binary, Y_train, train_seq_lens)

            if len(test_traces) > 0:
                y_test_onehot = model(X_test)
                y_test_binary = onehot_to_binary(y_test_onehot, D)
                test_mask = get_padding_mask(
                    test_seq_lens, X_test.shape[0], X_test.shape[1], device
                )
                test_step_acc = masked_accuracy(y_test_binary, Y_test, test_mask)
                test_trace_acc = trace_accuracy(y_test_binary, Y_test, test_seq_lens)
            else:
                test_step_acc = 0.0
                test_trace_acc = 0.0

        print(
            f"Epoch {final_epoch:4d} | Loss = {loss.item():.4f} | "
            f"Train Step = {train_step_acc:.4f} | Train Trace = {train_trace_acc:.4f} | "
            f"Test Step = {test_step_acc:.4f} | Test Trace = {test_trace_acc:.4f}"
        )

        if csv_path:
            append_csv_log(
                csv_path,
                final_epoch,
                loss.item(),
                train_step_acc,
                train_trace_acc,
                test_step_acc,
                test_trace_acc,
            )

    print("\nTraining complete!")

    if csv_path:
        print(f"Training log saved to {csv_path}")

    if args.save:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dot_file": args.dot,
                "input_aps": input_aps,
                "output_aps": output_aps,
                "epsilon": args.epsilon,
                "use_tanh": use_tanh,
            },
            args.save,
        )
        print(f"Model saved to {args.save}")


if __name__ == "__main__":
    main()
