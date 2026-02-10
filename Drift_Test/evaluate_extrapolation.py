#!/usr/bin/env python3
"""Evaluate MooreDotSSM extrapolation performance.

Tests how well a model trained on traces of length N performs on traces of length M > N,
specifically measuring accuracy on timesteps past the training length.

Usage:
    python evaluate_extrapolation.py \
        --model model.pt \
        --data long_traces.txt \
        --train-length 50
"""

import argparse
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# =============================================================================
# MINIMAL SSM (no DOT file needed)
# =============================================================================


class MinimalSSM(nn.Module):
    """SSM that just needs A, B, C matrices."""

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        h0: torch.Tensor,
        use_tanh: bool = False,
    ):
        super().__init__()
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)
        self.register_buffer("h0", h0)
        self.use_tanh = use_tanh

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass: inputs (B, T, num_inputs) -> outputs (B, T, num_output_symbols)"""
        batch_size, seq_len, _ = inputs.shape
        h = self.h0.unsqueeze(0).expand(batch_size, -1).clone()

        outputs = []
        for t in range(seq_len):
            sigma_t = inputs[:, t]
            mu_t = torch.einsum("bi,bj->bij", h, sigma_t).reshape(batch_size, -1)
            h = h @ self.A.T + mu_t @ self.B.T

            if self.use_tanh:
                h = torch.tanh(h)

            y_t = h @ self.C.T
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


def load_model_from_checkpoint(checkpoint_path: str, device) -> Tuple[MinimalSSM, dict]:
    """Load model from checkpoint without needing DOT file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint["model_state_dict"]
    A = state_dict["A"]
    B = state_dict["B"]
    C = state_dict["C"]
    h0 = state_dict["h0"]

    use_tanh = checkpoint.get("use_tanh", False)

    model = MinimalSSM(A, B, C, h0, use_tanh=use_tanh)
    model.to(device)
    model.eval()

    # Extract metadata
    metadata = {
        "input_aps": checkpoint.get("input_aps", []),
        "output_aps": checkpoint.get("output_aps", []),
        "epsilon": checkpoint.get("epsilon", 0.0),
        "use_tanh": use_tanh,
        "num_states": A.shape[0],
        "num_inputs": B.shape[1] // A.shape[0],
        "num_output_symbols": C.shape[0],
    }

    return model, metadata


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
        bit_pos = len(input_aps) - 1 - i
        if step.get(ap, False):
            idx |= 1 << bit_pos

    one_hot = torch.zeros(num_inputs)
    one_hot[idx] = 1.0
    return one_hot


def encode_output_binary(step: Dict[str, bool], output_aps: List[str]) -> torch.Tensor:
    """Encode output APs as binary vector."""
    return torch.tensor([float(step.get(ap, False)) for ap in output_aps])


def prepare_dataset(
    traces: List[List[Dict[str, bool]]], input_aps: List[str], output_aps: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
    """Prepare dataset from parsed traces."""
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


def build_symbol_to_binary_matrix(num_output_aps: int) -> torch.Tensor:
    """Build matrix to convert one-hot symbol to binary vector."""
    num_symbols = 2**num_output_aps
    D = torch.zeros(num_output_aps, num_symbols)

    for symbol_idx in range(num_symbols):
        for bit_idx in range(num_output_aps):
            bit_pos = num_output_aps - 1 - bit_idx
            if (symbol_idx >> bit_pos) & 1:
                D[bit_idx, symbol_idx] = 1.0

    return D


def onehot_to_binary(y_onehot: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Convert one-hot symbol logits to binary logits."""
    return y_onehot @ D.T


# =============================================================================
# EVALUATION
# =============================================================================


def compute_step_accuracy_by_position(
    y_hat: torch.Tensor, y_true: torch.Tensor, seq_lens: Dict[int, int], max_len: int
) -> Dict[int, Tuple[int, int]]:
    """Compute accuracy at each timestep position.

    Returns:
        Dict[position] -> (correct_count, total_count)
    """
    B = y_hat.shape[0]
    preds = (y_hat > 0).float()
    correct = (preds == y_true).all(dim=-1)  # (B, T)

    position_stats = {}
    for t in range(max_len):
        correct_count = 0
        total_count = 0
        for i in range(B):
            if t < seq_lens.get(i, 0):
                total_count += 1
                if correct[i, t]:
                    correct_count += 1
        if total_count > 0:
            position_stats[t] = (correct_count, total_count)

    return position_stats


def compute_range_accuracy(
    position_stats: Dict[int, Tuple[int, int]], start: int, end: int = None
) -> Tuple[float, int]:
    """Compute accuracy over a range of positions [start, end).

    Returns (acc, count).
    """
    total_correct = 0
    total_count = 0

    for pos, (correct, count) in position_stats.items():
        if pos >= start and (end is None or pos < end):
            total_correct += correct
            total_count += count

    acc = total_correct / total_count if total_count > 0 else 0.0
    return acc, total_count


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MooreDotSSM extrapolation on longer traces"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to saved model checkpoint"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Test data file with long traces (SPOT format)",
    )
    parser.add_argument(
        "--train-length",
        type=int,
        default=50,
        help="Length the model was trained on (default: 50)",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default=None,
        help="Override input APs (default: read from checkpoint)",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default=None,
        help="Override output APs (default: read from checkpoint)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-position accuracy"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Window size for accuracy breakdown (default: 10)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.model}...")
    model, metadata = load_model_from_checkpoint(args.model, device)

    print(
        f"Model: {metadata['num_states']} states, {metadata['num_inputs']} input symbols, "
        f"{metadata['num_output_symbols']} output symbols"
    )
    print(f"Config: epsilon={metadata['epsilon']}, use_tanh={metadata['use_tanh']}")

    # Get APs
    if args.inputs:
        input_aps = sorted([x.strip() for x in args.inputs.split(",") if x.strip()])
    else:
        input_aps = metadata["input_aps"]

    if args.outputs:
        output_aps = sorted([x.strip() for x in args.outputs.split(",") if x.strip()])
    else:
        output_aps = metadata["output_aps"]

    if not input_aps or not output_aps:
        parser.error(
            "APs not found in checkpoint. Please provide --inputs and --outputs."
        )

    all_aps = input_aps + output_aps
    print(f"Input APs: {input_aps}")
    print(f"Output APs: {output_aps}")

    # Build decoder matrix
    D = build_symbol_to_binary_matrix(len(output_aps)).to(device)

    # Load traces
    print(f"\nLoading traces from {args.data}...")
    traces = load_traces(args.data, all_aps)
    print(f"Loaded {len(traces)} traces")

    trace_lengths = [len(t) for t in traces]
    print(
        f"Trace lengths: min={min(trace_lengths)}, max={max(trace_lengths)}, "
        f"avg={sum(trace_lengths)/len(trace_lengths):.1f}"
    )

    # Prepare dataset
    X, Y, seq_lens = prepare_dataset(traces, input_aps, output_aps)
    X = X.to(device)
    Y = Y.to(device)

    max_len = X.shape[1]
    print(f"Data shape: X={tuple(X.shape)}, Y={tuple(Y.shape)}")

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        y_onehot = model(X)
        y_binary = onehot_to_binary(y_onehot, D)

    # Compute per-position accuracy
    position_stats = compute_step_accuracy_by_position(y_binary, Y, seq_lens, max_len)

    # Compute summary statistics
    train_length = args.train_length

    overall_acc, overall_n = compute_range_accuracy(position_stats, 0, None)
    in_dist_acc, in_dist_n = compute_range_accuracy(position_stats, 0, train_length)
    extrap_acc, extrap_n = compute_range_accuracy(position_stats, train_length, None)

    print("\n" + "=" * 60)
    print("EXTRAPOLATION RESULTS")
    print("=" * 60)
    print(f"Training length: {train_length}")
    print(f"Test trace max length: {max_len}")
    print()
    print(f"Overall step accuracy:           {overall_acc:.4f}  (n={overall_n})")
    print(
        f"In-distribution (t < {train_length}):       {in_dist_acc:.4f}  (n={in_dist_n})"
    )
    print(
        f"Extrapolation (t >= {train_length}):        {extrap_acc:.4f}  (n={extrap_n})"
    )
    print()

    # Compute accuracy in windows
    window_size = args.window
    print(f"Accuracy by position windows (size={window_size}):")
    print("-" * 50)

    for start in range(0, max_len, window_size):
        end = min(start + window_size, max_len)
        window_acc, window_n = compute_range_accuracy(position_stats, start, end)

        marker = ""
        if start < train_length <= end:
            marker = " <-- train boundary"
        elif start == train_length:
            marker = " <-- extrapolation starts"

        print(
            f"  t=[{start:3d}, {end:3d}): acc={window_acc:.4f}  (n={window_n}){marker}"
        )

    if args.verbose:
        print("\n" + "-" * 50)
        print("Per-position accuracy:")
        for t in sorted(position_stats.keys()):
            correct, total = position_stats[t]
            acc = correct / total if total > 0 else 0.0
            marker = " <-- train boundary" if t == train_length else ""
            print(f"  t={t:3d}: {acc:.4f} ({correct}/{total}){marker}")


if __name__ == "__main__":
    main()
