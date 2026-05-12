import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data.automaton_state_tracker import parse_hoa_transitions, simulate_trace
from data.data_prep_ssm import prepare_datasets_mealy
from models.State_Space_Model import FSM_SSM

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# AP configuration
# ============================================================
INPUT_APS = sys.argv[1].split(",")
OUTPUT_APS = sys.argv[2].split(",")

print(f"Inputs:  {INPUT_APS}")
print(f"Outputs: {OUTPUT_APS}")

input_dim = len(INPUT_APS)
output_dim = len(OUTPUT_APS)

# ============================================================
# LOAD DATA
# ============================================================
TRACE_FILE = "Training_Dataset.txt"

X_train, Y_train, X_test, Y_test, TRAIN_SEQ_LENS, TEST_SEQ_LENS = (
    prepare_datasets_mealy(TRACE_FILE, INPUT_APS, OUTPUT_APS, test_ratio=0.1)
)

X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

print("Train:", X_train.shape)
print("Test :", X_test.shape)


# ============================================================
# HOA Parsing
# ============================================================
def parse_hoa(hoa_file):
    """Parse HOA file to get state count and AP order."""
    with open(hoa_file, "r") as f:
        content = f.read()

    n_states = int(re.search(r"States:\s*(\d+)", content).group(1))
    start_state = int(re.search(r"Start:\s*(\d+)", content).group(1))

    ap_match = re.search(r"AP:\s*\d+\s+(.*)", content)
    ap_names = re.findall(r'"([^"]+)"', ap_match.group(1))

    return n_states, start_state, ap_names


HOA_FILE = "System.hoa"
N_HOA_STATES, HOA_START, HOA_AP_NAMES = parse_hoa(HOA_FILE)
print(f"HOA: {N_HOA_STATES} states, start={HOA_START}, APs={HOA_AP_NAMES}")

N_AUTOMATON_STATES = N_HOA_STATES


# ============================================================
# Get HOA state sequences using automaton_state_tracker functions
# ============================================================


def get_hoa_state_sequences(raw_traces, hoa_file):
    """Simulate HOA automaton on traces and return state sequences.

    Returns:
        List of state sequences, one per trace
    """
    # Parse HOA once
    n_states, start_state, ap_names, transitions = parse_hoa_transitions(hoa_file)

    # Simulate each trace
    state_sequences = []
    for trace in raw_traces:
        states = simulate_trace(trace, start_state, ap_names, transitions)
        state_sequences.append(states)

    return state_sequences


# ============================================================
# Model
# ============================================================
state_dim = 32

model = FSM_SSM(input_dim=input_dim, output_dim=output_dim, state_dim=state_dim).to(
    device
)

criterion = nn.BCEWithLogitsLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ============================================================
# Masking utilities
# ============================================================
def get_padding_mask(X, seq_lens=None):
    B, T, _ = X.shape
    if seq_lens is None:
        return torch.ones(B, T, dtype=torch.bool, device=X.device)
    mask = torch.zeros(B, T, dtype=torch.bool, device=X.device)
    for i in range(B):
        length = seq_lens.get(i, T)
        mask[i, :length] = True
    return mask


def masked_bce_loss(y_hat, y_true, mask):
    mask_expanded = mask.unsqueeze(-1).expand_as(y_hat)
    loss_per_elem = criterion(y_hat, y_true)
    masked_loss = loss_per_elem * mask_expanded.float()
    total_loss = masked_loss.sum()
    count = mask_expanded.float().sum()
    return total_loss / count if count > 0 else total_loss


# ============================================================
# Accuracy metrics
# ============================================================
def masked_accuracy(y_hat, y_true, mask):
    preds = (y_hat > 0).float()
    correct = preds == y_true
    timestep_correct = correct.all(dim=-1)
    valid_correct = (timestep_correct & mask).sum().item()
    valid_total = mask.sum().item()
    return valid_correct / valid_total if valid_total > 0 else 0.0


def trace_accuracy(model, X, Y, seq_lens):
    model.eval()
    with torch.no_grad():
        y_hat = model(X)
        mask = get_padding_mask(X, seq_lens)
        preds = (y_hat > 0).float()
        correct = preds == Y
        timestep_correct = correct.all(dim=-1)
        timestep_correct = timestep_correct | ~mask
        trace_correct = timestep_correct.all(dim=-1)
        return trace_correct.float().mean().item()


# ============================================================
# Trace reconstruction (for HOA evaluation)
# ============================================================
def reconstruct_trace(x_seq, y_hat_seq):
    steps = []
    for t in range(x_seq.shape[0]):
        valuation = {}

        output_vals = (y_hat_seq[t] > 0).int().tolist()
        for ap, v in zip(OUTPUT_APS, output_vals):
            valuation[ap] = v

        input_vals = (x_seq[t] >= 0.5).int().tolist()
        for ap, v in zip(INPUT_APS, input_vals):
            valuation[ap] = v

        step = "&".join(
            ap if valuation.get(ap, 0) == 1 else f"!{ap}" for ap in HOA_AP_NAMES
        )
        steps.append(step)

    return ";".join(steps) + ";cycle{1}"


def write_epoch_test_traces(epoch, X_test, seq_lens):
    model.eval()
    filename = f"epoch_{epoch}_test_eval.txt"

    with torch.no_grad(), open(filename, "w") as f:
        y_hat = model(X_test)

        for i in range(X_test.shape[0]):
            seq_len = seq_lens[i]
            x_seq = X_test[i, :seq_len]
            y_seq = y_hat[i, :seq_len]
            symbolic_trace = reconstruct_trace(x_seq, y_seq)
            f.write(symbolic_trace + "\n")

    return filename


# ============================================================
# State extraction (includes initial state h0)
# ============================================================
def extract_states_per_trace(model, X, seq_lens=None):
    """Extract hidden states organized by trace, including initial state."""
    model.eval()
    B, T, _ = X.shape

    with torch.no_grad():
        x_embed = torch.relu(model.embed(X))
        h = model.h0.unsqueeze(0).expand(B, -1)

        # Start with initial state
        trace_states = [[h[b].cpu().numpy()] for b in range(B)]

        for t in range(T):
            h = torch.tanh(h @ model.A.T + x_embed[:, t] @ model.B.T)

            for b in range(B):
                max_t = seq_lens.get(b, T) if seq_lens else T
                if t < max_t:
                    trace_states[b].append(h[b].cpu().numpy())

        trace_states = [np.array(s) for s in trace_states]
        return trace_states


# ============================================================
# K-means clustering
# ============================================================
def cluster_states_kmeans(all_trace_states, n_clusters):
    """Cluster all states from all traces."""
    all_states = np.vstack(all_trace_states)

    scaler = StandardScaler()
    states_scaled = scaler.fit_transform(all_states)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(states_scaled)

    return kmeans, scaler


def get_cluster_sequence(trace_states, kmeans, scaler):
    """Get cluster assignments for a trace."""
    if len(trace_states) == 0:
        return []
    states_scaled = scaler.transform(trace_states)
    return kmeans.predict(states_scaled).tolist()


# ============================================================
# State comparison with Hungarian algorithm
# ============================================================
def compute_state_mapping(cluster_sequences, hoa_sequences, n_clusters, n_hoa_states):
    """Find optimal cluster -> HOA state mapping."""
    cooccur = np.zeros((n_clusters, n_hoa_states))

    for clusters, states in zip(cluster_sequences, hoa_sequences):
        min_len = min(len(clusters), len(states))
        for i in range(min_len):
            c, s = clusters[i], states[i]
            if s >= 0 and c >= 0:
                cooccur[c, s] += 1

    row_ind, col_ind = linear_sum_assignment(-cooccur)
    mapping = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}

    # Compute accuracy
    total = 0
    correct = 0
    for clusters, states in zip(cluster_sequences, hoa_sequences):
        min_len = min(len(clusters), len(states))
        for i in range(min_len):
            c, s = clusters[i], states[i]
            if s >= 0 and c >= 0:
                total += 1
                if mapping.get(c, -1) == s:
                    correct += 1

    accuracy = correct / total if total > 0 else 0
    return mapping, accuracy, cooccur


def write_state_comparison(
    epoch,
    cluster_sequences,
    hoa_sequences,
    raw_traces,
    mapping,
    accuracy,
    filename=None,
):
    """Write detailed comparison with generated traces."""
    if filename is None:
        filename = f"epoch_{epoch}_state_comparison.txt"

    with open(filename, "w") as f:
        f.write(f"Epoch {epoch} State Comparison\n")
        f.write("=" * 80 + "\n\n")
        f.write("Cluster -> HOA State Mapping:\n")
        for c in sorted(mapping.keys()):
            f.write(f"  Cluster {c:2d} -> State {mapping[c]:2d}\n")
        f.write(f"\nOverall State Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n")
        f.write("=" * 80 + "\n\n")

        n_perfect = 0
        for i, (clusters, states, trace) in enumerate(
            zip(cluster_sequences, hoa_sequences, raw_traces)
        ):
            min_len = min(len(clusters), len(states))
            mapped = [mapping.get(c, -1) for c in clusters[:min_len]]
            matches = sum(
                1 for m, s in zip(mapped, states[:min_len]) if m == s and s >= 0
            )
            total = sum(1 for s in states[:min_len] if s >= 0)
            trace_acc = matches / total if total > 0 else 0

            if trace_acc == 1.0:
                n_perfect += 1

            if i < 20:  # First 20 traces in detail
                f.write(f"Trace {i:3d} (acc={trace_acc:.2f}):\n")
                f.write(
                    f"  Raw: {trace[:80]}...\n"
                    if len(trace) > 80
                    else f"  Raw: {trace}\n"
                )
                f.write(
                    f"  HOA:     {' -> '.join(f'{s:2d}' for s in states[:min_len])}\n"
                )
                f.write(
                    f"  Cluster: {' -> '.join(f'{c:2d}' for c in clusters[:min_len])}\n"
                )
                f.write(f"  Mapped:  {' -> '.join(f'{m:2d}' for m in mapped)}\n")

                mismatches = [
                    t
                    for t, (m, s) in enumerate(zip(mapped, states[:min_len]))
                    if m != s and s >= 0
                ]
                if mismatches:
                    f.write(f"  Errors at positions: {mismatches}\n")
                f.write("\n")

        f.write(
            f"\nSummary: {n_perfect}/{len(cluster_sequences)} traces with perfect state tracking\n"
        )


# ============================================================
# Training
# ============================================================
print(f"\nTraining FSM_SSM (Mealy machine) with state_dim={state_dim}")
print(
    f"Will compare against {N_AUTOMATON_STATES} HOA states when trace accuracy = 100%"
)
print("=" * 70)

for epoch in range(5000):
    model.train()
    optimizer.zero_grad()

    y_hat = model(X_train)
    mask = get_padding_mask(X_train, TRAIN_SEQ_LENS)

    loss = masked_bce_loss(y_hat, Y_train, mask)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            train_step_acc = masked_accuracy(y_hat, Y_train, mask)
            train_trace_acc = trace_accuracy(model, X_train, Y_train, TRAIN_SEQ_LENS)

            y_test_hat = model(X_test)
            test_mask = get_padding_mask(X_test, TEST_SEQ_LENS)
            test_step_acc = masked_accuracy(y_test_hat, Y_test, test_mask)
            test_trace_acc = trace_accuracy(model, X_test, Y_test, TEST_SEQ_LENS)

        print(
            f"Epoch {epoch:4d} | "
            f"Loss = {loss.item():.4f} | "
            f"Train Step = {train_step_acc:.4f} | "
            f"Train Trace = {train_trace_acc:.4f} | "
            f"Test Step = {test_step_acc:.4f} | "
            f"Test Trace = {test_trace_acc:.4f}"
        )

        # Write generated predictions to file
        generated_trace_file = write_epoch_test_traces(epoch, X_test, TEST_SEQ_LENS)

        # --------------------------------------------------------
        # Do state comparison when test trace accuracy >= 50%
        # Only analyze traces that are fully correct (accepted)
        # --------------------------------------------------------
        if test_trace_acc >= 0.5:
            # Find which traces are fully correct
            with torch.no_grad():
                y_test_hat = model(X_test)
                preds = (y_test_hat > 0).float()
                correct = preds == Y_test
                timestep_correct = correct.all(dim=-1)  # (B, T)

                # Mask out padding
                test_mask = get_padding_mask(X_test, TEST_SEQ_LENS)
                timestep_correct = timestep_correct | ~test_mask
                trace_correct = timestep_correct.all(dim=-1)  # (B,)

                accepted_indices = torch.where(trace_correct)[0].tolist()

            if len(accepted_indices) == 0:
                print("         | No accepted traces to analyze")
                continue

            # Read the generated traces from the epoch file
            with open(generated_trace_file, "r") as f:
                all_generated_traces = [line.strip() for line in f if line.strip()]

            # Filter to only accepted traces
            generated_traces = [all_generated_traces[i] for i in accepted_indices]

            # Get HOA state sequences for the accepted GENERATED traces
            hoa_state_sequences = get_hoa_state_sequences(generated_traces, HOA_FILE)

            if len(hoa_state_sequences) != len(generated_traces):
                print(
                    f"         | WARNING: HOA returned {len(hoa_state_sequences)} sequences for {len(generated_traces)} traces"
                )
                continue

            # Extract SSM states only for accepted traces
            trace_states_all = extract_states_per_trace(model, X_test, TEST_SEQ_LENS)
            trace_states = [trace_states_all[i] for i in accepted_indices]

            # Cluster with k = number of HOA states
            kmeans, scaler = cluster_states_kmeans(trace_states, N_AUTOMATON_STATES)

            cluster_sequences = [
                get_cluster_sequence(ts, kmeans, scaler) for ts in trace_states
            ]

            # Compare clusters to HOA states
            mapping, state_accuracy, cooccur = compute_state_mapping(
                cluster_sequences, hoa_state_sequences, N_AUTOMATON_STATES, N_HOA_STATES
            )

            print(
                f"         | "
                f"Accepted = {len(accepted_indices)}/{X_test.shape[0]} | "
                f"State Accuracy = {state_accuracy:.4f} | "
                f"Mapping: {dict(sorted(mapping.items()))}"
            )

            # Write comparison with generated traces
            write_state_comparison(
                epoch,
                cluster_sequences,
                hoa_state_sequences,
                generated_traces,
                mapping,
                state_accuracy,
            )

print("\nTraining complete!")
