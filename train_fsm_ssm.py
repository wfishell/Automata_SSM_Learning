import sys

import torch
import torch.nn as nn

from data_prep_ssm import prepare_datasets_mealy
from State_Space_Model import FSM_SSM

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
# LOAD DATA (full traces, no prefix expansion)
# ============================================================
X_train, Y_train, X_test, Y_test, TRAIN_SEQ_LENS, TEST_SEQ_LENS = (
    prepare_datasets_mealy(
        "Training_Dataset.txt", INPUT_APS, OUTPUT_APS, test_ratio=0.1
    )
)

X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

print("Train:", X_train.shape)
print("Test :", X_test.shape)


# ============================================================
# HOA AP ORDER
# ============================================================
def read_hoa_ap_order(hoa_file):
    with open(hoa_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("AP:"):
                parts = line.split()
                return [p.strip('"') for p in parts[2:]]
    raise ValueError("No AP: line found in HOA file")


HOA_FILE = "System.hoa"
HOA_AP_ORDER = read_hoa_ap_order(HOA_FILE)
print("HOA AP order:", HOA_AP_ORDER)

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
    """
    Returns a boolean mask where True = valid timestep, False = padding.
    X: (B, T, input_dim)
    seq_lens: dict mapping batch index to actual sequence length
    Returns: (B, T)

    If seq_lens not provided, assumes all timesteps are valid (no padding).
    """
    B, T, _ = X.shape
    if seq_lens is None:
        # Assume full sequences (no padding)
        return torch.ones(B, T, dtype=torch.bool, device=X.device)

    mask = torch.zeros(B, T, dtype=torch.bool, device=X.device)
    for i in range(B):
        length = seq_lens.get(i, T)
        mask[i, :length] = True
    return mask


def masked_bce_loss(y_hat, y_true, mask):
    """Compute BCE loss only on valid (non-padded) timesteps.

    y_hat: (B, T, output_dim) - logits
    y_true: (B, T, output_dim) - targets
    mask: (B, T) - True for valid timesteps

    Returns: scalar loss
    """
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
    """Per-timestep accuracy on valid positions.

    All output dims must match for a timestep to be correct.
    """
    preds = (y_hat > 0).float()
    correct = preds == y_true
    timestep_correct = correct.all(dim=-1)

    valid_correct = (timestep_correct & mask).sum().item()
    valid_total = mask.sum().item()

    return valid_correct / valid_total if valid_total > 0 else 0.0


def trace_accuracy(model, X, Y, seq_lens):
    """Fraction of sequences where ALL timesteps are correct."""
    model.eval()
    with torch.no_grad():
        y_hat = model(X)
        mask = get_padding_mask(X, seq_lens)

        preds = (y_hat > 0).float()
        correct = preds == Y
        timestep_correct = correct.all(dim=-1)  # (B, T)

        # A trace is correct if all valid timesteps are correct
        # Set padding positions to True so they don't affect the result
        timestep_correct = timestep_correct | ~mask
        trace_correct = timestep_correct.all(dim=-1)  # (B,)

        return trace_correct.float().mean().item()


# ============================================================
# Trace reconstruction (for HOA evaluation)
# ============================================================
def reconstruct_trace(x_seq, y_hat_seq):
    steps = []
    for t in range(x_seq.shape[0]):
        valuation = {}

        # Output predictions (logits, so >0 means True)
        output_vals = (y_hat_seq[t] > 0).int().tolist()
        for ap, v in zip(OUTPUT_APS, output_vals):
            valuation[ap] = v

        # Input values (binary 0/1, so >=0.5 or ==1 means True)
        input_vals = (x_seq[t] >= 0.5).int().tolist()
        for ap, v in zip(INPUT_APS, input_vals):
            valuation[ap] = v

        step = "&".join(
            ap if valuation.get(ap, 0) == 1 else f"!{ap}" for ap in HOA_AP_ORDER
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


# ============================================================
# Training (Mealy seq2seq: x_t → y_t at every timestep)
# ============================================================
print(f"\nTraining FSM_SSM (Mealy machine) with state_dim={state_dim}")
print("Full traces, supervising ALL timesteps: x_t → y_t")
print("=" * 70)

for epoch in range(1001):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    y_hat = model(X_train)
    mask = get_padding_mask(X_train, TRAIN_SEQ_LENS)

    # Loss over all valid timesteps
    loss = masked_bce_loss(y_hat, Y_train, mask)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            # Training metrics
            train_step_acc = masked_accuracy(y_hat, Y_train, mask)
            train_trace_acc = trace_accuracy(model, X_train, Y_train, TRAIN_SEQ_LENS)

            # Test metrics
            y_test_hat = model(X_test)
            test_mask = get_padding_mask(X_test, TEST_SEQ_LENS)
            test_step_acc = masked_accuracy(y_test_hat, Y_test, test_mask)
            test_trace_acc = trace_accuracy(model, X_test, Y_test, TEST_SEQ_LENS)

        print(
            f"Epoch {epoch:3d} | "
            f"Loss = {loss.item():.4f} | "
            f"Train Step = {train_step_acc:.4f} | "
            f"Train Trace = {train_trace_acc:.4f} | "
            f"Test Step = {test_step_acc:.4f} | "
            f"Test Trace = {test_trace_acc:.4f}"
        )

        write_epoch_test_traces(epoch, X_test, TEST_SEQ_LENS)

print("\nTraining complete!")
