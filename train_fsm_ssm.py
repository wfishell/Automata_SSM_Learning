import torch
import torch.nn as nn

from data_prep_ssm import prepare_datasets
from State_Space_Model import FSM_Mamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# DYNAMIC CONFIGURATION - CHANGE THESE AS NEEDED
INPUT_APS = ["a", "b"]  # Can add more: ["go", "cancel", "req", "start", "stop"]
OUTPUT_APS = ["p0", "p1"]

# Load datasets with dynamic inputs/outputs
X_train, Y_train, X_test, Y_test = prepare_datasets(
    "Training_Dataset.txt", INPUT_APS, OUTPUT_APS, test_ratio=0.2
)

# Move tensors to GPU
X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)

# Define model with dynamic dimensions
input_dim = len(INPUT_APS)
output_dim = len(OUTPUT_APS)

model = FSM_Mamba(input_dim=input_dim, output_dim=output_dim).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
print(f"Training with {input_dim} inputs and {output_dim} outputs...")
print(f"Inputs: {INPUT_APS}")
print(f"Outputs: {OUTPUT_APS}")
print()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    model.reset_hidden(batch_size=X_train.shape[0])
    y_hat = model(X_train, autoregressive=False)
    loss = criterion(y_hat, Y_train)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

print("\nTraining complete!\n")

# EVALUATION WITH DYNAMIC OUTPUTS
print("=" * 70)
print("EVALUATION - DYNAMIC MULTI-OUTPUT SYSTEM")
print("=" * 70)
print(f"\nInput signals: {INPUT_APS}")
print(f"Output signals: {OUTPUT_APS}")
print("=" * 70 + "\n")

# ======================================================
# GENERATE TRACES FROM TRAINED MODEL IN SPOT SEMANTICS
# ======================================================

output_file = "Generated_Predicted_Traces_SPOT.txt"

with open(output_file, "w") as f:
    model.eval()
    with torch.no_grad():
        num_sequences = min(50, X_test.shape[0])  # adjust as needed

        for seq_idx in range(num_sequences):
            X_seq = X_test[seq_idx]

            trace_atoms = []
            hidden = None

            seq_len = X_seq.shape[0]

            for t in range(seq_len):
                if X_seq[t].sum() == 0:
                    break  # stop if padding

                # One timestep input
                input_t = X_seq[t].unsqueeze(0)

                # Predict output with trained SSM
                output_t, hidden = model.forward_step(input_t, hidden)

                # Convert to binary
                pred_outputs = [
                    1 if v > 0.5 else 0 for v in output_t.cpu().numpy().flatten()
                ]

                atoms = []

                # Input atoms
                for i, name in enumerate(INPUT_APS):
                    if X_seq[t, i] > 0.5:
                        atoms.append(name)
                    else:
                        atoms.append(f"!{name}")

                # Predicted output atoms
                for i, name in enumerate(OUTPUT_APS):
                    if pred_outputs[i] == 1:
                        atoms.append(name)
                    else:
                        atoms.append(f"!{name}")

                trace_atoms.append("&".join(atoms))

            if trace_atoms:
                # Join timesteps with ; and add cycle{1}
                trace_line = ";".join(trace_atoms) + ";cycle{1}"
                f.write(trace_line + "\n")

print(f"Predicted SPOT traces written to {output_file}")
