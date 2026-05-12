import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from models.State_Space_Model import FSM_SSM

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser(description="Run trained FSM_SSM interactively")
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
    help="Path to saved FSM_SSM model (.pt)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="cuda or cpu",
)
args = parser.parse_args()


# ============================================================
# Load model checkpoint
# ============================================================
ckpt = torch.load(args.model_path, map_location=args.device)

input_dim = ckpt["input_dim"]
output_dim = ckpt["output_dim"]
state_dim = ckpt["state_dim"]
INPUT_APS = ckpt["input_aps"]
OUTPUT_APS = ckpt["output_aps"]

print("Loaded model")
print("Inputs :", INPUT_APS)
print("Outputs:", OUTPUT_APS)

model = FSM_SSM(
    input_dim=input_dim,
    output_dim=output_dim,
    state_dim=state_dim,
).to(args.device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()


# ============================================================
# Initialize hidden state
# ============================================================
# Assumes FSM_SSM exposes a state variable or allows manual reset
# If your FSM_SSM initializes state internally on first forward,
# you can remove this and rely on that behavior.
state = torch.zeros(1, state_dim, device=args.device)


# ============================================================
# Interactive rollout loop
# ============================================================
print("\nInteractive FSM_SSM rollout")
print("Enter inputs as comma-separated 0/1 values")
print(f"Order: {INPUT_APS}")
print("Ctrl+C to exit\n")
x_hist = torch.empty(1, 0, input_dim, device=args.device)

with torch.no_grad():
    while True:
        try:
            raw = input("> inputs: ").strip()
            parts = raw.split(",")

            if len(parts) != input_dim:
                print(f"Expected {input_dim} values")
                continue

            x = torch.tensor(
                [[float(v) for v in parts]],
                device=args.device,
            )  # (1, input_dim)

            # append timestep
            x_hist = torch.cat([x_hist, x.unsqueeze(1)], dim=1)

            y_hat = model(x_hist)
            y_logits = y_hat[:, -1, :]

            y = (y_logits > 0).int().squeeze(0).tolist()
            print("outputs:", dict(zip(OUTPUT_APS, y)))

        except KeyboardInterrupt:
            print("\nExiting.")
            break
