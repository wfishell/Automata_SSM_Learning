# TODO
# BUILD SSM
import torch  # noqa: F401
import torch.nn as nn
from mamba_ssm import Mamba  # noqa: F401


class FSM_Mamba(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.mamba = Mamba(d_model=d_model, n_layers=n_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: [B, T, input_dim]
        z = self.embed(x)
        h = self.mamba(z)  # [B, T, d_model]
        y = self.head(h)  # [B, T, output_dim]
        return y


if __name__ == "__main__":
    model = FSM_Mamba(input_dim=10, output_dim=5)  # Example usage
