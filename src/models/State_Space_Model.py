import torch
import torch.nn as nn


class FSM_SSM(nn.Module):
    """Moore-style SSM with explicit A/B/C matrices.

    x_embed = relu(E @ x_t) h_t = tanh(A @ h_{t-1} + B @ x_embed) y_t = C @ h_t
    """

    def __init__(self, input_dim, output_dim, state_dim=32):
        super().__init__()

        self.state_dim = state_dim
        self.embed = nn.Linear(input_dim, state_dim)

        # State transition: h_t = tanh(A @ h_{t-1} + B @ x_embed)
        self.A = nn.Parameter(0.01 * torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(0.01 * torch.randn(state_dim, state_dim))

        # Moore output: y_t = C @ h_t
        self.C = nn.Linear(state_dim, output_dim, bias=True)

        self.h0 = nn.Parameter(torch.zeros(state_dim))

    def forward(self, inputs):
        """
        inputs: (B, T, input_dim)
        returns: (B, T, output_dim)
        """
        B, T, _ = inputs.shape
        x_embed = torch.relu(self.embed(inputs))  # (B, T, state_dim)

        h = self.h0.unsqueeze(0).expand(B, -1)  # (B, state_dim)

        outputs = []
        for t in range(T):
            # h_t = tanh(A @ h_{t-1} + B @ x_embed_t)
            h = torch.tanh(h @ self.A.T + x_embed[:, t] @ self.B.T)

            # y_t = C @ h_t
            y_t = self.C(h)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    B, T, input_dim, output_dim = 4, 10, 3, 2
    model = FSM_SSM(input_dim, output_dim, state_dim=32)
    x = torch.randn(B, T, input_dim)
    y = model(x)
    print(f"FSM_SSM: input {x.shape} → output {y.shape}")
