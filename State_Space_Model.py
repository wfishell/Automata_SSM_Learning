import torch
import torch.nn as nn
from mamba_ssm import Mamba


class FSM_Mamba(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        self.embed = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                for _ in range(n_layers)
            ]
        )
        self.fc = nn.Linear(d_model, output_dim)
        self.activation = nn.Sigmoid()  # since output is binary

        # Store hidden states for stateful processing
        self.hidden_states = None

    def reset_hidden(self, batch_size=1):
        """Reset hidden states for new sequence processing."""
        self.hidden_states = None

    def forward(self, inputs, autoregressive=False):
        """Forward pass that maintains state across timesteps.

        Args:
            inputs: (B, T, input_dim) - Input sequences
            autoregressive: bool - Whether to use autoregressive mode (for compatibility)

        Returns:
            outputs: (B, T, output_dim) - Predicted outputs
        """
        B, T, _ = inputs.shape
        device = inputs.device

        # Initialize outputs
        outputs = []

        # Initialize hidden state if needed
        if self.hidden_states is None:
            h = torch.zeros(B, 1, self.d_model).to(device)
        else:
            h = self.hidden_states

        # Process sequence step by step
        for t in range(T):
            # Get current input
            x_t = inputs[:, t : t + 1, :]  # (B, 1, input_dim)

            # Embed input
            x_t = self.embed(x_t)  # (B, 1, d_model)

            # Process through Mamba layers with hidden state
            for layer in self.layers:
                # Mamba maintains its own internal state
                x_t = layer(x_t)

            # Update hidden state
            h = x_t

            # Predict output for this timestep
            y_t = self.activation(self.fc(x_t[:, -1, :]))  # (B, output_dim)
            outputs.append(y_t.unsqueeze(1))  # (B, 1, output_dim)

        # Store final hidden state for potential continued processing
        self.hidden_states = h

        return torch.cat(outputs, dim=1)  # (B, T, output_dim)

    def forward_step(self, input_t, hidden=None):
        """Single step forward for true autoregressive/online prediction.

        Args:
            input_t: (B, input_dim) - Single timestep input
            hidden: Optional hidden state from previous step

        Returns:
            output_t: (B, output_dim) - Single timestep output
            hidden: Updated hidden state
        """
        B = input_t.shape[0]
        device = input_t.device

        # Initialize hidden if needed
        if hidden is None:
            hidden = torch.zeros(B, 1, self.d_model).to(device)

        # Add time dimension
        x_t = input_t.unsqueeze(1)  # (B, 1, input_dim)

        # Embed
        x_t = self.embed(x_t)

        # Process through layers
        for layer in self.layers:
            x_t = layer(x_t)

        # Get output
        output_t = self.activation(self.fc(x_t[:, -1, :]))

        return output_t, x_t
