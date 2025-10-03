import torch
import torch.nn as nn
from mamba_ssm import Mamba


class FSM_Mamba(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Embed concatenated [inputs + previous outputs]
        self.embed = nn.Linear(input_dim + output_dim, d_model)
        self.mamba = Mamba(d_model=d_model, n_layers=n_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, inputs, outputs=None, autoregressive=False):
        """
        inputs: [B, T, input_dim]
        outputs: [B, T, output_dim] (optional for teacher forcing)
        autoregressive: if True, roll forward using own predictions
        """
        B, T, _ = inputs.shape
        preds = []
        prev_y = torch.zeros(B, 1, self.output_dim, device=inputs.device)

        for t in range(T):
            x_t = inputs[:, t : t + 1, :]

            if outputs is not None and not autoregressive:
                # Teacher forcing: feed in true outputs
                y_t_in = outputs[:, t : t + 1, :]
            else:
                # Inference: feed in last prediction
                y_t_in = prev_y

            inp = torch.cat([x_t, y_t_in], dim=-1)  # [B,1,input_dim+output_dim]
            z = self.embed(inp)
            h = self.mamba(z)
            y_hat = torch.sigmoid(self.head(h))  # [B,1,output_dim] in [0,1]

            preds.append(y_hat)
            prev_y = y_hat.detach()  # feedback

        return torch.cat(preds, dim=1)  # [B,T,output_dim]
