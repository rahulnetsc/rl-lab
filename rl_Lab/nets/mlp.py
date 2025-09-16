import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__()
        self.in_dim = in_dim # feature dim
        self.out_dim = out_dim # output dim or num_actions
        self._layers  = nn.ModuleList([
            nn.Linear(self.in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        ])

    def forward(self, x):

        assert x.shape[-1] == self.in_dim, f"Input feature dim: {x.shape[-1]} does not match mlp in_dim: {self.in_dim}"

        for layer in self._layers:
            x = layer(x)
        return x 

