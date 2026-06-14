"""
model/feedforward.py — SwiGLU feed-forward network.
"""

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """SwiGLU(x) = down( SiLU(gate(x)) * up(x) )"""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        self.up   = nn.Linear(hidden_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))