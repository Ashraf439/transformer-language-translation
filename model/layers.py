"""
model/layers.py — EncoderLayer and DecoderLayer (Pre-Norm, RMSNorm).
"""

import torch
import torch.nn as nn

from .attention import GQA, RotaryPositionalEncoding
from .feedforward import SwiGLU


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp       = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1     = nn.RMSNorm(hidden_dim)
        self.norm2     = nn.RMSNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RotaryPositionalEncoding | None = None,
    ) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.self_attn(n, n, n, mask, rope)
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn  = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.cross_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp        = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1      = nn.RMSNorm(hidden_dim)
        self.norm2      = nn.RMSNorm(hidden_dim)
        self.norm3      = nn.RMSNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RotaryPositionalEncoding | None = None,
    ) -> torch.Tensor:
        # Masked causal self-attention
        n = self.norm1(x)
        x = x + self.self_attn(n, n, n, mask, rope)
        # Cross-attention (no causal mask; no RoPE on encoder keys)
        n = self.norm2(x)
        x = x + self.cross_attn(n, enc_out, enc_out, None, None)
        # FFN
        x = x + self.mlp(self.norm3(x))
        return x