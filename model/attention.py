"""
model/attention.py — Rotary Positional Encoding and Grouped-Query Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RoPE helpers (inlined to avoid cross-package import issues) ───────────────

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        N = 10_000
        inv_freq = 1.0 / (N ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(max_seq_len).float()
        sinusoid = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid.cos())
        self.register_buffer("sin", sinusoid.sin())

    def forward(self, x: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return _apply_rotary_pos_emb(x, cos, sin)


class GQA(nn.Module):
    """Grouped-Query Attention with optional RoPE and additive mask."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim     = hidden_dim // num_heads
        self.dropout      = dropout

        self.q_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj   = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.v_proj   = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RotaryPositionalEncoding | None = None,
    ) -> torch.Tensor:
        B_q, S_q, H = q.shape
        B_k, S_k, _ = k.shape

        q = self.q_proj(q).view(B_q, S_q, self.num_heads,    self.head_dim)
        k = self.k_proj(k).view(B_k, S_k, self.num_kv_heads, self.head_dim)
        v = self.v_proj(v).view(B_k, S_k, self.num_kv_heads, self.head_dim)

        if rope is not None:
            q = rope(q, seq_len=S_q)
            k = rope(k, seq_len=S_k)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=True,
        )
        out = out.transpose(1, 2).reshape(B_q, S_q, H).contiguous()
        return self.out_proj(out)