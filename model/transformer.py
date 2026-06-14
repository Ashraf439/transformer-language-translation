"""
model/transformer.py — top-level Encoder-Decoder Transformer.
"""

import torch
import torch.nn as nn

from .attention import RotaryPositionalEncoding
from .layers import DecoderLayer, EncoderLayer


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_dim: int,
        max_seq_len: int,
        vocab_size_src: int,
        vocab_size_tgt: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.rope          = RotaryPositionalEncoding(head_dim, max_seq_len)
        self.src_embedding = nn.Embedding(vocab_size_src, hidden_dim)
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, hidden_dim)
        self.encoders = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, num_kv_heads, dropout)
            for _ in range(num_layers)
        ])
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_dim, vocab_size_tgt)

    def encode(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.src_embedding(src_ids)
        for layer in self.encoders:
            x = layer(x, src_mask, self.rope)
        return x

    def decode(
        self,
        tgt_ids: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.tgt_embedding(tgt_ids)
        for layer in self.decoders:
            x = layer(x, enc_out, tgt_mask, self.rope)
        return self.out(x)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc_out = self.encode(src_ids, src_mask)
        return self.decode(tgt_ids, enc_out, tgt_mask)