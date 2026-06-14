"""
utils.py — text normalization, RoPE math helpers, mask builders.
"""

import unicodedata
import torch


# ── Text ──────────────────────────────────────────────────────

def normalize(line: str) -> tuple[str, str]:
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()


# ── RoPE math ─────────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


# ── Masks ─────────────────────────────────────────────────────

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular -inf mask; shape (seq_len, seq_len)."""
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )


def create_padding_mask(
    batch: torch.Tensor,
    padding_token_id: int,
) -> torch.Tensor:
    """
    Additive padding mask; shape (B, 1, seq_len, seq_len).
    Pad positions → -inf, everything else → 0.
    """
    device = batch.device
    B, S = batch.shape
    padded = torch.zeros(B, S, device=device, dtype=torch.float).masked_fill(
        batch == padding_token_id, float("-inf")
    )
    mask = padded[:, :, None] + padded[:, None, :]   # (B, S, S)
    return mask[:, None, :, :]                         # (B, 1, S, S)