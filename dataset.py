"""
dataset.py — PyTorch Dataset and DataLoader for EN→FR translation.
"""

import torch
from torch.utils.data import DataLoader, Dataset

import config


class TranslationDataset(Dataset):
    def __init__(self, text_pairs: list[tuple[str, str]]):
        self.text_pairs = text_pairs

    def __len__(self) -> int:
        return len(self.text_pairs)

    def __getitem__(self, index: int) -> tuple[str, str]:
        eng, fra = self.text_pairs[index]
        return eng, f"{config.START_TOKEN} {fra} {config.END_TOKEN}"


def build_collate_fn(en_tokenizer, fr_tokenizer):
    """Returns a collate_fn closed over the two tokenizers."""
    def collate_fn(batch):
        en_strs, fr_strs = zip(*batch)
        en_enc = en_tokenizer.encode_batch(en_strs, add_special_tokens=True)
        fr_enc = fr_tokenizer.encode_batch(fr_strs, add_special_tokens=True)
        en_ids = torch.tensor([enc.ids for enc in en_enc])
        fr_ids = torch.tensor([enc.ids for enc in fr_enc])
        return en_ids, fr_ids
    return collate_fn


def build_dataloader(
    text_pairs: list[tuple[str, str]],
    en_tokenizer,
    fr_tokenizer,
) -> DataLoader:
    dataset = TranslationDataset(text_pairs)
    collate_fn = build_collate_fn(en_tokenizer, fr_tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,      # keep 0 on Windows
        pin_memory=True,
    )
    print(f"[DATALOADER] {len(dataset):,} samples | "
          f"batch={config.BATCH_SIZE} | {len(loader)} batches/epoch")
    return loader