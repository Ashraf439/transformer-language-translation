"""
main.py — entry point.  Run with:  python main.py
"""

import os
import zipfile

import requests
import torch

import config
from dataset import build_dataloader
from inference import run_inference
from model import Transformer
from tokenizer import load_or_train
from train import run_training
from utils import normalize


def download_dataset():
    if not os.path.exists(config.DATASET_ZIP):
        print("[DATA] Downloading fra-eng.zip...")
        r = requests.get(config.DATASET_URL)
        with open(config.DATASET_ZIP, "wb") as f:
            f.write(r.content)
        print("[DATA] Download complete.")
    else:
        print("[DATA] fra-eng.zip already exists, skipping.")


def load_text_pairs() -> list[tuple[str, str]]:
    print("[DATA] Reading and normalizing text pairs...")
    pairs = []
    with zipfile.ZipFile(config.DATASET_ZIP, "r") as z:
        for line in z.read(config.DATASET_FILE).decode("utf-8").splitlines():
            pairs.append(normalize(line))
    print(f"[DATA] {len(pairs):,} pairs loaded.")
    print(f"[DATA] Sample: EN='{pairs[0][0]}' | FR='{pairs[0][1]}'")
    return pairs


def build_model(vocab_size_src: int, vocab_size_tgt: int, device: torch.device):
    model = Transformer(
        num_layers     = config.NUM_LAYERS,
        num_heads      = config.NUM_HEADS,
        num_kv_heads   = config.NUM_KV_HEADS,
        hidden_dim     = config.HIDDEN_DIM,
        max_seq_len    = config.MAX_SEQ_LEN,
        vocab_size_src = vocab_size_src,
        vocab_size_tgt = vocab_size_tgt,
        dropout        = config.DROPOUT,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable parameters: {params:,}")
    print(f"[MODEL] VRAM after load: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    return model


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("=" * 60)
    print("[INIT] English-to-French Transformer")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device}")

    download_dataset()
    text_pairs = load_text_pairs()

    en_tokenizer, fr_tokenizer = load_or_train(text_pairs)

    model = build_model(
        vocab_size_src=len(en_tokenizer.get_vocab()),
        vocab_size_tgt=len(fr_tokenizer.get_vocab()),
        device=device,
    )

    dataloader = build_dataloader(text_pairs, en_tokenizer, fr_tokenizer)

    run_training(model, dataloader, en_tokenizer, fr_tokenizer, device)

    run_inference(model, text_pairs, en_tokenizer, fr_tokenizer, device)

    print("[DONE] All done.")