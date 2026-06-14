"""
train.py — training loop with warmup + cosine LR schedule.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import config
from utils import create_causal_mask, create_padding_mask


def _build_optimizer_and_scheduler(model: nn.Module, steps_per_epoch: int):
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.WARMUP_STEPS
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.N_EPOCHS * steps_per_epoch - config.WARMUP_STEPS,
        eta_min=0,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.WARMUP_STEPS]
    )
    return optimizer, scheduler


def run_training(model, dataloader, en_tokenizer, fr_tokenizer, device):
    pad_id_en = en_tokenizer.token_to_id(config.PAD_TOKEN)
    pad_id_fr = fr_tokenizer.token_to_id(config.PAD_TOKEN)

    loss_fn              = nn.CrossEntropyLoss(ignore_index=pad_id_fr)
    optimizer, scheduler = _build_optimizer_and_scheduler(model, len(dataloader))

    total_steps = config.N_EPOCHS * len(dataloader)
    print(f"[TRAINING] Epochs={config.N_EPOCHS} | LR={config.LR} | "
          f"Warmup={config.WARMUP_STEPS} | Total steps={total_steps:,}")

    for epoch in range(config.N_EPOCHS):
        model.train()
        epoch_loss = 0.0

        loop = tqdm.tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1:>3}/{config.N_EPOCHS}",
            leave=True,
        )
        for batch_idx, (en_ids, fr_ids) in enumerate(loop):
            en_ids = en_ids.to(device, non_blocking=True)
            fr_ids = fr_ids.to(device, non_blocking=True)

            src_mask = create_padding_mask(en_ids, pad_id_en)
            tgt_mask = (
                create_causal_mask(fr_ids.shape[1], device).unsqueeze(0)
                + create_padding_mask(fr_ids, pad_id_fr)
            )

            optimizer.zero_grad()
            outputs = model(en_ids, fr_ids, src_mask, tgt_mask)
            loss = loss_fn(
                outputs[:, :-1, :].reshape(-1, outputs.shape[-1]),
                fr_ids[:, 1:].reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.CLIP_NORM, error_if_nonfinite=False
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{epoch_loss / (batch_idx + 1):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.6f}",
            )

        avg = epoch_loss / len(dataloader)
        print(f"[EPOCH {epoch + 1:>3}/{config.N_EPOCHS}] "
              f"avg_loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.6f} | "
              f"VRAM={torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n[TRAINING] Training complete.")