import random
import os
import unicodedata
import zipfile

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tokenizers
from torch.utils.data import DataLoader
import tqdm

# ── Helper functions ──────────────────────────────────────────
def normalize(line):
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

def create_causal_mask(seq_len, device):
    return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

def create_padding_mask(batch, padding_token_id):
    batch_size, seq_len = batch.shape
    device = batch.device
    padded = torch.zeros_like(batch, device=device).float().masked_fill(
        batch == padding_token_id, float("-inf"))
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device) + \
           padded[:, :, None] + padded[:, None, :]
    return mask[:, None, :, :]

# ── Model classes ─────────────────────────────────────────────
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)


class GQA(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // self.num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None, rope=None):
        q_batch_size, q_seq_len, hidden_dim = q.shape
        k_batch_size, k_seq_len, _ = k.shape
        v_batch_size, v_seq_len, _ = v.shape

        q = self.q_proj(q).view(q_batch_size, q_seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(k_batch_size, k_seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(v).view(v_batch_size, v_seq_len, self.num_kv_heads, self.head_dim)

        if rope is not None:
            q = rope(q, seq_len=q_seq_len)
            k = rope(k, seq_len=k_seq_len)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=True
        )
        output = output.transpose(1, 2).reshape(q_batch_size, q_seq_len, hidden_dim).contiguous()
        return self.out_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, mask=None, rope=None):
        normed = self.norm1(x)
        out = self.self_attn(normed, normed, normed, mask, rope)
        x = out + x
        out = self.mlp(self.norm2(x))
        return out + x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.cross_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.norm3 = nn.RMSNorm(hidden_dim)

    def forward(self, x, enc_out, mask=None, rope=None):
        normed = self.norm1(x)
        out = self.self_attn(normed, normed, normed, mask, rope)
        x = out + x
        normed = self.norm2(x)
        out = self.cross_attn(normed, enc_out, enc_out, None, rope)
        x = out + x
        out = self.mlp(self.norm3(x))
        return out + x


class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 max_seq_len, vocab_size_src, vocab_size_tgt, dropout=0.1):
        super().__init__()
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
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

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        x = self.src_embedding(src_ids)
        for encoder in self.encoders:
            x = encoder(x, src_mask, self.rope)
        enc_out = x
        x = self.tgt_embedding(tgt_ids)
        for decoder in self.decoders:
            x = decoder(x, enc_out, tgt_mask, self.rope)
        return self.out(x)


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, index):
        eng, fra = self.text_pairs[index]
        return eng, "[start] " + fra + " [end]"


# ── Everything below runs only in the main process ────────────
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("=" * 60)
    print("[INIT] Starting English-to-French Transformer")
    print("=" * 60)

    # ── Dataset ───────────────────────────────────────────────
    print("\n[DATA] Checking for dataset...")
    if not os.path.exists("fra-eng.zip"):
        print("[DATA] fra-eng.zip not found — downloading...")
        url = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
        response = requests.get(url)
        with open("fra-eng.zip", "wb") as f:
            f.write(response.content)
        print("[DATA] Download complete.")
    else:
        print("[DATA] fra-eng.zip already exists, skipping download.")

    print("[DATA] Reading and normalizing text pairs...")
    text_pairs = []
    with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
        for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
            eng, fra = normalize(line)
            text_pairs.append((eng, fra))
    print(f"[DATA] Loaded {len(text_pairs):,} sentence pairs.")
    print(f"[DATA] Sample pair: EN='{text_pairs[0][0]}' | FR='{text_pairs[0][1]}'")

    # ── Tokenizers ────────────────────────────────────────────
    print("\n[TOKENIZER] Checking for saved tokenizers...")
    if os.path.exists("en_tokenizer.json") and os.path.exists("fr_tokenizer.json"):
        print("[TOKENIZER] Found saved tokenizers — loading from disk.")
        en_tokenizer = tokenizers.Tokenizer.from_file("en_tokenizer.json")
        fr_tokenizer = tokenizers.Tokenizer.from_file("fr_tokenizer.json")
        en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
        fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
        print("[TOKENIZER] Padding re-enabled after loading.")
    else:
        print("[TOKENIZER] No saved tokenizers found — training from scratch.")
        en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
        fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
        en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
        fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
        en_tokenizer.decoder = tokenizers.decoders.ByteLevel()
        fr_tokenizer.decoder = tokenizers.decoders.ByteLevel()

        VOCAB_SIZE = 8000
        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=["[start]", "[end]", "[pad]"],
            show_progress=True
        )
        print("[TOKENIZER] Training English BPE tokenizer...")
        en_tokenizer.train_from_iterator([x[0] for x in text_pairs], trainer=trainer)
        print("[TOKENIZER] Training French BPE tokenizer...")
        fr_tokenizer.train_from_iterator([x[1] for x in text_pairs], trainer=trainer)
        en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
        fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
        en_tokenizer.save("en_tokenizer.json", pretty=True)
        fr_tokenizer.save("fr_tokenizer.json", pretty=True)
        print("[TOKENIZER] Saved tokenizers to disk.")

    print(f"[TOKENIZER] EN vocab size: {len(en_tokenizer.get_vocab()):,}")
    print(f"[TOKENIZER] FR vocab size: {len(fr_tokenizer.get_vocab()):,}")

    # ── Model ─────────────────────────────────────────────────
    print("\n[MODEL] Building Transformer...")
    model_config = {
        "num_layers": 4,
        "num_heads": 8,
        "num_kv_heads": 4,
        "hidden_dim": 128,
        "max_seq_len": 768,
        "vocab_size_src": len(en_tokenizer.get_vocab()),
        "vocab_size_tgt": len(fr_tokenizer.get_vocab()),
        "dropout": 0.1,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MODEL] Device: {device}")
    model = Transformer(**model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Config: {model_config}")
    print(f"[MODEL] Total trainable parameters: {total_params:,}")
    print(f"[MODEL] VRAM after model load: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

    # ── DataLoader ────────────────────────────────────────────
    print("\n[DATALOADER] Setting up dataset and dataloader...")

    def collate_fn(batch):
        en_str, fr_str = zip(*batch)
        en_enc = en_tokenizer.encode_batch(en_str, add_special_tokens=True)
        fr_enc = fr_tokenizer.encode_batch(fr_str, add_special_tokens=True)
        en_ids = [enc.ids for enc in en_enc]
        fr_ids = [fra.ids for fra in fr_enc]
        return torch.tensor(en_ids), torch.tensor(fr_ids)

    BATCH_SIZE = 64
    dataset = TranslationDataset(text_pairs)
    dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
    print(f"[DATALOADER] Dataset size: {len(dataset):,} samples")
    print(f"[DATALOADER] Batch size: {BATCH_SIZE} | Batches per epoch: {len(dataloader)}")

    # ── Training setup ────────────────────────────────────────
    N_EPOCHS = 60
    LR = 0.005
    WARMUP_STEPS = 1000
    CLIP_NORM = 5.0

    print("\n[TRAINING] Setting up optimizer and schedulers...")
    print(f"[TRAINING] Epochs: {N_EPOCHS} | LR: {LR} | Warmup: {WARMUP_STEPS} steps | Clip: {CLIP_NORM}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=fr_tokenizer.token_to_id("[pad]"))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS * len(dataloader) - WARMUP_STEPS, eta_min=0)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_STEPS])

    total_steps = N_EPOCHS * len(dataloader)
    print(f"[TRAINING] Total steps: {total_steps:,} (warmup for first {WARMUP_STEPS})")
    print("\n[TRAINING] Starting training loop...\n")

    # ── Training loop ─────────────────────────────────────────
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0

        loop = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1:>3}/{N_EPOCHS}", leave=True)
        for batch_idx, (en_ids, fr_ids) in enumerate(loop):
            en_ids = en_ids.to(device, non_blocking=True)
            fr_ids = fr_ids.to(device, non_blocking=True)

            src_mask = create_padding_mask(en_ids, en_tokenizer.token_to_id("[pad]"))
            tgt_mask = create_causal_mask(fr_ids.shape[1], device).unsqueeze(0)
            tgt_mask = tgt_mask + create_padding_mask(fr_ids, fr_tokenizer.token_to_id("[pad]"))

            optimizer.zero_grad()
            outputs = model(en_ids, fr_ids, src_mask, tgt_mask)
            loss = loss_fn(
                outputs[:, :-1, :].reshape(-1, outputs.shape[-1]),
                fr_ids[:, 1:].reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=False)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{epoch_loss / (batch_idx + 1):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.6f}"
            )

        avg_loss = epoch_loss / len(dataloader)
        print(f"[EPOCH {epoch+1:>3}/{N_EPOCHS}] avg_loss={avg_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f} | "
              f"VRAM={torch.cuda.memory_allocated() / 1e9:.2f}GB")

    print("\n[TRAINING] Training complete.")

    # ── Inference ─────────────────────────────────────────────
    print("\n[INFERENCE] Switching model to eval mode...")
    model.eval()
    N_SAMPLES = 5
    MAX_LEN = 60
    print(f"[INFERENCE] Generating translations for {N_SAMPLES} random samples (max_len={MAX_LEN})\n")

    with torch.no_grad():
        start_token = torch.tensor([fr_tokenizer.token_to_id("[start]")]).to(device)

        for i, (en, true_fr) in enumerate(random.sample(text_pairs, N_SAMPLES), 1):
            print(f"[INFERENCE] Sample {i}/{N_SAMPLES}")
            print(f"  EN : '{en}'")

            en_ids = torch.tensor(en_tokenizer.encode(en).ids).unsqueeze(0).to(device)
            src_mask = create_padding_mask(en_ids, en_tokenizer.token_to_id("[pad]"))

            x = model.src_embedding(en_ids)
            for encoder in model.encoders:
                x = encoder(x, src_mask, model.rope)
            enc_out = x

            fr_ids = start_token.unsqueeze(0)
            for step in range(MAX_LEN):
                tgt_mask = create_causal_mask(fr_ids.shape[1], device).unsqueeze(0)
                tgt_mask = tgt_mask + create_padding_mask(fr_ids, fr_tokenizer.token_to_id("[pad]"))
                x = model.tgt_embedding(fr_ids)
                for decoder in model.decoders:
                    x = decoder(x, enc_out, tgt_mask, model.rope)
                next_token = model.out(x).argmax(dim=-1)[:, -1:]
                fr_ids = torch.cat([fr_ids, next_token], dim=-1)
                if fr_ids[0, -1] == fr_tokenizer.token_to_id("[end]"):
                    print(f"  [end] hit at step {step+1}")
                    break
            else:
                print(f"  max_len={MAX_LEN} reached without [end]")

            pred_fr = fr_tokenizer.decode(fr_ids[0].tolist())
            print(f"  FR : '{true_fr}'")
            print(f"  PRD: '{pred_fr}'")
            print()

    print("[DONE] All done.")