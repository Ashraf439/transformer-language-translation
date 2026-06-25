# TransFormer — English to French Neural Machine Translation

A full-stack neural machine translation system built from scratch using PyTorch, served via a Flask web application. The model is an encoder-decoder Transformer with modern architectural components — Rotary Positional Encoding, Grouped-Query Attention, and SwiGLU feed-forward layers — trained on ~190,000 English-French sentence pairs.

Inspired by the [Machine Learning Mastery tutorial](https://machinelearningmastery.com/building-a-transformer-model-for-language-translation/) by Adrian Tam. The reference ships as a single monolithic script. This project restructures it end-to-end into a modular codebase with a production-style web interface.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Architecture](#2-architecture)
   - [High-Level Overview](#21-high-level-overview)
   - [Rotary Positional Encoding (RoPE)](#22-rotary-positional-encoding-rope)
   - [Grouped-Query Attention (GQA)](#23-grouped-query-attention-gqa)
   - [SwiGLU Feed-Forward Network](#24-swiglu-feed-forward-network)
   - [RMSNorm and Pre-Norm](#25-rmsnorm-and-pre-norm)
   - [Encoder Layer](#26-encoder-layer)
   - [Decoder Layer](#27-decoder-layer)
   - [Full Transformer](#28-full-transformer)
3. [Dataset and Tokenizer](#3-dataset-and-tokenizer)
4. [Training](#4-training)
5. [Inference](#5-inference)
   - [Greedy Decode](#51-greedy-decode)
   - [Beam Search Decode](#52-beam-search-decode)
6. [Codebase Walkthrough](#6-codebase-walkthrough)
7. [Web Application](#7-web-application)
   - [Backend — Flask](#71-backend--flask)
   - [Frontend — UI Features](#72-frontend--ui-features)
8. [API Reference](#8-api-reference)
9. [Setup and Usage](#9-setup-and-usage)
   - [Installation](#91-installation)
   - [Train the Model](#92-train-the-model)
   - [Run the Web App](#93-run-the-web-app)
10. [Configuration](#10-configuration)
11. [What Changed from the Reference](#11-what-changed-from-the-reference)
12. [Sample Output](#12-sample-output)

---

## 1. Project Structure

```
machine-translation/
│
├── config.py               # All hyperparameters and file paths
├── utils.py                # Text normalization, RoPE math, mask builders
├── tokenizer.py            # BPE tokenizer — train from scratch or load from disk
├── dataset.py              # TranslationDataset, collate_fn, DataLoader builder
├── train.py                # Training loop, optimizer, LR scheduler
├── inference.py            # Greedy decode, beam search, interactive CLI
├── main.py                 # Entry point — wires all modules together
│
├── model/
│   ├── __init__.py         # Exports Transformer
│   ├── attention.py        # RotaryPositionalEncoding, GQA
│   ├── feedforward.py      # SwiGLU
│   ├── layers.py           # EncoderLayer, DecoderLayer
│   └── transformer.py      # Top-level Transformer model
│
├── app.py                  # Flask web server
├── templates/
│   ├── index.html          # Main translator UI
│   └── history.html        # Standalone history page
├── static/
│   └── script.js           # All frontend logic
│
├── en_tokenizer.json       # Saved English BPE tokenizer
├── fr_tokenizer.json       # Saved French BPE tokenizer
├── transformer_model.pth   # Saved model weights
├── translations.db         # SQLite database of saved translations
└── requirements.txt
```

---

## 2. Architecture

### 2.1 High-Level Overview

The model follows the original encoder-decoder Transformer design from *Attention Is All You Need* (Vaswani et al., 2017), but replaces several components with more modern alternatives:

| Component | Vanilla 2017 | This Project |
|---|---|---|
| Positional Encoding | Sinusoidal (additive) | RoPE (applied inside attention) |
| Attention | Multi-Head Attention | Grouped-Query Attention |
| Feed-Forward | ReLU two-layer FFN | SwiGLU |
| Normalization | Post-Norm, LayerNorm | Pre-Norm, RMSNorm |

**Model dimensions:**

| Parameter | Value |
|---|---|
| Encoder layers | 4 |
| Decoder layers | 4 |
| Hidden dimension | 128 |
| Query heads | 8 |
| KV heads | 4 |
| Head dimension | 16 |
| FFN intermediate dim | 512 (4 × hidden) |
| Max sequence length | 768 |
| Vocabulary size | 8,000 (per language) |
| Dropout | 0.1 |
| Total parameters | ~3.5M |

---

### 2.2 Rotary Positional Encoding (RoPE)

Vanilla Transformers add a sinusoidal embedding to the token embedding before feeding it into the first layer. RoPE instead encodes position by rotating the query and key vectors inside each attention layer. This means positional information is directly embedded in how tokens attend to each other rather than added to the representation.

**Why RoPE?**
- Position-relative attention: attention scores between two tokens naturally depend on their relative distance, not absolute positions.
- Works better for longer sequences without extrapolation issues.
- Used in modern LLMs including LLaMA, Mistral, and Gemma.

**Implementation:**

For each head of dimension `d`, compute frequency bands:
```
inv_freq = 1 / (10000 ^ (2i / d))   for i in 0..d/2
```

For a token at position `p`, build a sinusoid:
```
sinusoid[p] = outer_product(p, inv_freq)
cos_cache[p] = cos(sinusoid[p])
sin_cache[p] = sin(sinusoid[p])
```

To apply RoPE to a query/key vector `x` at position `p`:
```
x_rotated = rotate_half(x)
x_rope = (x * cos[p]) + (x_rotated * sin[p])
```

Where `rotate_half` splits `x` into two halves and returns `[-x2, x1]` — equivalent to a 2D rotation in each frequency pair.

In the code, `RotaryPositionalEncoding` precomputes the cos/sin cache once during `__init__` and slices it at forward time based on the actual sequence length.

---

### 2.3 Grouped-Query Attention (GQA)

Standard Multi-Head Attention uses `H` heads for queries, keys, and values. Grouped-Query Attention (Ainslie et al., 2023) reduces the number of KV heads to `G` where `G < H`, with each KV head shared across `H/G` query heads.

**This project:** 8 query heads, 4 KV heads → 2 query heads share each KV head.

**Why GQA?**
- Reduces memory usage: the KV cache scales with `G` not `H`.
- Faster inference: fewer key/value projections.
- Used in LLaMA 2/3, Mistral, Gemma.

**Projections:**
```
Q: (hidden_dim) → (hidden_dim)            # 8 heads × head_dim
K: (hidden_dim) → (num_kv_heads × head_dim) # 4 heads × head_dim
V: (hidden_dim) → (num_kv_heads × head_dim) # 4 heads × head_dim
```

PyTorch's `F.scaled_dot_product_attention` with `enable_gqa=True` handles the KV sharing automatically during the attention computation.

---

### 2.4 SwiGLU Feed-Forward Network

The vanilla FFN applies two linear layers with a ReLU in between:
```
FFN(x) = W2 · ReLU(W1 · x)
```

SwiGLU (Shazeer, 2020) uses three linear projections and a gating mechanism:
```
SwiGLU(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)
```

Where `SiLU(x) = x · sigmoid(x)` and `⊙` is element-wise multiplication.

**Why SwiGLU?**
- The gate controls information flow: the `W_gate` branch decides how much of the `W_up` branch to let through.
- Empirically outperforms ReLU and GELU FFNs on language tasks.
- Used in LLaMA, PaLM, and most modern large models.

In this project the intermediate dimension is `4 × hidden_dim = 512`.

---

### 2.5 RMSNorm and Pre-Norm

**RMSNorm** normalizes by the root mean square of the activations rather than the mean and variance:
```
RMSNorm(x) = x / RMS(x) · γ
RMS(x) = sqrt(mean(x²))
```

Simpler and faster than LayerNorm since it skips mean centering.

**Pre-Norm** applies the normalization *before* the sublayer (attention or FFN) rather than after. This stabilizes gradients in deep networks and makes training more reliable.

```
# Pre-Norm (this project)
x = x + sublayer(norm(x))

# Post-Norm (original 2017 paper)
x = norm(x + sublayer(x))
```

---

### 2.6 Encoder Layer

Each encoder layer has two sublayers with Pre-Norm residual connections:

```
norm1 → Self-Attention → residual add
norm2 → SwiGLU FFN    → residual add
```

Self-attention is GQA where `Q = K = V = norm1(x)` — every position attends to every other position in the source sentence. RoPE is applied to queries and keys.

A padding mask prevents attention to `[pad]` tokens.

---

### 2.7 Decoder Layer

Each decoder layer has three sublayers:

```
norm1 → Masked Self-Attention   → residual add
norm2 → Cross-Attention         → residual add
norm3 → SwiGLU FFN              → residual add
```

**Masked Self-Attention:** The decoder attends to previously generated tokens only. A causal (upper-triangular) mask plus a padding mask is applied. RoPE is applied to queries and keys.

**Cross-Attention:** Queries come from the decoder (`norm2(x)`), keys and values come from the encoder output (`enc_out`). This is how the decoder reads the source sentence. No causal mask and no RoPE are applied here — the encoder output has already been encoded with its own positions.

---

### 2.8 Full Transformer

```
src_ids ──► src_embedding ──► Encoder × 4 ──► enc_out ──┐
                                                          │
tgt_ids ──► tgt_embedding ──► Decoder × 4 ◄──────────────┘
                                    │
                               Linear(hidden → vocab)
                                    │
                               logits (B, T, vocab_size)
```

The `Transformer` class exposes three methods:
- `encode(src_ids, src_mask)` — runs the source through all encoder layers
- `decode(tgt_ids, enc_out, tgt_mask)` — runs the target through all decoder layers and the output projection
- `forward(src_ids, tgt_ids, src_mask, tgt_mask)` — calls encode then decode, used during training

---

## 3. Dataset and Tokenizer

**Dataset:** [Anki EN-FR sentence pairs](https://www.manythings.org/anki/) via the TensorFlow mirror, downloaded automatically on first run.

- ~190,000 English-French sentence pairs
- NFKC-normalized, lowercased
- French target prefixed with `[start]` and suffixed with `[end]`

**Tokenizer:** Byte-Pair Encoding (BPE) trained separately for English and French using the HuggingFace `tokenizers` library.

- Vocabulary size: 8,000 tokens per language
- ByteLevel pre-tokenizer (handles any Unicode without unknown tokens)
- Special tokens: `[start]`, `[end]`, `[pad]`
- Saved to `en_tokenizer.json` and `fr_tokenizer.json` after first training run

BPE works by starting with individual characters and iteratively merging the most frequent adjacent pair until the vocabulary size is reached. This produces subword units that balance coverage (rare words get split into known pieces) and compactness (common words stay as single tokens).

---

## 4. Training

**Loss:** Cross-entropy over the vocabulary at each target position, ignoring `[pad]` tokens (`ignore_index`).

The model is trained to predict token `t+1` from the first `t` tokens — teacher forcing. The output is shifted by one:
```python
loss = cross_entropy(
    outputs[:, :-1, :],   # model predictions at positions 0..T-2
    fr_ids[:, 1:]          # targets at positions 1..T-1
)
```

**Optimizer:** Adam

**LR Schedule:** Linear warmup for the first 1,000 steps followed by cosine annealing to zero.

```
Step 0        → LR = 0.01 × base_lr
Step 1000     → LR = base_lr (1.0 × 0.005)
Step N_EPOCHS × batches_per_epoch → LR = 0
```

**Gradient clipping:** `clip_grad_norm_` with `max_norm=5.0` to prevent exploding gradients.

**Hardware:** Trained on an NVIDIA RTX 3050 (4GB VRAM) with CUDA 12.1. `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.benchmark = True` enabled for speed.

**Training config:**

| Parameter | Value |
|---|---|
| Batch size | 64 |
| Epochs | 60 |
| Learning rate | 0.005 |
| Warmup steps | 1,000 |
| Gradient clip | 5.0 |

---

## 5. Inference

### 5.1 Greedy Decode

At each step, the model produces logits over the vocabulary. Greedy decode always picks the single highest-probability token:

```
next_token = argmax(logits[:, -1, :])
```

Simple and fast but suboptimal — one bad early choice can derail the rest of the sequence.

### 5.2 Beam Search Decode

Beam search keeps the top `k` partial sequences (beams) alive at every step instead of just one.

**Algorithm:**

1. Encode the source sentence once — all beams share the same encoder output.
2. Initialise with a single beam: `([start], score=0.0)`.
3. At each step, for every active beam:
   - Run the decoder, apply `log_softmax` to the logits.
   - Take the top `beam_width` tokens with `torch.topk`.
   - Add each as a new candidate with cumulative log probability score.
4. Keep only the top `beam_width` candidates by score across all expansions.
5. Move any beam that ends with `[end]` to a completed list.
6. Stop when `beam_width` beams have completed or `max_len` is reached.
7. Return the completed beam with the highest length-normalised score:

```
score_normalised = cumulative_log_prob / sequence_length
```

Length normalisation prevents the model from favouring short translations (which accumulate fewer negative log prob terms).

This project uses `beam_width=3` by default. The webapp calls both `/translate` (best beam only) and `/translate/candidates` (all 3 beams) simultaneously.

---

## 6. Codebase Walkthrough

| File | Responsibility |
|---|---|
| `config.py` | Single source of truth. All hyperparameters and file paths. Change anything here and it propagates everywhere. |
| `utils.py` | Pure functions with no side effects. `normalize()` for text cleaning, `rotate_half()` and `apply_rotary_pos_emb()` for RoPE math, `create_causal_mask()` and `create_padding_mask()` for attention masks. |
| `tokenizer.py` | `load_or_train()` checks for saved tokenizer files first. If absent, trains BPE from scratch on the dataset and saves. Returns `(en_tokenizer, fr_tokenizer)`. |
| `dataset.py` | `TranslationDataset` wraps text pairs. `build_collate_fn()` returns a closure that encodes and pads batches using the tokenizers. `build_dataloader()` wires them into a PyTorch `DataLoader`. |
| `model/attention.py` | `RotaryPositionalEncoding` precomputes cos/sin cache. `GQA` implements grouped-query attention with optional RoPE and additive mask support. |
| `model/feedforward.py` | `SwiGLU` — three linear projections, SiLU activation, gated output. |
| `model/layers.py` | `EncoderLayer` (self-attn + FFN) and `DecoderLayer` (masked self-attn + cross-attn + FFN), both using Pre-Norm with RMSNorm. |
| `model/transformer.py` | `Transformer` composes all layers. Exposes `encode()`, `decode()`, and `forward()` separately so inference can reuse encoder output across beam steps without re-encoding. |
| `train.py` | `run_training()` contains the full training loop with tqdm progress bars, gradient clipping, and LR scheduling. |
| `inference.py` | `greedy_decode()` and `beam_search_decode()` for single best output. `beam_search_candidates()` returns all beams. `run_inference()` displays random sample translations after training. |
| `main.py` | Downloads dataset, builds tokenizers, builds model, creates dataloader, calls `run_training()`, then `run_inference()`. |
| `app.py` | Flask server. Loads model once at startup. Exposes all API routes. Saves each translation to SQLite. |

---

## 7. Web Application

### 7.1 Backend — Flask (`app.py`)

The Flask server loads the model, tokenizers, and initialises the SQLite database once at startup. All subsequent requests are served without re-loading weights.

**Database:** SQLite (`translations.db`), single table:

```sql
CREATE TABLE translations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    english   TEXT NOT NULL,
    french    TEXT NOT NULL,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

Every successful translation is inserted automatically.

### 7.2 Frontend — UI Features

The frontend is a single-page application built in vanilla HTML, CSS, and JavaScript — no frameworks.

**Layout:** Dark sidebar navigation + full-height two-panel translator (source left, translation right) separated by a vertical divider with a pulsing green dot during translation.

**Features:**

| Feature | Detail |
|---|---|
| Auto-clean input | Lowercases, fixes punctuation spacing, adds trailing punctuation before sending to model |
| Language detection | Checks for French vocabulary — warns user if input appears to be French |
| Debounce auto-translate | Automatically translates after 1 second of no typing |
| Confidence badge | Color-coded — green (≥80%), amber (50–79%), red (<50%) |
| Alternative candidates | 3 beam search candidates shown below translation, ranked #1–#3 |
| Click candidate | Single-click sets it as the main translation |
| Double-click candidate | Copies directly to clipboard with visual feedback |
| Character counter | Live counter with red warning above 200 characters |
| Copy button | Copies current translation; `Ctrl+C` shortcut when no text is selected |
| Paste button | Pastes clipboard content into source textarea |
| History sidebar | Fetches `/history` on click, renders as a table with confidence pills |
| API Docs sidebar | Inline documentation for all endpoints |

**Keyboard shortcuts:**

| Key | Action |
|---|---|
| `Enter` | Translate |
| `Ctrl+Enter` | Translate |
| `Escape` | Clear all |
| `Ctrl+C` | Copy translation (when nothing selected) |

---

## 8. API Reference

All endpoints are served from the same host as the web app.

---

### `POST /translate`

Returns the best beam search translation and a confidence score.

**Request body:**
```json
{ "text": "hello , how are you ?" }
```

**Response:**
```json
{
  "translation": "bonjour , comment allez-vous ?",
  "confidence": 93.5
}
```

`confidence` is derived from the beam's length-normalised log probability, converted to a 0–100 percentage via `exp(score) × 100`. It represents how certain the model is about this translation relative to the vocabulary distribution at each step.

---

### `POST /translate/candidates`

Returns the top 3 beam search candidates ranked by score.

**Request body:**
```json
{ "text": "hello , how are you ?" }
```

**Response:**
```json
{
  "candidates": [
    "bonjour , comment allez-vous ?",
    "bonjour , comment vas-tu ?",
    "salut , comment allez-vous ?"
  ]
}
```

---

### `GET /history`

Returns the last 20 saved translations ordered by most recent first.

**Response:**
```json
{
  "history": [
    {
      "english": "hello , how are you ?",
      "french": "bonjour , comment allez-vous ?",
      "confidence": 93.5,
      "timestamp": "2026-06-24 13:12:54"
    }
  ]
}
```

---

### `GET /`

Serves the main TransFormer web interface.

---

### `GET /history-page`

Serves the standalone translation history page.

---

## 9. Setup and Usage

### 9.1 Installation

```bash
pip install torch tokenizers flask requests tqdm
```

Requires Python 3.10+ (uses `X | Y` union type hints).

For GPU training (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 9.2 Train the Model

```bash
cd machine-translation
python main.py
```

**What happens on first run:**
1. Downloads `fra-eng.zip` (~3 MB) from the TensorFlow dataset mirror
2. Trains English and French BPE tokenizers on all 190,000 sentence pairs
3. Saves `en_tokenizer.json` and `fr_tokenizer.json`
4. Trains the Transformer for 60 epochs
5. Saves `transformer_model.pth`
6. Runs 5 random sample translations

Tokenizers are reused on subsequent runs — only the model trains again.

**Estimated training time:** ~2–3 hours on an RTX 3050. ~8–10 hours on CPU.

### 9.3 Run the Web App

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

**Input tips:**
- Use lowercase
- Add a space before punctuation: `hello , how are you ?`
- End sentences with ` .` and questions with ` ?`

The `cleanInput()` function in `script.js` handles this automatically before sending to the server.

---

## 10. Configuration

All hyperparameters live in `config.py`. Change anything here and it propagates to every module automatically.

```python
# Paths
DATASET_ZIP   = "fra-eng.zip"
EN_TOKENIZER  = "en_tokenizer.json"
FR_TOKENIZER  = "fr_tokenizer.json"
CHECKPOINT    = "transformer_model.pth"

# Tokenizer
VOCAB_SIZE    = 8_000
PAD_TOKEN     = "[pad]"
START_TOKEN   = "[start]"
END_TOKEN     = "[end]"

# Model
NUM_LAYERS    = 4
NUM_HEADS     = 8
NUM_KV_HEADS  = 4
HIDDEN_DIM    = 128
MAX_SEQ_LEN   = 768
DROPOUT       = 0.1

# Training
BATCH_SIZE    = 64
N_EPOCHS      = 60
LR            = 0.005
WARMUP_STEPS  = 1_000
CLIP_NORM     = 5.0

# Inference
MAX_GEN_LEN   = 60
```

---

## 11. What Changed from the Reference

The reference tutorial is a single ~450-line script where dataset loading, tokenization, model definition, training, and inference are written sequentially in one file with no separation of concerns.

**Structural changes:**
- Split into 12 files across a `model/` package and root modules
- `config.py` centralises every magic number
- `Transformer` exposes `encode()` and `decode()` as first-class methods — inference does not re-implement the encoder loop
- `collate_fn` is a closure, not a global function with implicit tokenizer state
- `greedy_decode` is `@torch.no_grad()` decorated, independently testable

**Architecture fixes:**
- Cross-attention passes `None` for both mask and RoPE — the reference incorrectly applies RoPE to encoder keys in the cross-attention layer
- K/V projections in GQA sized to `num_kv_heads × head_dim` not `hidden_dim`
- `num_workers=0` in DataLoader to avoid Windows multiprocessing / tokenizer pickle failures

**Added features (not in reference):**
- Beam search decode with length normalisation
- Flask web application with full REST API
- SQLite translation history
- Confidence score derived from beam log probabilities
- Single-page UI with sidebar navigation, candidate alternatives, keyboard shortcuts

---

## 12. Sample Output

```
[1/5]
  EN : are there any bananas ?
  FR : y a-t-il des bananes ?
  PRD: y a-t-il des bananes ?

[2/5]
  EN : i miss my parents .
  FR : mes parents me manquent .
  PRD: mes parents me manquent .

[3/5]
  EN : turn left at the second traffic light .
  FR : tourne au second feu à gauche !
  PRD: tournez au deuxième feu à gauche !

[4/5]
  EN : she loves him .
  FR : elle l'aime .
  PRD: elle l'aime .

[5/5]
  EN : i think i saw you before .
  FR : je pense vous avoir vu auparavant .
  PRD: je pense t'avoir vu avant .
```

---

## Reference

Adrian Tam, *Building a Transformer Model for Language Translation*, Machine Learning Mastery, 2025.
https://machinelearningmastery.com/building-a-transformer-model-for-language-translation/

Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
https://arxiv.org/abs/1706.03762

Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
https://arxiv.org/abs/2104.09864

Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023.
https://arxiv.org/abs/2305.13245

Shazeer, *GLU Variants Improve Transformer*, 2020.
https://arxiv.org/abs/2002.05202