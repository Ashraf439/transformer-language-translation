"""
config.py — single source of truth for all hyperparameters and paths.
"""

# ── Paths ─────────────────────────────────────────────────────
DATASET_ZIP   = "fra-eng.zip"
DATASET_FILE  = "fra.txt"
DATASET_URL   = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
EN_TOKENIZER  = "en_tokenizer.json"
FR_TOKENIZER  = "fr_tokenizer.json"
CHECKPOINT    = "transformer_model.pth"

# ── Tokenizer ─────────────────────────────────────────────────
VOCAB_SIZE     = 8_000
PAD_TOKEN      = "[pad]"
START_TOKEN    = "[start]"
END_TOKEN      = "[end]"
SPECIAL_TOKENS = [START_TOKEN, END_TOKEN, PAD_TOKEN]

# ── Model ─────────────────────────────────────────────────────
NUM_LAYERS   = 4
NUM_HEADS    = 8
NUM_KV_HEADS = 4
HIDDEN_DIM   = 128
MAX_SEQ_LEN  = 768
DROPOUT      = 0.1

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE   = 64
N_EPOCHS     = 60
LR           = 0.005
WARMUP_STEPS = 1_000
CLIP_NORM    = 5.0

# ── Inference ─────────────────────────────────────────────────
N_SAMPLES   = 5
MAX_GEN_LEN = 60