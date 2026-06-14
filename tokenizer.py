"""
tokenizer.py — build BPE tokenizers from scratch or load from disk.
"""

import os
import tokenizers
import tokenizers.decoders
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.trainers

import config


def _make_tokenizer() -> tokenizers.Tokenizer:
    tok = tokenizers.Tokenizer(tokenizers.models.BPE())
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = tokenizers.decoders.ByteLevel()
    return tok


def _enable_padding(tok: tokenizers.Tokenizer) -> None:
    tok.enable_padding(
        pad_id=tok.token_to_id(config.PAD_TOKEN),
        pad_token=config.PAD_TOKEN,
    )


def load_or_train(
    text_pairs: list[tuple[str, str]],
) -> tuple[tokenizers.Tokenizer, tokenizers.Tokenizer]:
    """Load saved tokenizers if they exist, otherwise train and save."""

    if os.path.exists(config.EN_TOKENIZER) and os.path.exists(config.FR_TOKENIZER):
        print("[TOKENIZER] Found saved tokenizers — loading from disk.")
        en_tok = tokenizers.Tokenizer.from_file(config.EN_TOKENIZER)
        fr_tok = tokenizers.Tokenizer.from_file(config.FR_TOKENIZER)
        _enable_padding(en_tok)
        _enable_padding(fr_tok)
        print("[TOKENIZER] Padding re-enabled after loading.")
    else:
        print("[TOKENIZER] No saved tokenizers found — training from scratch.")
        en_tok = _make_tokenizer()
        fr_tok = _make_tokenizer()

        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=config.VOCAB_SIZE,
            special_tokens=config.SPECIAL_TOKENS,
            show_progress=True,
        )

        print("[TOKENIZER] Training English BPE tokenizer...")
        en_tok.train_from_iterator([p[0] for p in text_pairs], trainer=trainer)
        print("[TOKENIZER] Training French BPE tokenizer...")
        fr_tok.train_from_iterator([p[1] for p in text_pairs], trainer=trainer)

        _enable_padding(en_tok)
        _enable_padding(fr_tok)

        en_tok.save(config.EN_TOKENIZER, pretty=True)
        fr_tok.save(config.FR_TOKENIZER, pretty=True)
        print("[TOKENIZER] Saved tokenizers to disk.")

    print(f"[TOKENIZER] EN vocab size: {len(en_tok.get_vocab()):,}")
    print(f"[TOKENIZER] FR vocab size: {len(fr_tok.get_vocab()):,}")
    return en_tok, fr_tok