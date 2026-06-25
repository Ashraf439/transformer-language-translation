"""
inference.py — greedy autoregressive decoding and sample display.
"""

import random
import torch

import config
from utils import create_causal_mask, create_padding_mask


# @torch.no_grad()
def greedy_decode(
    model,
    en_ids: torch.Tensor,
    en_tokenizer,
    fr_tokenizer,
    device: torch.device,
    max_len: int = config.MAX_GEN_LEN,
) -> list[int]:
    """
    Greedy decode for a single source sequence.
    en_ids: (1, src_len) LongTensor on *device*.
    Returns list of predicted token ids.
    """
    start_id  = fr_tokenizer.token_to_id(config.START_TOKEN)
    end_id    = fr_tokenizer.token_to_id(config.END_TOKEN)
    pad_id_en = en_tokenizer.token_to_id(config.PAD_TOKEN)
    pad_id_fr = fr_tokenizer.token_to_id(config.PAD_TOKEN)

    src_mask = create_padding_mask(en_ids, pad_id_en)
    enc_out  = model.encode(en_ids, src_mask)

    fr_ids = torch.tensor([[start_id]], device=device)

    for _ in range(max_len):
        tgt_mask = (
            create_causal_mask(fr_ids.shape[1], device).unsqueeze(0)
            + create_padding_mask(fr_ids, pad_id_fr)
        )
        logits     = model.decode(fr_ids, enc_out, tgt_mask)
        next_token = logits.argmax(dim=-1)[:, -1:]
        fr_ids     = torch.cat([fr_ids, next_token], dim=-1)
        if next_token.item() == end_id:
            break

    return fr_ids[0].tolist()


def run_inference(model, text_pairs, en_tokenizer, fr_tokenizer, device):
    model.eval()
    print(f"\n[INFERENCE] {config.N_SAMPLES} random samples (max_len={config.MAX_GEN_LEN})\n")

    for i, (en, true_fr) in enumerate(random.sample(text_pairs, config.N_SAMPLES), 1):
        en_ids = torch.tensor(
            en_tokenizer.encode(en).ids
        ).unsqueeze(0).to(device)

        pred_ids = greedy_decode(model, en_ids, en_tokenizer, fr_tokenizer, device)
        pred_fr  = fr_tokenizer.decode(pred_ids)

        print(f"[{i}/{config.N_SAMPLES}]")
        print(f"  EN : {en}")
        print(f"  FR : {true_fr}")
        print(f"  PRD: {pred_fr}")
        print()
@torch.no_grad()
def beam_search_decode(
    model, 
    en_ids,
    en_tokenizer, 
    fr_tokenizer, 
    device, 
    beam_width=3, 
    max_len=config.MAX_GEN_LEN
):
    start_id  = fr_tokenizer.token_to_id(config.START_TOKEN)
    end_id    = fr_tokenizer.token_to_id(config.END_TOKEN)
    pad_id_en = en_tokenizer.token_to_id(config.PAD_TOKEN)
    pad_id_fr = fr_tokenizer.token_to_id(config.PAD_TOKEN)

    src_mask = create_padding_mask(en_ids, pad_id_en)
    enc_out  = model.encode(en_ids, src_mask)
    
    #Initialize beams
    fr_ids = torch.tensor([[start_id]], device=device)
    beams = [(fr_ids[0], 0.0)]

    completed = []
    for _ in range(max_len):
        candidates = []
        for token_ids, score in beams:
            if token_ids[-1].item() == end_id:
                completed.append((token_ids, score))
                continue
            fr_ids = token_ids.unsqueeze(0)          # (1, seq_len)
            tgt_mask = (
                create_causal_mask(fr_ids.shape[1], device).unsqueeze(0)
                + create_padding_mask(fr_ids, pad_id_fr)
            )
            logits = model.decode(fr_ids, enc_out, tgt_mask)
            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
            top_scores, top_tokens = torch.topk(log_probs, beam_width)
            
            for i in range(beam_width):
                new_ids = torch.cat([token_ids, top_tokens[i].unsqueeze(0)])
                new_score = score + top_scores[i].item()
                candidates.append((new_ids, new_score))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        if len(completed) >= beam_width:
            break
    
    all_beams = completed if completed else beams
    best = max(all_beams, key=lambda x: x[1] / len(x[0]))
    return best[0].tolist(), best[1] / len(best[0])

# @torch.no_grad()
def beam_search_candidates(
    model, 
    en_ids,
    en_tokenizer, 
    fr_tokenizer, 
    device, 
    beam_width=3, 
    max_len=config.MAX_GEN_LEN
):
    start_id  = fr_tokenizer.token_to_id(config.START_TOKEN)
    end_id    = fr_tokenizer.token_to_id(config.END_TOKEN)
    pad_id_en = en_tokenizer.token_to_id(config.PAD_TOKEN)
    pad_id_fr = fr_tokenizer.token_to_id(config.PAD_TOKEN)

    src_mask = create_padding_mask(en_ids, pad_id_en)
    enc_out  = model.encode(en_ids, src_mask)
    
    #Initialize beams
    fr_ids = torch.tensor([[start_id]], device=device)
    beams = [(fr_ids[0], 0.0)]

    completed = []
    for _ in range(max_len):
        candidates = []
        for token_ids, score in beams:
            if token_ids[-1].item() == end_id:
                completed.append((token_ids, score))
                continue
            fr_ids = token_ids.unsqueeze(0)          # (1, seq_len)
            tgt_mask = (
                create_causal_mask(fr_ids.shape[1], device).unsqueeze(0)
                + create_padding_mask(fr_ids, pad_id_fr)
            )
            logits = model.decode(fr_ids, enc_out, tgt_mask)
            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
            top_scores, top_tokens = torch.topk(log_probs, beam_width)
            
            for i in range(beam_width):
                new_ids = torch.cat([token_ids, top_tokens[i].unsqueeze(0)])
                new_score = score + top_scores[i].item()
                candidates.append((new_ids, new_score))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        if len(completed) >= beam_width:
            break
    
    all_beams = completed if completed else beams
    all_beams = sorted(all_beams, key=lambda x: x[1] / len(x[0]), reverse=True)
    return [beam[0].tolist() for beam in all_beams]

if __name__ == "__main__":
    import tokenizers
    from model import Transformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOAD] Device: {device}")

    print("[LOAD] Loading tokenizers...")
    en_tokenizer = tokenizers.Tokenizer.from_file(config.EN_TOKENIZER)
    fr_tokenizer = tokenizers.Tokenizer.from_file(config.FR_TOKENIZER)

    print("[LOAD] Building model and loading weights...")
    model = Transformer(
        num_layers     = config.NUM_LAYERS,
        num_heads      = config.NUM_HEADS,
        num_kv_heads   = config.NUM_KV_HEADS,
        hidden_dim     = config.HIDDEN_DIM,
        max_seq_len    = config.MAX_SEQ_LEN,
        vocab_size_src = len(en_tokenizer.get_vocab()),
        vocab_size_tgt = len(fr_tokenizer.get_vocab()),
        dropout        = config.DROPOUT,
    ).to(device)

    model.load_state_dict(torch.load(config.CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    print("[LOAD] Model ready.\n")

    print("=" * 50)
    print("  EN → FR Translator  (type 'quit' to exit)")
    print("=" * 50)

    while True:
        sentence = input("\nEnglish: ").strip()
        if sentence.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not sentence:
            continue
        en_ids = torch.tensor(
            en_tokenizer.encode(sentence.lower().strip()).ids
        ).unsqueeze(0).to(device)
        pred_ids = greedy_decode(model, en_ids, en_tokenizer, fr_tokenizer, device)
        print(f"French : {fr_tokenizer.decode(pred_ids)}")