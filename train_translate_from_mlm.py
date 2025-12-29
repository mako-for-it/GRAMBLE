import os, json, argparse
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertModel,
    BertLMHeadModel,
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
)

FLORES_MAP = {"nl": "nld_Latn", "es": "spa_Latn", "af": "afr_Latn", "xh": "xho_Latn", "as": "asm_Beng"}

def load_mlm_backbone(model_bin: str, config_json: str, device: str) -> BertForMaskedLM:
    with open(config_json, "r") as f:
        cfg = json.load(f)
    config = BertConfig(**cfg)
    mlm = BertForMaskedLM(config)

    ckpt = torch.load(model_bin, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    mlm.load_state_dict(state, strict=False)

    mlm.to(device).eval()
    return mlm

def load_tokenizer(tokenizer_json: str) -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)

    # If special tokens are missing, add them and later resize embeddings.
    add = {}
    if tok.pad_token is None: add["pad_token"] = "[PAD]"
    if tok.cls_token is None: add["cls_token"] = "[CLS]"
    if tok.sep_token is None: add["sep_token"] = "[SEP]"
    if tok.mask_token is None: add["mask_token"] = "[MASK]"
    if add:
        tok.add_special_tokens(add)

    # Ensure IDs exist
    if tok.pad_token_id is None or tok.cls_token_id is None or tok.sep_token_id is None:
        raise SystemExit(
            f"Tokenizer special token ids missing: "
            f"pad={tok.pad_token_id} cls={tok.cls_token_id} sep={tok.sep_token_id}. "
            f"Rebuild tokenizer with these tokens included."
        )
    return tok

def build_encoder_decoder_from_mlm(mlm: BertForMaskedLM) -> EncoderDecoderModel:
    enc_cfg = mlm.config
    dec_cfg = BertConfig(**enc_cfg.to_dict())
    dec_cfg.is_decoder = True
    dec_cfg.add_cross_attention = True

    encoder = BertModel(enc_cfg)
    decoder = BertLMHeadModel(dec_cfg)

    # Load MLM backbone weights into both sides (non-strict is fine)
    encoder.load_state_dict(mlm.bert.state_dict(), strict=False)
    decoder.bert.load_state_dict(mlm.bert.state_dict(), strict=False)

    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    return model

def set_generation_tokens(model: EncoderDecoderModel, tok: PreTrainedTokenizerFast) -> None:
    # Set on BOTH config and generation_config so generate() won't complain
    for cfg in (model.config, model.generation_config):
        cfg.decoder_start_token_id = tok.cls_token_id
        cfg.bos_token_id = tok.cls_token_id
        cfg.eos_token_id = tok.sep_token_id
        cfg.pad_token_id = tok.pad_token_id

def maybe_resize_embeddings(model: EncoderDecoderModel, tok: PreTrainedTokenizerFast) -> None:
    # If tokenizer length differs from model vocab size, resize both encoder and decoder embeddings.
    tok_len = len(tok)
    enc_vocab = model.encoder.get_input_embeddings().num_embeddings
    dec_vocab = model.decoder.get_input_embeddings().num_embeddings

    if tok_len != enc_vocab:
        model.encoder.resize_token_embeddings(tok_len)
    if tok_len != dec_vocab:
        model.decoder.resize_token_embeddings(tok_len)

def eval_zero_shot(model: EncoderDecoderModel, tok: PreTrainedTokenizerFast, tgt_code: str,
                   samples: int, max_len: int, beams: int, out_csv: str) -> None:
    if tgt_code not in FLORES_MAP:
        raise SystemExit(f"Unknown --tgt {tgt_code}. Choose from {list(FLORES_MAP.keys())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    eng = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    tgt = load_dataset("openlanguagedata/flores_plus", FLORES_MAP[tgt_code], split="devtest")

    bleu = BLEU()
    chrf = CHRF(word_order=2)

    rows = []
    n = min(samples, len(tgt))

    for i in tqdm(range(n), desc=f"ZERO-SHOT {tgt_code} (devtest)"):
        src = eng[i]["text"]
        ref = tgt[i]["text"]

        inputs = tok(src, return_tensors="pt", truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_length=max_len,
                num_beams=beams,
                early_stopping=True,
                decoder_start_token_id=tok.cls_token_id,  # extra-safe
            )

        hyp = tok.decode(out_ids[0], skip_special_tokens=True)

        b = bleu.sentence_score(hyp, [ref]).score
        c = chrf.sentence_score(hyp, [ref]).score

        rows.append({
            "flores eng": src,
            "flores targetlang(answer)": ref,
            "zero-shot output": hyp,
            "bleu score": round(b, 2),
            "chrf": round(c, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"\n>>> Saved: {out_csv}")
    print(f">>> Avg BLEU: {df['bleu score'].mean():.2f}")
    print(f">>> Avg CHRF: {df['chrf'].mean():.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help="like es-en")
    ap.add_argument("--tgt", required=True, help="like es / af / xh / as / nl")

    # Keep --steps for compatibility with your run_en.sh, but it is NOT used.
    ap.add_argument("--steps", type=int, default=3000, help="(ignored) kept for compatibility; no MT training is done.")

    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--beams", type=int, default=4)
    args = ap.parse_args()

    pair_dir = f"outputs/{args.pair}"
    tok_json = f"{pair_dir}/tokenizer/tokenizer.json"
    mlm_bin = f"{pair_dir}/mlm/model.bin"
    cfg_json = "model/elc-bert/configs/base.json"

    if not os.path.exists(tok_json):
        raise SystemExit(f"Missing tokenizer: {tok_json}")
    if not os.path.exists(mlm_bin):
        raise SystemExit(f"Missing MLM model.bin: {mlm_bin} (did Step C finish?)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer(tok_json)
    mlm = load_mlm_backbone(mlm_bin, cfg_json, device)
    model = build_encoder_decoder_from_mlm(mlm)

    # If we added special tokens, or tokenizer length differs, make embeddings match
    maybe_resize_embeddings(model, tok)

    set_generation_tokens(model, tok)

    # Save this “no-parallel” model snapshot under outputs/<pair>/mt/
    out_dir = f"{pair_dir}/mt"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # Evaluate (zero-shot)
    out_csv = os.path.join(out_dir, "evaluation_results.csv")
    eval_zero_shot(model, tok, args.tgt, args.samples, args.max_len, args.beams, out_csv)

if __name__ == "__main__":
    main()