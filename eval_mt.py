import os
import argparse
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, EncoderDecoderModel
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm

FLORES_MAP = {
    "nl": "nld_Latn",
    "es": "spa_Latn",
    "af": "afr_Latn",
    "xh": "xho_Latn",
    "as": "asm_Beng",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help="e.g., es-en")
    ap.add_argument("--tgt", required=True, help="e.g., es / af / xh / as / nl")
    ap.add_argument("--samples", type=int, default=100)
    args = ap.parse_args()

    mt_dir = f"outputs/{args.pair}/mt"
    if not os.path.exists(mt_dir):
        raise SystemExit(f"Missing model dir: {mt_dir}")

    tok = AutoTokenizer.from_pretrained(mt_dir)
    model = EncoderDecoderModel.from_pretrained(mt_dir)

    # ---- critical: set generation special tokens on BOTH config + generation_config ----
    if tok.cls_token_id is None or tok.sep_token_id is None or tok.pad_token_id is None:
        raise SystemExit(
            f"Tokenizer missing special token ids: "
            f"cls={tok.cls_token_id} sep={tok.sep_token_id} pad={tok.pad_token_id}"
        )

    # Use CLS as start/BOS, SEP as EOS, PAD as PAD
    for cfg in (model.config, model.generation_config):
        cfg.decoder_start_token_id = tok.cls_token_id
        cfg.bos_token_id = tok.cls_token_id
        cfg.eos_token_id = tok.sep_token_id
        cfg.pad_token_id = tok.pad_token_id
    # -------------------------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    if args.tgt not in FLORES_MAP:
        raise SystemExit(f"Unknown tgt {args.tgt}. Choose from: {list(FLORES_MAP.keys())}")

    eng = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    tgt = load_dataset("openlanguagedata/flores_plus", FLORES_MAP[args.tgt], split="devtest")

    bleu = BLEU()
    chrf = CHRF(word_order=2)

    rows = []
    n = min(args.samples, len(tgt))

    for i in tqdm(range(n), desc=f"eval {args.pair}"):
        src = eng[i]["text"]
        ref = tgt[i]["text"]

        inputs = tok(src, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )

        hyp = tok.decode(out_ids[0], skip_special_tokens=True)

        rows.append({
            "flores eng": src,
            "flores targetlang(answer)": ref,
            "my targetlang sentence": hyp,
            "bleu score": round(bleu.sentence_score(hyp, [ref]).score, 2),
            "chrf": round(chrf.sentence_score(hyp, [ref]).score, 2),
        })

    df = pd.DataFrame(rows)
    out_csv = f"outputs/{args.pair}/evaluation_results.csv"
    df.to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    print(f">>> Avg BLEU: {df['bleu score'].mean():.2f}")
    print(f">>> Avg CHRF: {df['chrf'].mean():.2f}")

if __name__ == "__main__":
    main()
