import json
import torch
import sys
import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM
from tokenizers import Tokenizer
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF

# 1. Path Fix & Safe Globals
sys.path.append(os.path.join(os.getcwd(), "WALS"))
import wals 
from dictionary_index import DictionaryLookup

# Fix for PyTorch 2.6+ loading Namespace objects
torch.serialization.add_safe_globals([argparse.Namespace])

class BERTNaturalizer:
    def __init__(self, lang_code):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Paths
        model_bin = f"outputs/{lang_code}/model/model.bin"
        vocab_json = f"outputs/{lang_code}/tokenizer/tokenizer.json"
        config_json = "model/elc-bert/configs/base.json"
        
        with open(config_json, 'r') as f:
            config_data = json.load(f)
        
        # Initialize Architecture
        config = BertConfig(**config_data)
        self.model = BertForMaskedLM(config)
        
        # Load Checkpoint and Unwrap Weights
        if not os.path.exists(model_bin):
            raise FileNotFoundError(f"Model weights not found at {model_bin}")
            
        checkpoint = torch.load(model_bin, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        
        # Load into model (strict=False handles ELC-BERT head naming differences)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()
        
        # Load Tokenizer
        self.tokenizer = Tokenizer.from_file(vocab_json)

    def _should_mask(self, word, pos):
        # Only consider unknowns
        if pos != "OTHER":
            return False

        # Don't mask punctuation
        if re.fullmatch(r"[.,!?;:]", word):
            return False

        # Don't mask pure numbers / numeric-like
        if re.fullmatch(r"\d+([.,]\d+)*", word):
            return False

        # Don't mask short fragments
        if len(word) <= 2:
            return False

        # Don't mask likely proper nouns (very conservative)
        if word[:1].isupper():
            return False

        # Mask Latin alphabet words (your "English leak / unknown" case)
        if re.fullmatch(r"[A-Za-z][A-Za-z']+", word):
            return True

        # Otherwise keep as-is
        return False


    def refine(self, tokens_pos):
        """
        tokens_pos: list of (word, pos) tuples
        - mask only selected tokens
        - replace only [MASK] positions
        """
        mask_tok = "[MASK]"
        mask_id = self.tokenizer.token_to_id(mask_tok)
        if mask_id is None:
            raise ValueError("Tokenizer vocab does not contain [MASK].")

        words = [w for (w, _) in tokens_pos]

        masked_words = []
        for (w, p) in tokens_pos:
            masked_words.append(mask_tok if self._should_mask(w, p) else w)

        # If nothing is masked, return original
        if mask_tok not in masked_words:
            return " ".join(words)

        text = " ".join(masked_words)
        enc = self.tokenizer.encode(text)
        ids = torch.tensor([enc.ids], device=self.device)

        with torch.no_grad():
            logits = self.model(ids).logits[0]   # [L, V]
            preds = logits.argmax(dim=-1)        # [L]

        input_ids = ids[0]
        new_ids = input_ids.clone()

        mask_positions = (input_ids == mask_id).nonzero(as_tuple=True)[0]
        new_ids[mask_positions] = preds[mask_positions]

        return self.tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)


def run_evaluation(lang_code, num_samples=50):
    flores_map = {"nl": "nld_Latn", "es": "spa_Latn", "af": "afr_Latn", "xh": "xho_Latn", "as": "asm_Beng"}
    wals_map = {"nl": "dut", "es": "spa", "af": "afr", "xh": "xho", "as": "ass"}
    
    target_flores = flores_map.get(lang_code)
    target_wals = wals_map.get(lang_code)

    print(f">>> Initializing components for {lang_code}...")
    dl = DictionaryLookup("dictionary/translations.jsonl", lang_code)
    naturalizer = BERTNaturalizer(lang_code)
    bleu = BLEU()
    chrf = CHRF(word_order=2)

    # 1. Load BOTH datasets to get the pairs
    print(f">>> Loading FLORES+ (English and {lang_code})...")
    eng_ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    tgt_ds = load_dataset("openlanguagedata/flores_plus", target_flores, split="devtest")
    
    results = []
    
    # 2. Iterate through them (they are pre-aligned by index)
    for i in tqdm(range(min(num_samples, len(tgt_ds))), desc="Translating"):
        # Access the 'text' field as per the data fields you provided
        eng_sent = eng_ds[i]["text"] 
        ref_sent = tgt_ds[i]["text"]
        
        # Step 4: Dictionary Lookup
        gloss_tokens_pos = dl.translate_sentence(eng_sent)
        
        # Step 5: WALS Reordering
        prof = wals.wals_profiles.get(target_wals)
        reordered_pos = wals.reorder_sentence(gloss_tokens_pos, prof)
        
        # Step 6: BERT Naturalization
        final_sent = naturalizer.refine(reordered_pos)
        
        # 5. Metrics
        b_score = bleu.sentence_score(final_sent, [ref_sent]).score
        c_score = chrf.sentence_score(final_sent, [ref_sent]).score
        
        results.append({
            "flores eng": eng_sent,
            "flores targetlang(answer)": ref_sent,
            "my targetlang sentence": final_sent,
            "bleu score": round(b_score, 2),
            "chrf": round(c_score, 2)
        })

    # Save to CSV
    output_dir = f"outputs/{lang_code}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/evaluation_results.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    print(f"\n>>> Done! Average BLEU: {df['bleu score'].mean():.2f}")
    print(f"\n>>> Done! Average CHRF: {df['chrf'].mean():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="nl", help="Language code (nl, af, as, es, xh)")
    parser.add_argument("--samples", type=int, default=50, help="Number of sentences to evaluate")
    args = parser.parse_args()
    
    run_evaluation(args.lang, args.samples)