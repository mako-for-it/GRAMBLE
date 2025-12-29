#!/usr/bin/env python3
"""
create_training_dataset.py

Process monolingual_corpus/*.txt.xz (1 sentence per line), streaming:

1) Remove lines that also appear in FLORES (exact match after normalization).
2) Optionally remove near-duplicates of FLORES lines (SimHash + LSH).
3) Write output until reaching a BabyLM-style "word budget" (default 10,000,000).

Word budget:
- --word_mode whitespace (default): counts len(text.split()) on normalized text.
- --word_mode fallback_chars:
    * Initially counts whitespace words, but samples early kept lines.
    * If most sampled lines have < 2 whitespace words, automatically switches to
      counting characters instead (so progress isn't near-zero for low-whitespace scripts).
    * When switched, the target is interpreted as "target units" (chars) rather than words.

Modes:
- first  : streaming-friendly; writes deterministic first lines until budget reached (recommended)
- random : reservoir sampling by *lines only* (NOT word-budget). Documented restriction.

Progress:
- tqdm progress bar is driven by cumulative units written toward target_words/target_units.
  If tqdm isn't installed, it disables automatically and prints a warning.

Python 3.10+
"""

from __future__ import annotations

import argparse
import hashlib
import lzma
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None


# ---------------------------
# Language mapping (your files -> FLORES codes)
# ---------------------------
FILE_TO_FLORES_LANG = {
    "as.txt.xz": "asm_Beng",  # Assamese
    "af.txt.xz": "afr_Latn",  # Afrikaans
    "xh.txt.xz": "xho_Latn",  # Xhosa
    "lo.txt.xz": "lao_Laoo",  # Lao
    "ko.txt.xz": "kor_Hang",  # Korean
    "es.txt.xz": "spa_Latn",  # Spanish
    "nl.txt.xz": "nld_Latn",  # Dutch
    "en.txt.xz": "eng_Latn",  # English
}


# ---------------------------
# Normalization
# ---------------------------
_ws_re = re.compile(r"\s+", flags=re.UNICODE)

def normalize_line(s: str) -> str:
    """
    Normalize text for matching:
      - Unicode NFKC
      - remove common invisible characters
      - strip
      - collapse whitespace runs to single space
    """
    s = unicodedata.normalize("NFKC", s)
    # common invisible chars in web corpora
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    s = s.strip()
    s = _ws_re.sub(" ", s)
    return s


# ---------------------------
# FLORES loading (HF datasets)
# ---------------------------
def load_flores_membership(lang_code: str, splits: List[str]) -> Set[str]:
    """
    FLORES+ (openlanguagedata/flores_plus) - gated.
    - config: lang_code like "lao_Laoo"
    - splits: "dev", "devtest"
    - sentence field: "text"
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise SystemExit("Missing dependency: datasets. Install with: pip install datasets") from e

    flores_set: Set[str] = set()
    for split in splits:
        ds = load_dataset(
            "openlanguagedata/flores_plus",
            lang_code,
            split=split,
            streaming=True,
        )
        for ex in ds:
            sent = ex.get("text")
            if isinstance(sent, str):
                n = normalize_line(sent)
                if n:
                    flores_set.add(n)
    return flores_set


# ---------------------------
# Near-duplicate detection (SimHash + LSH)
# ---------------------------
def _char_ngrams(text: str, n: int = 5):
    if len(text) <= n:
        yield text
        return
    for i in range(len(text) - n + 1):
        yield text[i:i+n]

def simhash64(text: str, ngram: int = 5) -> int:
    v = [0] * 64
    for ng in _char_ngrams(text, n=ngram):
        h = hashlib.blake2b(ng.encode("utf-8", errors="strict"), digest_size=8).digest()
        x = int.from_bytes(h, "big")
        for i in range(64):
            v[i] += 1 if ((x >> i) & 1) else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

@dataclass
class LSHIndex:
    flores_hashes: List[int]
    buckets: List[Dict[int, List[int]]]
    bands: int
    band_bits: int
    mask: int
    ngram: int

def build_lsh_index(flores_lines: Set[str], ngram: int = 5, bands: int = 4) -> LSHIndex:
    if 64 % bands != 0:
        raise ValueError("bands must divide 64 (e.g., 1,2,4,8,16,32,64)")
    band_bits = 64 // bands
    mask = (1 << band_bits) - 1

    flores_texts = list(flores_lines)
    flores_hashes = [simhash64(t, ngram=ngram) for t in flores_texts]
    buckets: List[Dict[int, List[int]]] = [dict() for _ in range(bands)]

    for idx, sh in enumerate(flores_hashes):
        for b in range(bands):
            key = (sh >> (b * band_bits)) & mask
            buckets[b].setdefault(key, []).append(idx)

    return LSHIndex(
        flores_hashes=flores_hashes,
        buckets=buckets,
        bands=bands,
        band_bits=band_bits,
        mask=mask,
        ngram=ngram,
    )

def is_near_duplicate(text: str, idx: LSHIndex, max_hamming: int) -> bool:
    sh = simhash64(text, ngram=idx.ngram)
    candidates: Set[int] = set()
    for b in range(idx.bands):
        key = (sh >> (b * idx.band_bits)) & idx.mask
        for j in idx.buckets[b].get(key, []):
            candidates.add(j)
    for j in candidates:
        if hamming64(sh, idx.flores_hashes[j]) <= max_hamming:
            return True
    return False


# ---------------------------
# IO helpers / counters
# ---------------------------
def open_xz_text(path: str, mode: str):
    return lzma.open(path, mode=mode, encoding="utf-8", errors="strict")

@dataclass
class Counters:
    read: int = 0
    removed_overlap: int = 0  # exact + near-dup
    kept_after_filter: int = 0
    written_lines: int = 0
    written_units: int = 0   # words or chars depending on mode
    unit_label: str = "words"


# ---------------------------
# Word counting / fallback logic
# ---------------------------
def count_whitespace_words(s: str) -> int:
    # s is normalized already
    return 0 if not s else len(s.split())

def count_chars(s: str) -> int:
    return len(s)

@dataclass
class FallbackDetector:
    """
    For word_mode=fallback_chars:
      - sample the first `sample_n` kept lines
      - if the majority have <2 whitespace words, switch to char counting
    """
    sample_n: int = 5000
    threshold_ratio_lowword: float = 0.7  # if >=70% of sampled lines have <2 words -> switch
    seen: int = 0
    lowword: int = 0
    decided: bool = False
    use_chars: bool = False

    def observe(self, whitespace_words: int) -> None:
        if self.decided:
            return
        self.seen += 1
        if whitespace_words < 2:
            self.lowword += 1
        if self.seen >= self.sample_n:
            ratio = self.lowword / max(1, self.seen)
            self.use_chars = ratio >= self.threshold_ratio_lowword
            self.decided = True

    def maybe_decide_early(self) -> None:
        # optional early decision if it's already obvious (keeps it simple)
        if self.decided:
            return
        if self.seen >= 1000:
            ratio = self.lowword / max(1, self.seen)
            # if it's extremely likely, decide early
            if ratio >= 0.9:
                self.use_chars = True
                self.decided = True
            elif ratio <= 0.1:
                self.use_chars = False
                self.decided = True


def get_units_and_label(
    norm: str,
    word_mode: str,
    detector: Optional[FallbackDetector],
) -> Tuple[int, str]:
    """
    Returns (units, unit_label) for progress/stop condition.
    """
    if word_mode == "whitespace":
        return count_whitespace_words(norm), "words"

    # fallback_chars
    assert detector is not None
    w = count_whitespace_words(norm)
    detector.observe(w)
    detector.maybe_decide_early()
    if detector.decided and detector.use_chars:
        return count_chars(norm), "chars"
    return w, "words"


# ---------------------------
# Processing modes
# ---------------------------
def run_first_mode_wordbudget(
    infile_xz: str,
    outfile_txt: str,
    flores_set: Set[str],
    target_words: int,
    word_mode: str,
    log_every: int,
    near_dup: bool,
    lsh_index: Optional[LSHIndex],
    max_hamming: int,
    show_pbar: bool,
    pbar_desc: str,
) -> Counters:
    c = Counters()
    detector = FallbackDetector() if word_mode == "fallback_chars" else None

    pbar = None
    if show_pbar and tqdm is not None:
        # label will be updated if fallback switches
        pbar = tqdm(total=target_words, unit="words", dynamic_ncols=True, desc=pbar_desc, leave=True)

    with open_xz_text(infile_xz, "rt") as fin, open(outfile_txt, "wt", encoding="utf-8", errors="strict", newline="\n") as fout:
        for line in fin:
            c.read += 1
            if log_every > 0 and (c.read % log_every == 0):
                print(
                    f"[progress] read={c.read:,} removed={c.removed_overlap:,} kept={c.kept_after_filter:,} "
                    f"written_lines={c.written_lines:,} written_{c.unit_label}={c.written_units:,}/{target_words:,}",
                    file=sys.stderr
                )

            norm = normalize_line(line)
            if not norm:
                continue

            # exact match
            if norm in flores_set:
                c.removed_overlap += 1
                continue

            # near-duplicate match
            if near_dup and lsh_index is not None:
                if is_near_duplicate(norm, lsh_index, max_hamming=max_hamming):
                    c.removed_overlap += 1
                    continue

            c.kept_after_filter += 1

            # stop if already reached target
            if c.written_units >= target_words:
                break

            units, label = get_units_and_label(norm, word_mode, detector)
            if units <= 0:
                # e.g., whitespace mode line with no tokens
                continue

            # If fallback just decided and switched to chars, update label + progress bar unit
            if label != c.unit_label:
                c.unit_label = label
                if detector is not None and detector.decided and detector.use_chars:
                    print(
                        f"[info] fallback_chars activated: counting chars instead of whitespace-words (target={target_words:,} chars).",
                        file=sys.stderr
                    )
                    if pbar is not None:
                        pbar.set_postfix_str("mode=chars", refresh=True)
                        pbar.set_description_str(pbar_desc + " (chars)")
                        pbar.unit = "chars"

            fout.write(norm + "\n")
            c.written_lines += 1
            c.written_units += units

            if pbar is not None:
                # don't over-advance beyond total (tqdm is ok with slight over, but keep tidy)
                remaining = max(0, target_words - pbar.n)
                pbar.update(min(units, remaining))

    if pbar is not None:
        pbar.close()

    return c


def run_reservoir_mode_lines(
    infile_xz: str,
    outfile_txt: str,
    flores_set: Set[str],
    target_lines: int,
    log_every: int,
    seed: int,
    near_dup: bool,
    lsh_index: Optional[LSHIndex],
    max_hamming: int,
    show_pbar: bool,
    pbar_desc: str,
) -> Counters:
    """
    Random mode is line-based only (documented restriction).
    """
    rng = random.Random(seed)
    c = Counters(unit_label="lines")

    pbar = None
    if show_pbar and tqdm is not None:
        pbar = tqdm(total=target_lines, unit="lines", dynamic_ncols=True, desc=pbar_desc, leave=True)

    reservoir: List[str] = []
    seen_kept = 0

    with open_xz_text(infile_xz, "rt") as fin:
        for line in fin:
            c.read += 1
            if log_every > 0 and (c.read % log_every == 0):
                print(
                    f"[progress] read={c.read:,} removed={c.removed_overlap:,} kept={c.kept_after_filter:,} reservoir={len(reservoir):,}",
                    file=sys.stderr
                )

            norm = normalize_line(line)
            if not norm:
                continue

            if norm in flores_set:
                c.removed_overlap += 1
                continue

            if near_dup and lsh_index is not None:
                if is_near_duplicate(norm, lsh_index, max_hamming=max_hamming):
                    c.removed_overlap += 1
                    continue

            c.kept_after_filter += 1
            seen_kept += 1

            if len(reservoir) < target_lines:
                reservoir.append(norm)
                if pbar is not None:
                    pbar.update(1)
            else:
                j = rng.randint(1, seen_kept)
                if j <= target_lines:
                    reservoir[j - 1] = norm

    if pbar is not None:
        pbar.close()

    to_write = min(target_lines, len(reservoir))
    if to_write < target_lines:
        print(f"[warn] Only {to_write:,} lines available after filtering; writing all.", file=sys.stderr)

    with open(outfile_txt, "wt", encoding="utf-8", errors="strict", newline="\n") as fout:
        for i in range(to_write):
            fout.write(reservoir[i] + "\n")

    c.written_lines = to_write
    c.written_units = to_write
    return c


# ---------------------------
# Main
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="monolingual_corpus")
    p.add_argument("--output_dir", default="training_corpus")
    p.add_argument("--files", default="", help="Comma-separated filenames (default: all in mapping)")
    p.add_argument("--flores_splits", default="dev,devtest")
    p.add_argument("--sample_mode", choices=["first", "random"], default="first")
    p.add_argument("--seed", type=int, default=42)

    # Word-budget settings (BabyLM-style)
    p.add_argument("--target_words", type=int, default=10_000_000, help="Target word budget (default 10,000,000)")
    p.add_argument("--word_mode", choices=["whitespace", "fallback_chars"], default="whitespace",
                   help="How to count 'words'. fallback_chars switches to char-count if whitespace words are mostly <2.")

    # random mode restriction
    p.add_argument("--target_lines", type=int, default=10_000_000,
                   help="(random mode only) target lines for reservoir sampling (default 10,000,000)")

    # logging/progress
    p.add_argument("--log_every", type=int, default=5_000_000, help="Print log every N input lines (0 disables)")
    p.add_argument("--no_pbar", action="store_true", help="Disable tqdm progress bar")

    # near-duplicate options
    p.add_argument("--near_dup", action="store_true", help="Remove near-duplicates of FLORES (SimHash+LSH)")
    p.add_argument("--max_hamming", type=int, default=4, help="Max Hamming distance for near-dup removal (default: 4)")
    p.add_argument("--simhash_ngram", type=int, default=5, help="Char n-gram size for SimHash (default: 5)")
    p.add_argument("--simhash_bands", type=int, default=4, help="Number of LSH bands (default: 4)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if tqdm is None and not args.no_pbar:
        print("[warn] tqdm not installed; progress bar disabled. Install with: python -m pip install tqdm", file=sys.stderr)
    show_pbar = (not args.no_pbar) and (tqdm is not None)

    splits = [s.strip() for s in args.flores_splits.split(",") if s.strip()]

    if args.files.strip():
        files = [f.strip() for f in args.files.split(",") if f.strip()]
    else:
        files = list(FILE_TO_FLORES_LANG.keys())

    for fname in files:
        if fname not in FILE_TO_FLORES_LANG:
            print(f"[skip] No FLORES mapping for: {fname}", file=sys.stderr)
            continue

        in_path = os.path.join(args.input_dir, fname)
        if not os.path.exists(in_path):
            print(f"[skip] Input not found: {in_path}", file=sys.stderr)
            continue

        lang_code = FILE_TO_FLORES_LANG[fname]
        base = fname.replace(".txt.xz", "")

        if args.sample_mode == "first":
            out_path = os.path.join(args.output_dir, f"{base}.wordbudget_{args.target_words}.txt")
        else:
            out_path = os.path.join(args.output_dir, f"{base}.random_lines_{args.target_lines}.txt")

        print(f"\n[info] Processing {in_path}", file=sys.stderr)
        print(f"[info] FLORES lang={lang_code} splits={splits}", file=sys.stderr)

        if args.sample_mode == "first":
            print(f"[info] Mode=first target_words={args.target_words:,} word_mode={args.word_mode} -> {out_path}", file=sys.stderr)
        else:
            print(f"[info] Mode=random target_lines={args.target_lines:,} seed={args.seed} -> {out_path}", file=sys.stderr)
            print("[note] random mode is line-based ONLY (not word-budget).", file=sys.stderr)

        flores_set = load_flores_membership(lang_code, splits)
        print(f"[info] FLORES unique lines loaded: {len(flores_set):,}", file=sys.stderr)

        lsh_index: Optional[LSHIndex] = None
        if args.near_dup:
            lsh_index = build_lsh_index(flores_set, ngram=args.simhash_ngram, bands=args.simhash_bands)
            print(
                f"[info] Near-dup enabled: max_hamming={args.max_hamming} simhash_ngram={args.simhash_ngram} bands={args.simhash_bands}",
                file=sys.stderr
            )

        pbar_desc = f"{base} ({lang_code}) writing"

        if args.sample_mode == "first":
            c = run_first_mode_wordbudget(
                infile_xz=in_path,
                outfile_txt=out_path,
                flores_set=flores_set,
                target_words=args.target_words,
                word_mode=args.word_mode,
                log_every=args.log_every,
                near_dup=args.near_dup,
                lsh_index=lsh_index,
                max_hamming=args.max_hamming,
                show_pbar=show_pbar,
                pbar_desc=pbar_desc,
            )

            if c.written_units < args.target_words:
                print(
                    f"[warn] Reached EOF before target: written_{c.unit_label}={c.written_units:,} target={args.target_words:,}.",
                    file=sys.stderr
                )

            print("[done] Summary:", file=sys.stderr)
            print(f"  total input lines read        : {c.read:,}", file=sys.stderr)
            print(f"  total removed due to FLORES   : {c.removed_overlap:,}", file=sys.stderr)
            print(f"  total kept after filtering    : {c.kept_after_filter:,}", file=sys.stderr)
            print(f"  total written lines           : {c.written_lines:,}", file=sys.stderr)
            print(f"  total written {c.unit_label:<5}          : {c.written_units:,}", file=sys.stderr)
            print(f"  target {c.unit_label:<5}                : {args.target_words:,}", file=sys.stderr)
            print(f"  output                         : {out_path}", file=sys.stderr)

        else:
            c = run_reservoir_mode_lines(
                infile_xz=in_path,
                outfile_txt=out_path,
                flores_set=flores_set,
                target_lines=args.target_lines,
                log_every=args.log_every,
                seed=args.seed,
                near_dup=args.near_dup,
                lsh_index=lsh_index,
                max_hamming=args.max_hamming,
                show_pbar=show_pbar,
                pbar_desc=pbar_desc,
            )

            if c.written_lines < args.target_lines:
                print(f"[warn] Wrote fewer than target_lines: {c.written_lines:,}/{args.target_lines:,}", file=sys.stderr)

            print("[done] Summary:", file=sys.stderr)
            print(f"  total input lines read        : {c.read:,}", file=sys.stderr)
            print(f"  total removed due to FLORES   : {c.removed_overlap:,}", file=sys.stderr)
            print(f"  total kept after filtering    : {c.kept_after_filter:,}", file=sys.stderr)
            print(f"  total written lines           : {c.written_lines:,}", file=sys.stderr)
            print(f"  output                         : {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
