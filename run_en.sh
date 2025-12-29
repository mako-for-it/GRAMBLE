#!/bin/bash
# Usage: ./run_en.sh <target_lang_code> <mlm_steps> <mt_steps>
# Example: ./run_en.sh es 30000 3000

set -euo pipefail

TGT="$1"
MLM_STEPS="$2"
MT_STEPS="${3:-3000}"

BATCH_SIZE=32
PAIR="${TGT}-en"

echo ">>> PAIR: $PAIR"
echo ">>> MLM_STEPS: $MLM_STEPS"
echo ">>> MT_STEPS : $MT_STEPS"

# 1) folders
mkdir -p "outputs/$PAIR/tokenizer" "outputs/$PAIR/cache" "outputs/$PAIR/mlm" "outputs/$PAIR/mt"
mkdir -p "model/data/processed"
mkdir -p "logs"

# 2) env fixes
export SLURM_PROCID=0 SLURM_NPROCS=1 SLURM_NODEID=0 SLURM_LOCALID=0 SLURM_GPUS_ON_NODE=1
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=12355
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3) build bilingual corpus (interleave lines to keep balance)
COMBINED="model/training_corpus/${PAIR}.wordbudget_20000000.txt"
if [ ! -f "$COMBINED" ]; then
  echo ">>> Building bilingual corpus: $COMBINED"
  python3 - <<PY
import os, itertools

tgt = "${TGT}"
pair = "${PAIR}"
out_path = f"model/training_corpus/{pair}.wordbudget_20000000.txt"
a_path = f"model/training_corpus/{tgt}.wordbudget_10000000.txt"
b_path = "model/training_corpus/en.wordbudget_10000000.txt"

if not os.path.exists(a_path):
    raise SystemExit(f"Missing: {a_path}")
if not os.path.exists(b_path):
    raise SystemExit(f"Missing: {b_path}")

with open(a_path, "r", encoding="utf-8", errors="strict") as fa, \
     open(b_path, "r", encoding="utf-8", errors="strict") as fb, \
     open(out_path, "w", encoding="utf-8", errors="strict", newline="\\n") as fo:
    for la, lb in itertools.zip_longest(fa, fb, fillvalue=""):
        if la:
            fo.write(la.rstrip("\\n") + "\\n")
        if lb:
            fo.write(lb.rstrip("\\n") + "\\n")

print("Wrote:", out_path)
PY
fi

# 4) Create Tokenizer on combined corpus (if missing)
if [ ! -f "outputs/$PAIR/tokenizer/tokenizer.json" ]; then
  echo ">>> Step A: Creating BILINGUAL Tokenizer..."
  python3 model/elc-bert/tokenizers/create_tokenizer.py \
    --input_path "$(pwd)/$COMBINED" \
    --vocab_path "$(pwd)/outputs/$PAIR/tokenizer/tokenizer.json" \
    --vocab_size 30000
fi

# 5) cache on combined corpus
echo ">>> Step B: Starting Cache..."
cd model/elc-bert/pre_training/
python3 cache_dataset.py \
  --segments_path "$(pwd)/../../../$COMBINED" \
  --tokenizer_path "$(pwd)/../../../outputs/$PAIR/tokenizer/tokenizer.json" \
  --sequence_length 128
cd ../../../

# move cache to pair folder (overwrite if exists)
mv -f "$(pwd)/model/data/processed/cached_128.txt" "$(pwd)/outputs/$PAIR/cache/cached_128.txt"

# 6) MLM pretraining (ELC-BERT)
echo ">>> Step C: MLM Training for $MLM_STEPS steps..."
python3 model/elc-bert/train_elc_bert_base.py \
  --input_path "$(pwd)/outputs/$PAIR/cache/cached_128.txt" \
  --vocab_path "$(pwd)/outputs/$PAIR/tokenizer/tokenizer.json" \
  --output_dir "$(pwd)/outputs/$PAIR/mlm/" \
  --config_file "$(pwd)/model/elc-bert/configs/base.json" \
  --max_steps "$MLM_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --seed 42 \
  2>&1 | tee "logs/${PAIR}_mlm_$(date +%F_%H%M).log"


echo ">>> DONE: outputs/$PAIR/"
