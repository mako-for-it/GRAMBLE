#!/bin/bash
# Usage: ./run_lang.sh <lang_code> <steps>

LANG_CODE=$1
MAX_STEPS=$2
BATCH_SIZE=32

echo ">>> PROCESSING LANGUAGE: $LANG_CODE"
echo ">>> STEPS: $MAX_STEPS"

# 1. Setup Folders
mkdir -p outputs/$LANG_CODE/tokenizer outputs/$LANG_CODE/cache outputs/$LANG_CODE/model
mkdir -p model/data/processed/

# 2. Environment Fixes (CUDA, Memory, DDP)
export SLURM_PROCID=0 SLURM_NPROCS=1 SLURM_NODEID=0 SLURM_LOCALID=0 SLURM_GPUS_ON_NODE=1
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=12355
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. Create Tokenizer (if missing)
if [ ! -f "outputs/$LANG_CODE/tokenizer/tokenizer.json" ]; then
    echo ">>> Step A: Creating Tokenizer..."
    python3 model/elc-bert/tokenizers/create_tokenizer.py \
      --input_path $(pwd)/model/training_corpus/$LANG_CODE.wordbudget_10000000.txt \
      --vocab_path $(pwd)/outputs/$LANG_CODE/tokenizer/tokenizer.json \
      --vocab_size 30000
fi

# 4. Create Cache (with the directory context fix)
echo ">>> Step B: Starting Cache..."
cd model/elc-bert/pre_training/
python3 cache_dataset.py \
  --segments_path $(pwd)/../../../model/training_corpus/$LANG_CODE.wordbudget_10000000.txt \
  --tokenizer_path $(pwd)/../../../outputs/$LANG_CODE/tokenizer/tokenizer.json \
  --sequence_length 128
cd ../../../
mv $(pwd)/model/data/processed/cached_128.txt $(pwd)/outputs/$LANG_CODE/cache/cached_128.txt

# 5. Long Training Run
echo ">>> Step C: Training for $MAX_STEPS steps..."
python3 model/elc-bert/train_elc_bert_base.py \
  --input_path $(pwd)/outputs/$LANG_CODE/cache/cached_128.txt \
  --vocab_path $(pwd)/outputs/$LANG_CODE/tokenizer/tokenizer.json \
  --output_dir $(pwd)/outputs/$LANG_CODE/model/ \
  --config_file $(pwd)/model/elc-bert/configs/base.json \
  --max_steps $MAX_STEPS \
  --batch_size $BATCH_SIZE \
  --seed 42

# 6. Inference and Evaluation
echo ">>> Step D: Running Inference Evaluation..."
python3 ./inference.py --lang $LANG_CODE --samples 100