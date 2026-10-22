#!/usr/bin/env bash
# Reproduces the multi-task vs single-task comparison in the README.
# CPU budget: distilroberta-base, embeddings + bottom 3 of 6 layers frozen,
# 9,000 training examples per task, 3 epochs, full official test sets.
set -euo pipefail
COMMON="--encoder distilroberta-base --freeze_layers 3 --max_train_per_task 9000 --epochs 3 --batch_size 32 --lr 5e-5 --seed 42"

for run in "mtl intent,emotion" "stl_intent intent" "stl_emotion emotion"; do
  set -- $run
  python -m training.train --tasks "$2" $COMMON --output_dir "runs/$1" 2>&1 | tee "runs/$1.log"
  mkdir -p results && cp "runs/$1/metrics.json" "results/$1.json"
done
python -m scripts.summarize_results
