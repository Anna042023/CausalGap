#!/usr/bin/env bash
set -e
MODELS=(PatchTST iTransformer TimesNet FEDformer DLinear Autoformer)
for M in "${MODELS[@]}"; do
  echo "===== $M / METR-LA ====="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python run.py train --model "$M" --dataset METR-LA --epochs 50 --batch-size 32
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python run.py intervene --model "$M" --dataset METR-LA --num-interventions 100 --batch-size 32
done
