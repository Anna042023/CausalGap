#!/usr/bin/env bash
set -euo pipefail
GPU=${1:-0}
MODELS=(STGCN DCRNN ASTGNN GWNet AGCRN STEVE DCGCN ST-Norm CCHMM CausalGRIT TF-MVGNN PatchTST iTransformer TimesNet FEDformer DLinear Autoformer)
for M in "${MODELS[@]}"; do
  echo "========== $M =========="
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u run.py train --model "$M" --dataset METR-LA --track A --seed 0 --profile smoke --epochs 1
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u paper_gate.py --model "$M" --dataset METR-LA --track A --seed 0 --profile smoke --gate-samples 2
done
