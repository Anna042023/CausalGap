#!/usr/bin/env bash
set -euo pipefail
GPU=${1:-0}
cd "$(dirname "$0")/.."
for M in CausalGRIT TF-MVGNN; do
  echo "========== ${M} Track-A =========="
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u run.py train --model "$M" --dataset METR-LA --track A --seed 0 --profile smoke --epochs 1
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u paper_gate.py --model "$M" --dataset METR-LA --track A --seed 0 --profile smoke --gate-samples 2
  echo "========== ${M} Track-B =========="
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u run.py train --model "$M" --dataset METR-LA --track B --seed 0 --profile smoke --epochs 1
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u paper_gate.py --model "$M" --dataset METR-LA --track B --seed 0 --profile smoke --gate-samples 2
done
