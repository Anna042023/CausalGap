#!/usr/bin/env bash
set -euo pipefail
GPU=${1:-0}
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 python -u paper_gate.py \
  --model "Moirai" --dataset METR-LA --track A --seed 0 --profile smoke \
  --gate-samples "${FOUNDATION_GATE_SAMPLES:-2}" \
  --foundation-num-samples "${FOUNDATION_NUM_SAMPLES:-4}"
