#!/usr/bin/env bash
set -euo pipefail
# Example ONLY. Prepare aligned x/y/adj NPZ files first; see configs/external_intervention_format.md.
ROOT=${1:-/data/wanganna/CausalGap/physical_envs}
MODEL=${MODEL:-PatchTST}
DATASET=${DATASET:-METR-LA}
GPU=${GPU:-0}
SEED=${SEED:-0}
CUDA_VISIBLE_DEVICES=${GPU} python run.py intervene \
  --model "${MODEL}" --dataset "${DATASET}" --seed "${SEED}" \
  --external-intervention-root "${ROOT}" \
  --types edge_removal weight_modification
