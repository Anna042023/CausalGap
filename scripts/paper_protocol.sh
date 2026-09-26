#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Example 1: native Track-A model
CUDA_VISIBLE_DEVICES=0 python run.py train --model STGCN --dataset METR-LA --track A --seed 0
CUDA_VISIBLE_DEVICES=0 python run.py intervene --model STGCN --dataset METR-LA --track A --seed 0 \
  --edge-removal-mode paper_top40_uniform --edge-pool-ratio 0.40

# Example 2: information-equivalent Track-B graph-free model
CUDA_VISIBLE_DEVICES=0 python run.py train --model PatchTST --dataset METR-LA --track B --seed 0
CUDA_VISIBLE_DEVICES=0 python run.py intervene --model PatchTST --dataset METR-LA --track B --seed 0 \
  --edge-removal-mode paper_top40_uniform --edge-pool-ratio 0.40

# Example 3: architecture-complete PHYSICAL structural evaluation.
# Prepare the SAME SUMO/synthetic/I-405 Xk/Yk/Ak environments for every model first.
# External files should follow the same intensity x seed plan (seeds 0..19).
# CUDA_VISIBLE_DEVICES=0 python run.py intervene --model PatchTST --dataset METR-LA --track A --seed 0 \
#   --types edge_removal weight_modification node_isolation \
#   --external-intervention-root ${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}/physical_envs --require-external-structural
