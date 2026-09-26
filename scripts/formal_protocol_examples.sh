#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Paper-aligned structural protocol (run.py defaults):
#   edge removal: s={0.05,0.10,0.20}, top-40% betweenness candidate pool, uniform without replacement
#   weight modification: 15% edges, gamma={0.8,0.5,0.2}
#   node isolation: s={0.1,0.2,0.4,0.6}, X unchanged, evaluate unisolated nodes
#   intervention seeds: 0..19, independent of model-training seed
#
# Example:
# CUDA_VISIBLE_DEVICES=0 python run.py intervene --model STGCN --dataset METR-LA \
#   --types edge_removal weight_modification node_isolation \
#   --edge-removal-mode paper_top40_uniform --seed 0 --artifact-level minimal

# Model-training seeds required by the draft reproducibility checklist are 0..9.
