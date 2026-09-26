#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u repair_saved_metrics.py \
  --run-root /data/wanganna/CausalGap/runs_paper \
  --speed-missing-policy masked_zero \
  --speed-missing-input-fill keep_zero
python collect_results.py --tracks A --seeds 0
