#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u run_benchmark.py \
  --only-v33-retrain-required \
  --tracks A \
  --seeds 0 \
  --gpus "${1:-0}" "${2:-1}" \
  --speed-missing-policy masked_zero \
  --speed-missing-input-fill keep_zero \
  2>&1 | tee v33_rerun_changed_models.log
