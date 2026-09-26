#!/usr/bin/env bash
set -euo pipefail
ROOT="${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}"
EXP_ROOT="${ROOT}/experiment_results/observational_diagnostics_lagllama_smoke"

# Performance smoke only. It uses fewer windows than the final primary_fast run.
python -u experiments/observational_diagnostics/run_diagnostics.py \
  --protocol primary_fast \
  --models Lag-Llama \
  --datasets METR-LA \
  --track B \
  --gpus 0 \
  --project-root "${ROOT}" \
  --experiment-root "${EXP_ROOT}" \
  --diagnostic-node-series-budget 2048 \
  --diagnostic-max-windows 12 \
  --diagnostic-min-windows 8 \
  --foundation-concurrency 1 \
  --foundation-num-samples 100

python -u experiments/observational_diagnostics/check_diagnostics.py \
  --experiment-root "${EXP_ROOT}"
