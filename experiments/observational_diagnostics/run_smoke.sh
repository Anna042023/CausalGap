#!/usr/bin/env bash
set -euo pipefail
ROOT="${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}"
EXP_ROOT="${ROOT}/experiment_results/observational_diagnostics_smoke"

python -u experiments/observational_diagnostics/run_diagnostics.py \
  --protocol primary_fast \
  --models STGCN \
  --datasets METR-LA \
  --track B \
  --gpus 0 \
  --project-root "${ROOT}" \
  --experiment-root "${EXP_ROOT}" \
  --diagnostic-node-series-budget 4096 \
  --diagnostic-max-windows 24 \
  --diagnostic-min-windows 12 \
  --foundation-concurrency 1 \
  --train-missing

python -u experiments/observational_diagnostics/check_diagnostics.py \
  --experiment-root "${EXP_ROOT}"
