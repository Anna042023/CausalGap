#!/usr/bin/env bash
set -euo pipefail

# Section 5.1 compute-reduced main association protocol.
# Server has two GPUs: 0 and 1.
# Run from the unified repository root.
ROOT="${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}"
EXP_ROOT="${ROOT}/experiment_results/observational_diagnostics"

python -u experiments/observational_diagnostics/run_diagnostics.py \
  --protocol primary_fast \
  --track B \
  --gpus 0 1 \
  --project-root "${ROOT}" \
  --experiment-root "${EXP_ROOT}" \
  --diagnostic-node-series-budget 8192 \
  --diagnostic-max-windows 64 \
  --diagnostic-min-windows 16 \
  --foundation-concurrency 1 \
  --foundation-num-samples 100 \
  --train-missing

python -u experiments/observational_diagnostics/check_diagnostics.py \
  --experiment-root "${EXP_ROOT}"

python -u experiments/observational_diagnostics/analyze_diagnostics.py \
  --experiment-root "${EXP_ROOT}" \
  --track B \
  --bootstrap 2000 \
  --bootstrap-seed 2026
