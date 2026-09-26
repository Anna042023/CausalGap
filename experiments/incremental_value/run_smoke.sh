#!/usr/bin/env bash
set -euo pipefail
PROJECT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
EXP=${H1_ROOT:-$PROJECT/experiment_results/incremental_value_smoke}

python -u experiments/incremental_value/preflight.py --experiment-root "$EXP"
python -u experiments/incremental_value/synthetic_env.py \
  --project-root "$PROJECT" --experiment-root "$EXP" \
  --max-test-windows 8 --eval-seeds 0 --calibration-seeds 1000 1001 1002 1003 1004
python -u experiments/incremental_value/run_h1.py \
  --project-root "$PROJECT" --experiment-root "$EXP" \
  --settings synthetic --models STGCN --model-seeds 0 --eval-seeds 0 \
  --gpus 0 --profile smoke --artifact-level full --max-rep-samples 8 --train-missing

echo "[SMOKE PASS] This validates the data/evaluator path only; one model is not sufficient for H1 inference."
