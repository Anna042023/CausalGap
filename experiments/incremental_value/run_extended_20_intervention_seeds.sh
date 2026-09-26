#!/usr/bin/env bash
set -euo pipefail
PROJECT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
EXP=${H1_ROOT:-$PROJECT/experiment_results/incremental_value_extended}
SUMO_SRC=${H1_SUMO_SOURCE_ROOT:?Set H1_SUMO_SOURCE_ROOT to the calibrated SUMO source directory}
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

# Extended robustness: full test split and all 20 intervention seeds, still one model-training seed. This is intentionally expensive and is not the default.
python -u experiments/incremental_value/synthetic_env.py \
  --project-root "$PROJECT" --experiment-root "$EXP" --max-test-windows 0 --eval-seeds $SEEDS
python -u experiments/incremental_value/prepare_sumo_envs.py \
  --source-root "$SUMO_SRC" --project-root "$PROJECT" --experiment-root "$EXP" --eval-seeds $SEEDS --hardlink
python -u experiments/incremental_value/run_h1.py \
  --project-root "$PROJECT" --experiment-root "$EXP" \
  --settings synthetic sumo_metrla --model-seeds 0 --eval-seeds $SEEDS \
  --gpus 0 1 --profile paper --artifact-level full --max-rep-samples 256 --train-missing
python -u experiments/incremental_value/build_h1_units.py \
  --experiment-root "$EXP" --settings synthetic sumo_metrla --eval-seeds $SEEDS
python -u experiments/incremental_value/analyze_h1.py \
  --experiment-root "$EXP" --bootstrap 2000
