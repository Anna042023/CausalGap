#!/usr/bin/env bash
set -euo pipefail
PROJECT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
EXP=${H1_ROOT:-$PROJECT/experiment_results/incremental_value}
SEEDS=${H1_INTERVENTION_SEEDS:-"0 1 2 3 4"}
MAXWIN=${H1_MAX_TEST_WINDOWS:-32}
BOOT=${H1_BOOTSTRAP:-2000}
SUMO_SRC=${H1_SUMO_SOURCE_ROOT:-}

python -u experiments/incremental_value/preflight.py --experiment-root "$EXP"
python -u experiments/incremental_value/synthetic_env.py \
  --project-root "$PROJECT" --experiment-root "$EXP" \
  --max-test-windows "$MAXWIN" --eval-seeds $SEEDS

if [[ ! -f "$EXP/environments/sumo_metrla/manifest.csv" ]]; then
  if [[ -z "$SUMO_SRC" ]]; then
    echo "[STOP] SUMO-METR-LA has not been staged."
    echo "Set H1_SUMO_SOURCE_ROOT to the calibrated SUMO source directory, then rerun."
    echo "Example: export H1_SUMO_SOURCE_ROOT=/data/wanganna/CausalGap/sumo_h1_source"
    exit 3
  fi
  python -u experiments/incremental_value/prepare_sumo_envs.py \
    --source-root "$SUMO_SRC" --project-root "$PROJECT" --experiment-root "$EXP" \
    --eval-seeds $SEEDS --hardlink
fi

python -u experiments/incremental_value/preflight.py --experiment-root "$EXP" --require-sumo
python -u experiments/incremental_value/run_h1.py \
  --project-root "$PROJECT" --experiment-root "$EXP" \
  --settings synthetic sumo_metrla --model-seeds 0 --eval-seeds $SEEDS \
  --gpus 0 1 --profile paper --artifact-level full --max-rep-samples "$MAXWIN" --train-missing
python -u experiments/incremental_value/build_h1_units.py \
  --source-experiment-root "$EXP" --experiment-root "$EXP" --settings synthetic sumo_metrla --eval-seeds $SEEDS
python -u experiments/incremental_value/analyze_h1.py \
  --experiment-root "$EXP" --bootstrap "$BOOT"
