#!/usr/bin/env bash
set -euo pipefail

PROJECT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
CODE=${CAUSALGAP_CODE_ROOT:-$PROJECT/unified}
EXP=${H1_SUMO_ROOT:-$PROJECT/experiment_results/sumo_incremental_value}
SUMO_SRC=${H1_SUMO_SOURCE_ROOT:-$PROJECT/sumo_h1_source}
SEEDS=${H1_INTERVENTION_SEEDS:-"0 1 2 3 4"}
CACHE=${H1_SUMO_CACHE_ROOT:-$PROJECT/cache/sumo_incremental_value_cache}

cd "$CODE"

python -u experiments/sumo_incremental_value/check_sumo_source.py \
  --source-root "$SUMO_SRC" --eval-seeds $SEEDS

python -u experiments/sumo_incremental_value/prepare_sumo_envs.py \
  --source-root "$SUMO_SRC" --project-root "$PROJECT" --experiment-root "$EXP" \
  --eval-seeds $SEEDS --hardlink

python -u experiments/sumo_incremental_value/preflight.py \
  --experiment-root "$EXP" --require-sumo

mkdir -p "$CACHE/smoke"
python -u experiments/sumo_incremental_value/run_h1.py \
  --project-root "$PROJECT" --experiment-root "$EXP" \
  --settings sumo_metrla --models STGCN --model-seeds 0 --eval-seeds 0 \
  --gpus 0 --profile paper --artifact-level full --max-rep-samples 32 \
  --foundation-num-samples 100 --foundation-concurrency 1 \
  --foundation-cache-dir "$CACHE/smoke" --train-missing

printf '\n[PASS] SUMO smoke finished. If no error occurred, run run_sumo_primary.sh.\n'
