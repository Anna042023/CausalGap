#!/usr/bin/env bash
set -euo pipefail

PROJECT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
CODE=${CAUSALGAP_CODE_ROOT:-$PROJECT/unified}
EXP=${H1_SUMO_ROOT:-$PROJECT/experiment_results/sumo_incremental_value}
SUMO_SRC=${H1_SUMO_SOURCE_ROOT:-$PROJECT/sumo_h1_source}
SEEDS=${H1_INTERVENTION_SEEDS:-"0 1 2 3 4"}
CACHE=${H1_SUMO_CACHE_ROOT:-$PROJECT/cache/sumo_incremental_value_cache}
MAXREP=${H1_MAX_REP_SAMPLES:-32}

cd "$CODE"

# Stage/validate the intervention-consistent SUMO source every time. This is cheap and
# prevents accidentally running models against stale or incomplete environments.
python -u experiments/sumo_incremental_value/check_sumo_source.py \
  --source-root "$SUMO_SRC" --eval-seeds $SEEDS

python -u experiments/sumo_incremental_value/prepare_sumo_envs.py \
  --source-root "$SUMO_SRC" --project-root "$PROJECT" --experiment-root "$EXP" \
  --eval-seeds $SEEDS --hardlink

python -u experiments/sumo_incremental_value/preflight.py \
  --experiment-root "$EXP" --require-sumo

mkdir -p "$CACHE/primary"

# Two-GPU schedule only (GPU 0/1). Foundation models are serialized because each
# launches its own official worker and is substantially heavier than ordinary models.
python -u experiments/sumo_incremental_value/run_h1.py \
  --project-root "$PROJECT" --experiment-root "$EXP" \
  --settings sumo_metrla --model-seeds 0 --eval-seeds $SEEDS \
  --gpus 0 1 --profile paper --artifact-level full --max-rep-samples "$MAXREP" \
  --foundation-num-samples 100 --foundation-concurrency 1 \
  --foundation-cache-dir "$CACHE/primary" --train-missing

printf '\n[DONE] All SUMO-METR-LA H1 raw jobs completed.\n'
printf 'Next: bash experiments/sumo_incremental_value/analyze_sumo_only.sh\n'
