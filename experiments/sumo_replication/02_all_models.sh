#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
D=experiments/sumo_replication
P="${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}/experiment_results/sumo_replication"
python -u "$D/model_eval/run_replicate.py" --settings sumo_metrla \
  --experiment-root "$P" --model-seeds 0 --eval-seeds 0 1 2 3 4 \
  --gpus 0 1 --profile paper --max-rep-samples 2 \
  --train-missing --artifact-level minimal --foundation-concurrency 1 \
  --foundation-cache-dir ${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}/cache/sumo_foundation
python -u "$D/07_check_replication.py" --experiment-root "$P" \
  --model-seeds 0 --eval-seeds 0 1 2 3 4
