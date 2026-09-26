#!/usr/bin/env bash
set -euo pipefail
ROOT=${H1_ROOT:-/data/wanganna/CausalGap/experiment_results/incremental_value}
SETTINGS=${H1_SETTINGS:-"synthetic sumo_metrla"}
BOOT=${H1_BOOTSTRAP:-2000}
python -u experiments/incremental_value/build_h1_units.py \
  --source-experiment-root "$ROOT" --experiment-root "$ROOT" --settings $SETTINGS
python -u experiments/incremental_value/analyze_h1.py \
  --experiment-root "$ROOT" --bootstrap "$BOOT"
