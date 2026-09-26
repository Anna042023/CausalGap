#!/usr/bin/env bash
set -euo pipefail
PROJECT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
CODE=${CAUSALGAP_CODE_ROOT:-$PROJECT/unified}
EXP=${H1_SUMO_ROOT:-$PROJECT/experiment_results/sumo_incremental_value}
SEEDS=${H1_INTERVENTION_SEEDS:-"0 1 2 3 4"}
COUNTS=${H1_FOUNDATION_COUNTS_CSV:-$CODE/experiments/incremental_value/foundation_parameter_counts_fixed.csv}
BOOT=${H1_BOOTSTRAP:-2000}

cd "$CODE"
python -u experiments/sumo_incremental_value/build_h1_units.py \
  --source-experiment-root "$EXP" --experiment-root "$EXP" \
  --project-root "$PROJECT" --foundation-counts-csv "$COUNTS" \
  --settings sumo_metrla --eval-seeds $SEEDS
python -u experiments/sumo_incremental_value/analyze_h1.py \
  --experiment-root "$EXP" --bootstrap "$BOOT"
python -u experiments/sumo_incremental_value/audit_h1.py \
  --experiment-root "$EXP"

printf '\n[DONE] SUMO-only diagnostic analysis complete.\n'
