#!/usr/bin/env bash
set -euo pipefail

PROJECT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
CODE=${CAUSALGAP_CODE_ROOT:-$PROJECT/unified}
SYN=${H1_SYNTHETIC_FROZEN_ROOT:-$PROJECT/experiment_results/incremental_value}
SUMO=${H1_SUMO_ROOT:-$PROJECT/experiment_results/sumo_incremental_value}
FINAL=${H1_FINAL_ROOT:-$PROJECT/experiment_results/incremental_value_combined}
COUNTS=${H1_FOUNDATION_COUNTS_CSV:-$CODE/experiments/incremental_value/foundation_parameter_counts_fixed.csv}
SEEDS=${H1_INTERVENTION_SEEDS:-"0 1 2 3 4"}
BOOT=${H1_BOOTSTRAP:-2000}
SRC="$FINAL/source_view"

cd "$CODE"

# Fail loudly instead of silently analyzing incomplete evidence.
test -f "$SYN/environments/synthetic/calibration_distance_reference.csv" || { echo "Missing frozen Synthetic environment: $SYN"; exit 2; }
test -f "$SUMO/environments/sumo_metrla/calibration_distance_reference.csv" || { echo "Missing SUMO environment: $SUMO"; exit 2; }
test -d "$SYN/runs/synthetic" || { echo "Missing frozen Synthetic runs: $SYN/runs/synthetic"; exit 2; }
test -d "$SUMO/runs/sumo_metrla" || { echo "Missing SUMO runs: $SUMO/runs/sumo_metrla"; exit 2; }
test -f "$COUNTS" || { echo "Missing audited Foundation counts CSV: $COUNTS"; exit 2; }

mkdir -p "$SRC/runs" "$SRC/environments"
rm -f "$SRC/runs/synthetic" "$SRC/runs/sumo_metrla" "$SRC/environments/synthetic" "$SRC/environments/sumo_metrla"
ln -s "$SYN/runs/synthetic" "$SRC/runs/synthetic"
ln -s "$SUMO/runs/sumo_metrla" "$SRC/runs/sumo_metrla"
ln -s "$SYN/environments/synthetic" "$SRC/environments/synthetic"
ln -s "$SUMO/environments/sumo_metrla" "$SRC/environments/sumo_metrla"

python -u experiments/sumo_incremental_value/build_h1_units.py \
  --source-experiment-root "$SRC" --experiment-root "$FINAL" \
  --project-root "$PROJECT" --foundation-counts-csv "$COUNTS" \
  --settings synthetic sumo_metrla --eval-seeds $SEEDS

python -u experiments/sumo_incremental_value/analyze_h1.py \
  --experiment-root "$FINAL" --bootstrap "$BOOT"

python -u experiments/sumo_incremental_value/audit_h1.py \
  --experiment-root "$FINAL"

python -u experiments/sumo_incremental_value/print_final_results.py \
  --experiment-root "$FINAL"

printf '\n[DONE] Final combined Synthetic + SUMO H1 analysis: %s\n' "$FINAL"
