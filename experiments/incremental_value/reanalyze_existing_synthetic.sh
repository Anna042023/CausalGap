#!/usr/bin/env bash
set -euo pipefail

PROJECT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
SOURCE=${H1_V1_ROOT:-$PROJECT/experiment_results/incremental_value}
OUT=${H1_V2_ROOT:-$PROJECT/experiment_results/incremental_value}
BOOT=${H1_BOOTSTRAP:-2000}
FOUNDATION_COUNTS=${H1_FOUNDATION_COUNTS_CSV:-}

mkdir -p "$OUT"

BUILD_ARGS=(
  --source-experiment-root "$SOURCE"
  --experiment-root "$OUT"
  --project-root "$PROJECT"
  --settings synthetic
  --eval-seeds 0 1 2 3 4
)
if [[ -n "$FOUNDATION_COUNTS" ]]; then
  BUILD_ARGS+=(--foundation-counts-csv "$FOUNDATION_COUNTS")
fi

python -u experiments/incremental_value/build_h1_units.py "${BUILD_ARGS[@]}"
python -u experiments/incremental_value/analyze_h1.py \
  --experiment-root "$OUT" --bootstrap "$BOOT"
python -u experiments/incremental_value/audit_h1.py \
  --experiment-root "$OUT"

echo "===== V2 CPU-only reanalysis complete ====="
echo "Source V1 raw results: $SOURCE"
echo "V2 outputs:            $OUT"
echo "No model training or intervention inference was rerun."
