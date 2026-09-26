#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
D=experiments/sumo_replication
P="${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}/experiment_results/sumo_replication"
python -u "$D/07_check_replication.py" --experiment-root "$P" \
  --model-seeds 0 --eval-seeds 0 1 2 3 4
if [ -n "${SYNTHETIC_ROOT:-}" ]; then
  test -d "$SYNTHETIC_ROOT/runs/synthetic" || { echo "Invalid SYNTHETIC_ROOT: $SYNTHETIC_ROOT";exit 2; }
  python -u "$D/05_regime_replication.py" --sumo-root "$P" --synthetic-root "$SYNTHETIC_ROOT"
else
  echo '[INFO] Synthetic-vs-SUMO comparison skipped; set SYNTHETIC_ROOT to add it.'
  python -u "$D/05_regime_replication.py" --sumo-root "$P"
fi
