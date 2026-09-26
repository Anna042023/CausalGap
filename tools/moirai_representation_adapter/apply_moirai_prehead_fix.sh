#!/usr/bin/env bash
set -euo pipefail

ROOT="${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"

cp -a "$HERE/moirai_worker_fixed.py" \
      foundation_workers/moirai_worker.py

python "$HERE/patch_bridge_moirai_rep_name.py"

python -m py_compile \
    foundation_workers/moirai_worker.py \
    causalgap/models/foundation/bridge.py

echo "[PASS] Moirai worker + bridge metadata updated."
echo "IMPORTANT: if metrics.py was changed to uncentered DMIS during debugging, restore the centered paper version before final reruns."
