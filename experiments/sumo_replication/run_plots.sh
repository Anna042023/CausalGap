#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
D=experiments/sumo_replication
python -u "$D/plot_replication.py" "$@"
