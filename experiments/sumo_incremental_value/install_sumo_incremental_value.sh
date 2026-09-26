#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=${1:-${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
DEST="$PROJECT/experiments/sumo_incremental_value"
mkdir -p "$DEST"
cp -a "$SRC_DIR"/*.py "$SRC_DIR"/*.sh "$SRC_DIR"/README.md "$DEST"/
chmod +x "$DEST"/*.sh "$DEST"/*.py || true
printf '[DONE] installed to %s\n' "$DEST"
