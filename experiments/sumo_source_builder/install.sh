#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
DST=${1:-${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/experiments/sumo_source_builder}}
mkdir -p "$(dirname "$DST")"
rm -rf "$DST"
cp -a "$SRC_DIR" "$DST"
echo "[PASS] installed -> $DST"
