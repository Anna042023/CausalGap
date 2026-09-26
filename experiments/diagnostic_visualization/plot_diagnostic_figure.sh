#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/data/wanganna/CausalGap}
UNIFIED_ROOT=${2:-${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
ZOOM_XMAX=${S51_FIG3_V6_ZOOM_XMAX:-5}
POINT_SIZE=${S51_FIG3_V6_POINT_SIZE:-28}
ALPHA=${S51_FIG3_V6_ALPHA:-0.62}

cd "$UNIFIED_ROOT"

python -u experiments/diagnostic_visualization/plot_diagnostic_figure.py \
  --project-root "$ROOT" \
  --zoom-xmax "$ZOOM_XMAX" \
  --point-size "$POINT_SIZE" \
  --alpha "$ALPHA"
