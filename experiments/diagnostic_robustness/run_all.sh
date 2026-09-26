#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRIPT_DIR="${PROJECT_ROOT}/experiments/diagnostic_robustness"

# Figure 3 is CPU-only and is generated immediately from already completed 1,680-point results.
bash "${SCRIPT_DIR}/plot_dmis_rd.sh"

# Then run the small representative multi-intervention-seed robustness check.
bash "${SCRIPT_DIR}/run_multiseed_robustness.sh"
