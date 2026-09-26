#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PRIMARY_ROOT="${CAUSALGAP_S51_ROOT:-/data/wanganna/CausalGap/experiment_results/observational_diagnostics}"
SCRIPT_DIR="${PROJECT_ROOT}/experiments/diagnostic_robustness"

python -u "${SCRIPT_DIR}/plot_dmis_rd.py" \
  --experiment-root "${PRIMARY_ROOT}" \
  --inset-xmax "${S51_FIG3_INSET_XMAX:-5.0}" \
  --bootstrap "${S51_FIG3_BOOTSTRAP:-1000}" \
  --bootstrap-seed 2041

echo "Final Figure 3: ${PRIMARY_ROOT}/analysis_final/figures/Figure3_DMIScal_vs_RD.pdf"
