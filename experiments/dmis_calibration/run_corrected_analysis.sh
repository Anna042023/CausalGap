#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXPERIMENT_ROOT="${CAUSALGAP_S51_ROOT:-/data/wanganna/CausalGap/experiment_results/observational_diagnostics}"
DATA_ROOT="${CAUSALGAP_DATA_ROOT:-/data/wanganna/CausalGap/datasets}"
SCRIPT_DIR="${PROJECT_ROOT}/experiments/dmis_calibration"
CAL_DIR="${EXPERIMENT_ROOT}/dmis_calibration"
REF="${CAL_DIR}/dmis_magnitude_calibration_reference.csv"

echo "================================================================================"
echo "CausalGap Section 5.1 corrected analysis v3"
echo "Existing expensive results: ${EXPERIMENT_ROOT}/work_runs"
echo "Calibration output        : ${CAL_DIR}"
echo "Corrected analysis output : ${EXPERIMENT_ROOT}/analysis_calibrated"
echo "GPU/model inference       : NONE"
echo "================================================================================"

if [[ ! -d "${EXPERIMENT_ROOT}/work_runs" ]]; then
  echo "ERROR: missing existing Fast-V2 work_runs: ${EXPERIMENT_ROOT}/work_runs" >&2
  exit 2
fi

if [[ ! -f "${REF}" ]]; then
  python -u "${SCRIPT_DIR}/build_dmis_calibration.py" \
    --output-dir "${CAL_DIR}" \
    --data-root "${DATA_ROOT}" \
    --seeds 0-19
else
  echo "[REUSE] existing magnitude calibration reference: ${REF}"
fi

python -u "${SCRIPT_DIR}/analyze_calibrated_diagnostics.py" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --calibration-reference "${REF}" \
  --track B \
  --bootstrap 2000 \
  --bootstrap-seed 2026

echo "================================================================================"
echo "DONE. Important outputs:"
echo "  ${EXPERIMENT_ROOT}/analysis_calibrated/tables/Table1_DMIScal_RD_Association.csv"
echo "  ${EXPERIMENT_ROOT}/analysis_calibrated/figures/Figure3_DMIScal_vs_RD.pdf"
echo "  ${EXPERIMENT_ROOT}/analysis_calibrated/diagnostics/correlation_within_intervention_family_DMISraw.csv"
echo "  ${EXPERIMENT_ROOT}/analysis_calibrated/diagnostics/correlation_by_dataset_and_family_DMISraw.csv"
echo "================================================================================"
