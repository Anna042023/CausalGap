#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CAUSALGAP_ROOT="${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}"
PRIMARY_ROOT="${CAUSALGAP_S51_ROOT:-/data/wanganna/CausalGap/experiment_results/observational_diagnostics}"
ROBUST_ROOT="${CAUSALGAP_S51_ROBUST_ROOT:-/data/wanganna/CausalGap/experiment_results/diagnostic_robustness}"
DATA_ROOT="${CAUSALGAP_DATA_ROOT:-/data/wanganna/CausalGap/datasets}"
SCRIPT_DIR="${PROJECT_ROOT}/experiments/diagnostic_robustness"
FAST_RUNNER="${PROJECT_ROOT}/experiments/observational_diagnostics/run_diagnostics.py"
CAL_DIR="${PRIMARY_ROOT}/dmis_calibration"
CAL_REF="${CAL_DIR}/dmis_magnitude_calibration_reference.csv"

MODELS=(STGCN STEVE CausalGRIT PatchTST Chronos)
DATASETS=(METR-LA PEMS-BAY)

if [[ ! -f "${FAST_RUNNER}" ]]; then
  echo "ERROR: missing Fast-V2 runner: ${FAST_RUNNER}" >&2
  exit 2
fi

# Verify that the Fast-V2 core patch is actually installed. This prevents the
# old run.py/new experiment-script version mismatch encountered previously.
if ! python "${PROJECT_ROOT}/run.py" intervene --help 2>&1 | grep -q -- "--resume-interventions"; then
  echo "ERROR: ${PROJECT_ROOT}/run.py does not contain the Fast-V2 intervention/resume arguments." >&2
  echo "Use the repository root run.py before starting robustness." >&2
  exit 2
fi
if [[ ! -d "${PRIMARY_ROOT}/work_runs" ]]; then
  echo "ERROR: missing completed primary work_runs: ${PRIMARY_ROOT}/work_runs" >&2
  exit 2
fi

# Do not rerun intervention seed 0. It is already in the completed primary run.
# This dedicated folder computes only seeds 1..4, then analysis merges primary seed 0.
python -u "${FAST_RUNNER}" \
  --protocol multi_intervention_robustness \
  --models "${MODELS[@]}" \
  --datasets "${DATASETS[@]}" \
  --track B \
  --model-seeds 0 \
  --intervention-seeds 1 2 3 4 \
  --gpus 0 1 \
  --project-root "${CAUSALGAP_ROOT}" \
  --experiment-root "${ROBUST_ROOT}" \
  --checkpoint-roots \
      "${PRIMARY_ROOT}/work_runs" \
      "${CAUSALGAP_ROOT}/runs_paper" \
      "${CAUSALGAP_ROOT}/runs" \
  --diagnostic-node-series-budget 8192 \
  --diagnostic-max-windows 64 \
  --diagnostic-min-windows 16 \
  --foundation-concurrency 1 \
  --foundation-num-samples 100 \
  --foundation-cache-dir "${CAUSALGAP_ROOT}/cache/foundation_inference_paper"

if [[ ! -f "${CAL_REF}" ]]; then
  echo "[CALIBRATION] Existing V3 calibration not found; building topology-only reference."
  mkdir -p "${CAL_DIR}"
  python -u "${SCRIPT_DIR}/build_dmis_calibration.py" \
    --output-dir "${CAL_DIR}" \
    --data-root "${DATA_ROOT}" \
    --seeds 0-19
else
  echo "[REUSE] calibration reference: ${CAL_REF}"
fi

python -u "${SCRIPT_DIR}/analyze_multiseed_robustness.py" \
  --primary-root "${PRIMARY_ROOT}" \
  --robustness-root "${ROBUST_ROOT}" \
  --calibration-reference "${CAL_REF}" \
  --models "${MODELS[@]}" \
  --datasets "${DATASETS[@]}" \
  --seeds 0 1 2 3 4 \
  --bootstrap 2000 \
  --bootstrap-seed 2051

echo "================================================================================"
echo "DONE: targeted robustness check"
echo "Robustness root: ${ROBUST_ROOT}"
echo "Main table: ${ROBUST_ROOT}/analysis/tables/Table_R1_multiseed_mean_DMIScal_RD.csv"
echo "Seed-wise:  ${ROBUST_ROOT}/analysis/tables/Table_R2_seedwise_DMIScal_RD.csv"
echo "Stability:  ${ROBUST_ROOT}/analysis/tables/Table_R5_seed_sign_stability.csv"
echo "================================================================================"
