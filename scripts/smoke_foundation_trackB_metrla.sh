#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}
TIMEOUT_SEC=${FOUNDATION_TRACKB_TIMEOUT:-600}
GATE_SAMPLES=${FOUNDATION_TRACKB_GATE_SAMPLES:-1}
NUM_SAMPLES=${FOUNDATION_TRACKB_NUM_SAMPLES:-2}
MODELS=(Chronos TimesFM Moirai Lag-Llama)

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "============================================================"
echo "CausalGap Foundation Track-B QUICK Gate"
echo "Dataset       : METR-LA"
echo "GPU           : ${GPU}"
echo "Gate windows  : ${GATE_SAMPLES}"
echo "Forecast draws: ${NUM_SAMPLES}"
echo "Timeout/model : ${TIMEOUT_SEC}s"
echo "IMPORTANT     : gate-only; NO run.py evaluate/intervene"
echo "============================================================"

for M in "${MODELS[@]}"; do
  echo
  echo "[$(stamp)] ========== Track-B zero-shot: ${M} START =========="
  START=$(date +%s)
  set +e
  timeout --signal=INT "${TIMEOUT_SEC}s" \
    env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    python -u paper_gate.py \
      --model "$M" \
      --dataset METR-LA \
      --track B \
      --seed 0 \
      --profile smoke \
      --gate-samples "$GATE_SAMPLES" \
      --foundation-num-samples "$NUM_SAMPLES"
  RC=$?
  set -e
  if [ "$RC" -ne 0 ]; then
    echo "[$(stamp)] [FAIL] ${M} Track-B gate (exit=${RC})"
    exit "$RC"
  fi
  END=$(date +%s)
  echo "[$(stamp)] [PASS] ${M} Track-B completed in $((END-START))s"
done

echo
echo "============================================================"
echo "[$(stamp)] [PASS] All 4 foundation Track-B quick gates passed."
echo "============================================================"
