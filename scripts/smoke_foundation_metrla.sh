#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}
TIMEOUT_SEC=${FOUNDATION_SMOKE_TIMEOUT:-600}
GATE_SAMPLES=${FOUNDATION_GATE_SAMPLES:-2}
NUM_SAMPLES=${FOUNDATION_NUM_SAMPLES:-4}
MODELS=(Chronos TimesFM Moirai Lag-Llama)

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "============================================================"
echo "CausalGap Foundation Model QUICK Smoke Gate"
echo "Dataset       : METR-LA"
echo "GPU           : ${GPU}"
echo "Gate windows  : ${GATE_SAMPLES}"
echo "Forecast draws: ${NUM_SAMPLES}"
echo "Timeout/model : ${TIMEOUT_SEC}s"
echo "IMPORTANT     : gate-only; NO full 6850-window evaluation"
echo "============================================================"

for M in "${MODELS[@]}"; do
  echo
  echo "[$(stamp)] ========== ${M} START =========="
  START=$(date +%s)

  # timeout prevents an environment/API mismatch from looking like an endless hang.
  set +e
  timeout --signal=INT "${TIMEOUT_SEC}s" \
    env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    python -u paper_gate.py \
      --model "$M" \
      --dataset METR-LA \
      --track A \
      --seed 0 \
      --profile smoke \
      --gate-samples "$GATE_SAMPLES" \
      --foundation-num-samples "$NUM_SAMPLES"
  RC=$?
  set -e
  if [ "$RC" -ne 0 ]; then
    echo "[$(stamp)] [FAIL] ${M} smoke gate (exit=${RC})"
    exit "$RC"
  fi

  END=$(date +%s)
  echo "[$(stamp)] [PASS] ${M} completed in $((END-START))s"
done

echo
echo "============================================================"
echo "[$(stamp)] [PASS] All 4 foundation quick smoke gates passed."
echo "============================================================"
