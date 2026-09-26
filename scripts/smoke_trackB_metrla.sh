#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}
SUP_TIMEOUT=${SUPERVISED_TRACKB_TIMEOUT:-1800}
FOUND_TIMEOUT=${FOUNDATION_TRACKB_TIMEOUT:-600}
SUP_GATE_SAMPLES=${SUPERVISED_TRACKB_GATE_SAMPLES:-2}
FOUND_GATE_SAMPLES=${FOUNDATION_TRACKB_GATE_SAMPLES:-1}
FOUND_NUM_SAMPLES=${FOUNDATION_TRACKB_NUM_SAMPLES:-2}
FORCE_RETRAIN=${FORCE_RETRAIN_TRACKB:-0}

SUPERVISED=(STGCN DCRNN ASTGNN GWNet AGCRN STEVE DCGCN ST-Norm CCHMM CausalGRIT TF-MVGNN PatchTST iTransformer TimesNet FEDformer DLinear Autoformer)
FOUNDATION=(Chronos TimesFM Moirai Lag-Llama)
RUN_ROOT=/data/wanganna/CausalGap/runs
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "============================================================"
echo "CausalGap 21-model Track-B QUICK Smoke"
echo "Dataset: METR-LA | GPU: ${GPU}"
echo "Foundation models are GATE-ONLY (no full-test evaluation)."
echo "============================================================"

for M in "${SUPERVISED[@]}"; do
  echo
  echo "[$(stamp)] ========== Track-B supervised: ${M} =========="
  CK="${RUN_ROOT}/METR-LA/${M}/trackB/seed0/checkpoint.pt"
  if [[ "$FORCE_RETRAIN" == "1" || ! -f "$CK" ]]; then
    echo "[INFO] smoke Track-B checkpoint missing -> train 1 epoch"
    timeout --signal=INT "${SUP_TIMEOUT}s" \
      env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
      python -u run.py train \
        --model "$M" --dataset METR-LA --track B --seed 0 \
        --profile smoke --epochs 1
  else
    echo "[INFO] reuse existing Track-B smoke checkpoint: $CK"
  fi
  timeout --signal=INT "${SUP_TIMEOUT}s" \
    env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    python -u paper_gate.py \
      --model "$M" --dataset METR-LA --track B --seed 0 \
      --profile smoke --gate-samples "$SUP_GATE_SAMPLES"
done

for M in "${FOUNDATION[@]}"; do
  echo
  echo "[$(stamp)] ========== Track-B zero-shot: ${M} =========="
  # IMPORTANT: never call `run.py evaluate` here. That would evaluate all 6850
  # METR-LA windows x 207 sensors and is not a smoke test.
  timeout --signal=INT "${FOUND_TIMEOUT}s" \
    env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    python -u paper_gate.py \
      --model "$M" --dataset METR-LA --track B --seed 0 \
      --profile smoke --gate-samples "$FOUND_GATE_SAMPLES" \
      --foundation-num-samples "$FOUND_NUM_SAMPLES"
done

echo
echo "============================================================"
echo "[$(stamp)] [PASS] 21-model Track-B quick smoke completed."
echo "============================================================"
