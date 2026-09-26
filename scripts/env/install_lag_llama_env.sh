#!/usr/bin/env bash
set -euo pipefail

ROOT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
PREFIX=${LAGLLAMA_ENV:-/data/wanganna/envs/lagllama}
CONDA_CHANNEL=${CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}
PIP_INDEX=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
TORCH_INDEX=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}
SRC="$ROOT/class5/Lag-Llama"

if [[ -n "${CONDA_EXE:-}" ]]; then CONDA="$CONDA_EXE"; else CONDA=$(command -v conda || true); fi
[[ -n "$CONDA" ]] || { echo '[FAIL] conda not found'; exit 2; }
[[ -d "$SRC" ]] || { echo "[FAIL] Lag-Llama source not found: $SRC"; exit 2; }

if [[ ! -x "$PREFIX/bin/python" ]]; then
  "$CONDA" create -y -p "$PREFIX" python=3.10 pip \
    --override-channels -c "$CONDA_CHANNEL"
else
  echo "[INFO] Reusing existing env: $PREFIX"
fi

"$PREFIX/bin/python" -m pip install -U pip wheel -i "$PIP_INDEX"
# Lag-Llama/Lightning imports pkg_resources; pin a setuptools release that still ships it.
"$PREFIX/bin/python" -m pip install "setuptools==69.5.1" -i "$PIP_INDEX"

# Lag-Llama is an older GluonTS/Lightning stack; torch 2.4.1+cu124 is a safer pinned runtime.
"$PREFIX/bin/python" -m pip install \
  'torch==2.4.1' 'torchvision==0.19.1' 'torchaudio==2.4.1' \
  --index-url "$TORCH_INDEX"

"$PREFIX/bin/python" -m pip install \
  'numpy==1.26.4' \
  'gluonts[torch]==0.14.4' \
  'lightning>=2.0,<2.5' \
  'pytorch-lightning>=2.0,<2.5' \
  'wandb' \
  'scipy' \
  'pandas>=2,<3' \
  'huggingface_hub[cli]>=0.23' \
  'matplotlib' \
  -i "$PIP_INDEX"

"$PREFIX/bin/python" -m pip install -e "$SRC" --no-deps -i "$PIP_INDEX"
"$PREFIX/bin/python" -m pip check

"$PREFIX/bin/python" - <<'PY'
import torch
import lag_llama
print('[OK] Lag-Llama import')
print('torch =', torch.__version__)
print('torch cuda =', torch.version.cuda)
print('cuda available =', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('[FAIL] Lag-Llama environment cannot access CUDA')
print('GPU count =', torch.cuda.device_count())
print('GPU 0 =', torch.cuda.get_device_name(0))
PY
