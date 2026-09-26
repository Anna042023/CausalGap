#!/usr/bin/env bash
set -euo pipefail

ROOT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
PREFIX=${TIMESFM_ENV:-/data/wanganna/envs/timesfm}
CONDA_CHANNEL=${CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}
PIP_INDEX=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
TORCH_INDEX=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}
SRC="$ROOT/class5/TimesFM/v1"

if [[ -n "${CONDA_EXE:-}" ]]; then CONDA="$CONDA_EXE"; else CONDA=$(command -v conda || true); fi
[[ -n "$CONDA" ]] || { echo '[FAIL] conda not found'; exit 2; }
[[ -d "$SRC" ]] || { echo "[FAIL] TimesFM v1 source not found: $SRC"; exit 2; }

if [[ ! -x "$PREFIX/bin/python" ]]; then
  "$CONDA" create -y -p "$PREFIX" python=3.11 pip \
    --override-channels -c "$CONDA_CHANNEL"
else
  echo "[INFO] Reusing existing env: $PREFIX"
fi

"$PREFIX/bin/python" -m pip install -U pip setuptools wheel poetry-core -i "$PIP_INDEX"

# TimesFM v1 PyTorch extra requires Python 3.11; pin CUDA 12.4 PyTorch explicitly.
"$PREFIX/bin/python" -m pip install 'torch==2.6.0' --index-url "$TORCH_INDEX"

# Install only the PyTorch-path runtime dependencies. JAX/Pax are intentionally omitted;
# TimesFM's __init__ falls back to TimesFmTorch when JAX/Pax are unavailable.
"$PREFIX/bin/python" -m pip install \
  'einshape>=1.0.0' \
  'numpy>=1.26.4,<3' \
  'pandas>=2,<3' \
  'utilsforecast>=0.1.10' \
  'huggingface_hub[cli]>=0.23' \
  'scikit-learn>=1.2.2' \
  'typer>=0.12.3' \
  'wandb>=0.17.5' \
  'absl-py>=1.4.0' \
  'safetensors>=0.5.3' \
  -i "$PIP_INDEX"

"$PREFIX/bin/python" -m pip install -e "$SRC" --no-deps -i "$PIP_INDEX"
"$PREFIX/bin/python" -m pip check

"$PREFIX/bin/python" - <<'PY'
import torch
import timesfm
print('[OK] TimesFM import')
print('TimesFM class =', timesfm.TimesFm)
print('torch =', torch.__version__)
print('torch cuda =', torch.version.cuda)
print('cuda available =', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('[FAIL] TimesFM environment cannot access CUDA')
print('GPU count =', torch.cuda.device_count())
print('GPU 0 =', torch.cuda.get_device_name(0))
PY
