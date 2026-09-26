#!/usr/bin/env bash
set -euo pipefail

ROOT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
PREFIX=${CHRONOS_ENV:-/data/wanganna/envs/chronos}
CONDA_CHANNEL=${CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}
PIP_INDEX=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
TORCH_INDEX=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}
SRC="$ROOT/class5/Chronos"

if [[ -n "${CONDA_EXE:-}" ]]; then CONDA="$CONDA_EXE"; else CONDA=$(command -v conda || true); fi
[[ -n "$CONDA" ]] || { echo '[FAIL] conda not found'; exit 2; }
[[ -d "$SRC" ]] || { echo "[FAIL] Chronos source not found: $SRC"; exit 2; }

if [[ ! -x "$PREFIX/bin/python" ]]; then
  "$CONDA" create -y -p "$PREFIX" python=3.10 pip \
    --override-channels -c "$CONDA_CHANNEL"
else
  echo "[INFO] Reusing existing env: $PREFIX"
fi

"$PREFIX/bin/python" -m pip install -U pip setuptools wheel -i "$PIP_INDEX"

# Pin a driver-compatible CUDA build first. Do not let the local package upgrade torch.
"$PREFIX/bin/python" -m pip install 'torch==2.6.0' --index-url "$TORCH_INDEX"

"$PREFIX/bin/python" -m pip install \
  'transformers>=4.41,<6' \
  'accelerate>=1.1,<2' \
  'numpy>=1.21,<3' \
  'einops>=0.7,<1' \
  'pandas>=2,<4' \
  'safetensors>=0.4' \
  -i "$PIP_INDEX"

"$PREFIX/bin/python" -m pip install -e "$SRC" --no-deps -i "$PIP_INDEX"
"$PREFIX/bin/python" -m pip check

"$PREFIX/bin/python" - <<'PY'
import chronos, torch
print('[OK] Chronos import')
print('torch =', torch.__version__)
print('torch cuda =', torch.version.cuda)
print('cuda available =', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('[FAIL] Chronos environment cannot access CUDA')
print('GPU count =', torch.cuda.device_count())
print('GPU 0 =', torch.cuda.get_device_name(0))
PY
