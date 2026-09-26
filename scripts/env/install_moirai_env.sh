#!/usr/bin/env bash
set -euo pipefail

ROOT=${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}
PREFIX=${MOIRAI_ENV:-/data/wanganna/envs/moirai}
CONDA_CHANNEL=${CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}
PIP_INDEX=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
TORCH_INDEX=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}
SRC="$ROOT/class5/Moirai/uni2ts-main"

if [[ -n "${CONDA_EXE:-}" ]]; then CONDA="$CONDA_EXE"; else CONDA=$(command -v conda || true); fi
[[ -n "$CONDA" ]] || { echo '[FAIL] conda not found'; exit 2; }
[[ -d "$SRC" ]] || { echo "[FAIL] Moirai/uni2ts source not found: $SRC"; exit 2; }

if [[ ! -x "$PREFIX/bin/python" ]]; then
  "$CONDA" create -y -p "$PREFIX" python=3.10 pip \
    --override-channels -c "$CONDA_CHANNEL"
else
  echo "[INFO] Reusing existing env: $PREFIX"
fi

"$PREFIX/bin/python" -m pip install -U pip setuptools wheel hatchling -i "$PIP_INDEX"

# The uploaded uni2ts source requires torch>=2.1,<2.5, so use the newest compatible cu124 wheel.
"$PREFIX/bin/python" -m pip install \
  'torch==2.4.1' 'torchvision==0.19.1' 'torchaudio==2.4.1' \
  --index-url "$TORCH_INDEX"

"$PREFIX/bin/python" -m pip install \
  'lightning>=2.0,<2.5' \
  'gluonts==0.14.3' \
  'numpy==1.26.4' \
  'scipy==1.11.3' \
  'einops>=0.7,<0.8' \
  'jaxtyping>=0.2.24,<0.3' \
  'python-dotenv==1.0.0' \
  'hydra-core>=1.3,<1.4' \
  'orjson' \
  'tensorboard' \
  'multiprocess' \
  'huggingface_hub>=0.23' \
  'safetensors' \
  'datasets>=2.17,<2.18' \
  'jax==0.4.30' 'jaxlib==0.4.30' \
  -i "$PIP_INDEX"

"$PREFIX/bin/python" -m pip install -e "$SRC" --no-deps -i "$PIP_INDEX"
"$PREFIX/bin/python" -m pip check

"$PREFIX/bin/python" - <<'PY'
import torch, uni2ts
print('[OK] Moirai/uni2ts import')
print('torch =', torch.__version__)
print('torch cuda =', torch.version.cuda)
print('cuda available =', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('[FAIL] Moirai environment cannot access CUDA')
print('GPU count =', torch.cuda.device_count())
print('GPU 0 =', torch.cuda.get_device_name(0))
PY
