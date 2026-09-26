#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${CAUSALGAP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
D=experiments/sumo_replication
P="${CAUSALGAP_ROOT:-/data/wanganna/CausalGap}/experiment_results/sumo_replication"
python -u "$D/04_validate_source.py" --seeds 0 1 2 3 4
python -u "$D/04_stage_source.py" --seeds 0 1 2 3 4
python -u "$D/model_eval/preflight.py" --experiment-root "$P" --require-sumo
python - "$P" <<'PY'
import sys
from pathlib import Path
import numpy as np,pandas as pd
p=Path(sys.argv[1])/'environments'/'sumo_metrla'
m=pd.read_csv(p/'manifest.csv')
with np.load(p/'baseline.npz',allow_pickle=False) as b: x,y,adj=b['x'],b['y'],b['adj']
assert len(m)==60 and x.shape==y.shape==(2,12,24,1) and adj.shape==(24,24), (len(m),x.shape,y.shape,adj.shape)
assert all(Path(s).exists() for s in m.path)
print('[PASS] staged 60 intervention scenarios + baseline; B=2, H=F=12, N=24')
PY
