import os
import sys
from pathlib import Path

import numpy as np
import torch

from common import parse, serve, silence_library_stdout

args = parse()
silence_library_stdout()
src = Path(args.root) / 'class5' / 'TimesFM' / 'v1' / 'src'
sys.path.insert(0, str(src))
import timesfm

backend = 'gpu' if torch.cuda.is_available() else 'cpu'
ckpt = Path(args.weights) / 'torch_model.ckpt'
PER_CORE_BATCH = max(1, int(os.environ.get('CAUSALGAP_TIMESFM_PER_CORE_BATCH', '32')))
CHUNK = max(1, int(os.environ.get('CAUSALGAP_TIMESFM_CHUNK', '128')))
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend=backend,
        per_core_batch_size=PER_CORE_BATCH,
        horizon_len=128,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(version='torch', path=str(ckpt)),
)


def infer(flat, H, num_samples, need_rep=True, progress_cb=None):
    preds = []
    reps = [] if need_rep else None
    total = len(flat)
    for s in range(0, total, CHUNK):
        chunk = np.ascontiguousarray(flat[s:s + CHUNK], dtype=np.float32)
        cache = []
        handle = None
        if need_rep:
            def hook(m, i, o):
                if torch.is_tensor(o):
                    cache.append(o.detach().float().cpu())
            handle = tfm._model.stacked_transformer.register_forward_hook(hook)
        try:
            mean, _ = tfm.forecast(
                [a for a in chunk],
                freq=[0] * len(chunk),
            )
        finally:
            if handle is not None:
                handle.remove()
        preds.append(np.asarray(mean, dtype=np.float32)[:, :H])
        if need_rep:
            if not cache:
                raise RuntimeError('TimesFM representation hook did not fire')
            rr = torch.cat([c.mean(dim=1) for c in cache], dim=0)[:len(chunk)].numpy()
            reps.append(rr)
        if progress_cb is not None:
            progress_cb(min(s + len(chunk), total), total, 'timesfm_chunk')
    return np.concatenate(preds, 0), (np.concatenate(reps, 0) if need_rep else None)


serve(infer)
