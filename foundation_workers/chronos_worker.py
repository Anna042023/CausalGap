import os
import sys
from pathlib import Path

import numpy as np
import torch

from common import parse, serve, silence_library_stdout

args = parse()
silence_library_stdout()
sys.path.insert(0, str(Path(args.root) / 'class5' / 'Chronos' / 'src'))
from chronos import ChronosPipeline

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline = ChronosPipeline.from_pretrained(
    args.weights,
    device_map=DEVICE,
    torch_dtype=(torch.float16 if DEVICE == 'cuda' else torch.float32),
    local_files_only=True,
)
CHUNK = max(1, int(os.environ.get('CAUSALGAP_CHRONOS_CHUNK', '128')))


def infer(flat, H, num_samples, need_rep=True, progress_cb=None):
    preds = []
    reps = [] if need_rep else None
    total = len(flat)
    for s in range(0, total, CHUNK):
        chunk = np.ascontiguousarray(flat[s:s + CHUNK], dtype=np.float32)
        ctx = torch.from_numpy(chunk).float()
        with torch.inference_mode():
            samples = pipeline.predict(ctx, prediction_length=H, num_samples=num_samples)
        preds.append(samples.median(dim=1).values.cpu().numpy())
        if need_rep:
            with torch.inference_mode():
                emb, _ = pipeline.embed(ctx)
            reps.append(emb.float().mean(dim=1).cpu().numpy())
        if progress_cb is not None:
            progress_cb(min(s + len(chunk), total), total, 'chronos_chunk')
        del ctx, samples
    return np.concatenate(preds, 0), (np.concatenate(reps, 0) if need_rep else None)


serve(infer)
