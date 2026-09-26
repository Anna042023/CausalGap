import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import parse, serve, silence_library_stdout

args = parse()
silence_library_stdout()
src = Path(args.root) / 'class5' / 'Lag-Llama'
sys.path.insert(0, str(src))
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.common import ListDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = Path(args.weights) / 'lag-llama.ckpt'
raw = torch.load(ckpt, map_location='cpu', weights_only=False)
hp = raw.get('hyper_parameters', {})
model_kwargs = hp.get('model_kwargs', {})
PRED_BATCH = max(1, int(os.environ.get('CAUSALGAP_LAGLLAMA_PRED_BATCH', '32')))
# Critical v3.2.1 fix: do not hand thousands of node-series to one GluonTS
# generator call. Process a bounded number of univariate series at a time and
# emit progress after every chunk. 128 is conservative on a 24-GB RTX 3090.
REQUEST_CHUNK = max(PRED_BATCH, int(os.environ.get('CAUSALGAP_LAGLLAMA_REQUEST_CHUNK', '64')))
PREDICTOR_CACHE = {}


def get_predictor(H, num_samples, context):
    key = (int(H), int(num_samples), int(min(max(1, context), 32)), int(PRED_BATCH))
    if key in PREDICTOR_CACHE:
        return PREDICTOR_CACHE[key]
    kw = dict(model_kwargs)
    est = LagLlamaEstimator(
        prediction_length=H,
        context_length=key[2],
        input_size=int(kw.get('input_size', 1)),
        batch_size=PRED_BATCH,
        n_layer=int(kw.get('n_layer', 8)),
        n_embd_per_head=int(kw.get('n_embd_per_head', 16)),
        n_head=int(kw.get('n_head', 9)),
        max_context_length=int(kw.get('max_context_length', 2048)),
        rope_scaling=kw.get('rope_scaling', None),
        scaling=kw.get('scaling', 'robust'),
        num_parallel_samples=num_samples,
        time_feat=bool(kw.get('time_feat', False)),
        dropout=0.0,
        ckpt_path=str(ckpt),
        trainer_kwargs={
            'accelerator': 'gpu' if DEVICE == 'cuda' else 'cpu',
            'devices': 1,
        },
    )
    trans = est.create_transformation()
    lightning = est.create_lightning_module()
    predictor = est.create_predictor(trans, lightning)
    PREDICTOR_CACHE[key] = (predictor, lightning.model)
    return PREDICTOR_CACHE[key]


def infer(flat, H, num_samples, need_rep=True, progress_cb=None):
    predictor, module = get_predictor(H, num_samples, flat.shape[1])
    pred_parts, rep_parts = [], ([] if need_rep else None)
    total = len(flat)

    for s in range(0, total, REQUEST_CHUNK):
        chunk = np.ascontiguousarray(flat[s:s + REQUEST_CHUNK], dtype=np.float32)
        cache = []
        handle = None
        if need_rep:
            def hk(m, i, o):
                if torch.is_tensor(o):
                    cache.append(o.detach().float().cpu())
            handle = module.transformer.ln_f.register_forward_hook(hk)

        ds = ListDataset(
            [{'start': pd.Period('2000-01-01', freq='s'), 'target': a} for a in chunk],
            freq='s',
        )
        try:
            fc = list(predictor.predict(ds))
        finally:
            if handle is not None:
                handle.remove()

        pred_parts.append(np.stack([np.median(f.samples, axis=0)[:H] for f in fc]).astype(np.float32))
        if need_rep:
            if not cache:
                raise RuntimeError('Lag-Llama representation hook did not fire')
            rep_parts.append(torch.cat([c.mean(dim=1) for c in cache], dim=0)[:len(chunk)].numpy())

        if progress_cb is not None:
            progress_cb(min(s + len(chunk), total), total, 'lagllama_chunk')
        del fc, ds
        if DEVICE == 'cuda' and ((s // REQUEST_CHUNK) + 1) % 8 == 0:
            torch.cuda.empty_cache()

    return np.concatenate(pred_parts, 0), (np.concatenate(rep_parts, 0) if need_rep else None)


serve(infer)
