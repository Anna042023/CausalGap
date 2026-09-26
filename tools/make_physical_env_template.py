"""Create ONE format example only; this does NOT simulate traffic physics.

Use SUMO, the paper's synthetic graph dynamics, or the I-405 preprocessing pipeline
for real Xk/Yk/Ak.  This utility documents the exact file contract expected by run.py.
"""
import argparse, json
from pathlib import Path
import numpy as np
from causalgap.data import load_dataset


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--out-root',required=True)
    ap.add_argument('--kind',choices=['edge_removal','weight_modification'],default='edge_removal')
    args=ap.parse_args(); b=load_dataset(args.dataset)
    out=Path(args.out_root)/b.name; out.mkdir(parents=True,exist_ok=True)
    p=out/f'{args.kind}_000.npz'
    # Explicitly marked TEMPLATE/non-physical; run.py should not be used to claim results from it.
    np.savez_compressed(p,
        x=b.scaler.inverse(b.test_x).astype(np.float32),
        y=b.scaler.inverse(b.test_y).astype(np.float32),
        adj=b.adj.astype(np.float32))
    meta={
        'baseline_start':0,'baseline_stop':len(b.test_x),'causal_valid':True,
        'leakage_detected':False,'generator':'FORMAT_TEMPLATE_NOT_PHYSICAL',
        'severity':0.0,'notes':'Replace x/y/adj with SUMO/synthetic/I-405 outputs before paper use.'
    }
    p.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(p)
    print('[WARNING] This is a format template, NOT a physically valid intervention.')

if __name__=='__main__': main()
