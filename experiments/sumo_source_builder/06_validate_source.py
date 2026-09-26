#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from config import *
from common import safe_name, load_model_adj_and_ids


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=SOURCE_ROOT); args=ap.parse_args()
    root=args.root
    expected=[root/'baseline.npz']
    for kind,levels in H1_LEVELS.items():
        for spec in levels:
            for seed in EVAL_INTERVENTION_SEEDS:
                expected.append(root/'interventions'/kind/safe_name(spec['level'])/f'seed_{seed:04d}.npz')
    missing=[p for p in expected if not p.exists()]
    print('expected=',len(expected),'found=',len(expected)-len(missing),'missing=',len(missing))
    if missing:
        print('\n'.join(map(str,missing[:30]))); raise SystemExit(2)
    a0,_=load_model_adj_and_ids(); b=np.load(root/'baseline.npz'); bx=b['x']; by=b['y']; badj=b['adj']
    assert bx.ndim==4 and bx.shape[1:]==(12,207,1),bx.shape
    assert by.shape==bx.shape,by.shape
    assert np.allclose(badj,a0,atol=1e-6,rtol=0)
    maxdx=0.0
    for p in expected[1:]:
        z=np.load(p); assert z['x'].shape==bx.shape and z['y'].shape==by.shape and z['adj'].shape==a0.shape
        maxdx=max(maxdx,float(np.max(np.abs(z['x']-bx))))
    print('baseline shape=',bx.shape)
    print('max intervention-vs-baseline x diff=',maxdx)
    if maxdx>1e-5: raise RuntimeError('x is not strictly pre-intervention/common across paired runs')
    print('[PASS] SUMO H1 source contract satisfied: 61/61 files')

if __name__=='__main__': main()
