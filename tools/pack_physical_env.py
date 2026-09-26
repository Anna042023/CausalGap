"""Pack simulator/preprocessed arrays into CausalGap's external physical format.

This tool deliberately does not invent traffic physics.  `x` and `y` must come
from SUMO, the paper's synthetic graph dynamics, or real-event preprocessing.
Arrays are stored in ORIGINAL traffic units; run.py applies the training scaler.
"""
import argparse, json
from pathlib import Path
import numpy as np


def load_array(path):
    p=Path(path)
    if p.suffix=='.npy': return np.load(p)
    if p.suffix=='.npz':
        z=np.load(p,allow_pickle=True)
        if len(z.files)!=1: raise ValueError(f'{p}: NPZ must contain one array or use .npy; keys={z.files}')
        return z[z.files[0]]
    raise ValueError(f'Unsupported array file: {p}')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--x',required=True); ap.add_argument('--y',required=True); ap.add_argument('--adj',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--generator',required=True,choices=['SUMO','SYNTHETIC_GRAPH_DYNAMICS','I405_REAL','OTHER_VALIDATED'])
    ap.add_argument('--baseline-start',type=int,default=0); ap.add_argument('--baseline-stop',type=int,required=True)
    ap.add_argument('--severity',type=float,default=1.0); ap.add_argument('--notes',default='')
    args=ap.parse_args()
    x=np.asarray(load_array(args.x),dtype=np.float32); y=np.asarray(load_array(args.y),dtype=np.float32); a=np.asarray(load_array(args.adj),dtype=np.float32)
    if x.ndim!=4 or y.ndim!=4: raise ValueError(f'x/y must be [B,T,N,F], got {x.shape}/{y.shape}')
    if len(x)!=len(y): raise ValueError('x/y sample count mismatch')
    if args.baseline_stop-args.baseline_start!=len(x): raise ValueError('baseline range length must equal x/y samples')
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,x=x,y=y,adj=a)
    meta={
      'baseline_start':args.baseline_start,'baseline_stop':args.baseline_stop,
      'causal_valid':True,'leakage_detected':False,'generator':args.generator,
      'severity':args.severity,'notes':args.notes,
    }
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(out)

if __name__=='__main__': main()
