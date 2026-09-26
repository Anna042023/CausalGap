import argparse
import json
from pathlib import Path
import numpy as np
from causalgap.data import load_dataset

KINDS = ('edge_removal','weight_modification')


def find_file(root, dataset, kind, i):
    for p in (
        root/dataset/f'{kind}_{i:03d}.npz',
        root/dataset/kind/f'{kind}_{i:03d}.npz',
        root/f'{kind}_{i:03d}.npz',
    ):
        if p.exists(): return p
    return None


def main():
    ap=argparse.ArgumentParser('Validate physically coupled CausalGap environments')
    ap.add_argument('--root', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--num-interventions', type=int, default=100)
    ap.add_argument('--types', nargs='*', default=list(KINDS), choices=KINDS)
    args=ap.parse_args()
    root=Path(args.root); b=load_dataset(args.dataset)
    errors=[]; ok=0
    for kind in args.types:
        for i in range(args.num_interventions):
            p=find_file(root,b.name,kind,i)
            if p is None:
                errors.append(f'missing {kind}_{i:03d}.npz'); continue
            try:
                z=np.load(p,allow_pickle=True)
                for k in ('x','y','adj'):
                    if k not in z.files: raise ValueError(f'missing key {k}')
                x=np.asarray(z['x']); y=np.asarray(z['y']); a=np.asarray(z['adj'])
                if x.ndim!=4 or y.ndim!=4: raise ValueError(f'x/y must be 4D, got {x.shape}/{y.shape}')
                if x.shape[2]!=b.num_nodes or y.shape[2]!=b.num_nodes: raise ValueError('node count mismatch')
                if a.shape!=b.adj.shape: raise ValueError(f'adj shape {a.shape} != {b.adj.shape}')
                if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(a).all():
                    raise ValueError('non-finite value')
                side=p.with_suffix('.json')
                if not side.exists(): raise ValueError('missing sidecar JSON with alignment/provenance')
                meta=json.loads(side.read_text(encoding='utf-8'))
                for req in ('baseline_start','baseline_stop','causal_valid','generator'):
                    if req not in meta: raise ValueError(f'sidecar missing {req}')
                if not bool(meta['causal_valid']): raise ValueError('causal_valid=false')
                if int(meta['baseline_stop'])-int(meta['baseline_start']) != len(x):
                    raise ValueError('baseline alignment length mismatch')
                ok+=1
            except Exception as e:
                errors.append(f'{p}: {e}')
    print(f'validated={ok}; errors={len(errors)}')
    for e in errors[:50]: print('[FAIL]',e)
    if errors: raise SystemExit(2)
    print('[OK] Physical environment set is ready for architecture-complete Track-A evaluation.')

if __name__=='__main__': main()
