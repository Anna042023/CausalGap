import argparse, json
from pathlib import Path
import numpy as np
from config import ROOT,LEVELS,H,F

def validate(root,seeds,complete=True):
    src=root/'source';bp=src/'baseline.npz'
    if not bp.exists():raise FileNotFoundError(bp)
    with np.load(bp,allow_pickle=False) as b:
        bx,by,a=(b[k] for k in ('x','y','adj'))
    n=len(a);expected=(len(bx),H,n,1)
    assert bx.shape==by.shape==expected and a.shape==(n,n) and n<207,(bx.shape,by.shape,a.shape)
    assert np.isfinite(bx).all() and np.isfinite(by).all()
    count=1;invalid=[];effect=[]
    for kind,lvls in ([('edge_removal', LEVELS['edge_removal'])] if not complete else LEVELS.items()):
        for level,v in (lvls if complete else lvls[:1]):
            for seed in seeds:
                p=src/'interventions'/kind/level/f'seed_{seed:04d}.npz'
                if not p.exists():raise FileNotFoundError(p)
                with np.load(p,allow_pickle=False) as z: x,y,adj=(z[k] for k in ('x','y','adj'))
                if x.shape!=expected or y.shape!=expected or adj.shape!=(n,n):invalid.append(f'{p}: shape')
                if not np.allclose(x,bx,atol=1e-5,rtol=0):invalid.append(f'{p}: input not paired')
                if not np.isfinite(y).all():invalid.append(f'{p}: nonfinite targets')
                if np.allclose(adj,a):invalid.append(f'{p}: zero topology change')
                diff=float(np.max(abs(y-by)));effect.append((str(p),diff))
                # No-effect topology changes are legitimate counterfactuals; report, never fabricate y.
                count+=1
    expected_count=1+len(seeds)*(12 if complete else 1)
    print('N=',n,'B=',len(bx),'files=',count,'expected=',expected_count)
    print('nontrivial_future_effects=',sum(d>1e-5 for _,d in effect),'/',len(effect))
    if not any(d>1e-5 for _,d in effect):invalid.append('ALL physical intervention futures identical to baseline; inspect TraCI actuation and traffic')
    if invalid:
        for i in invalid[:30]:print('[FAIL]',i)
        raise RuntimeError(f'{len(invalid)} validation failures; do not claim regime replication')
    print('[PASS] paired topology+outcomes source; not a real-traffic calibration assertion.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(5)));p.add_argument('--smoke',action='store_true')
    a=p.parse_args();validate(a.root,a.seeds[:1] if a.smoke else a.seeds,not a.smoke)
if __name__=='__main__':main()
