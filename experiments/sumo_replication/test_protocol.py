#!/usr/bin/env python3
"""Fast mock/source contract tests; no SUMO runs, no fabricated experimental results."""
import importlib,sys,tempfile,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
build=importlib.import_module('01_build_graph')
coarse=importlib.import_module('02_coarse_calibrate')
gen=importlib.import_module('03_generate_source')
check=importlib.import_module('04_validate_source')
from config import LEVELS

def main():
    rng=np.random.default_rng(8)
    a=np.zeros((40,40),float)
    for i in range(39):a[i,i+1]=a[i+1,i]=1
    for i in range(36):a[i,i+3]=a[i+3,i]=.7
    ix,sub=build.select_subgraph(a,24,4)
    assert sub.shape==(24,24) and len(set(ix.tolist()))==24
    seen={0};active=[0]
    while active:
        for j in np.flatnonzero(sub[active.pop(0)]>0):
            if j not in seen:seen.add(int(j));active.append(int(j))
    assert len(seen)==24
    for k, specs in LEVELS.items():
        for level,value in specs:
            ak,edges,isolated=gen.adj_for(sub,k,value,0)
            assert ak.shape==sub.shape and edges and not np.allclose(ak,sub), (k,level)
            if k=='node_isolation':assert isolated and all(np.allclose(ak[i],0) for i in isolated)
    x=rng.uniform(25,70,(100,24))
    stats=coarse.stats(x,70)
    assert set(stats)=={'mean_speed_mph','speed_std_mph','congestion_fraction','peak_valley_speed_ratio'}
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);src=root/'source';src.mkdir()
        bx=rng.uniform(20,65,(2,12,24,1)).astype('float32')
        y=rng.uniform(20,65,(2,12,24,1)).astype('float32')
        np.savez_compressed(src/'baseline.npz',x=bx,y=y,adj=sub.astype('float32'))
        for k,specs in LEVELS.items():
            for level,v in specs:
                for seed in range(2):
                    adj,edges,isolated=gen.adj_for(sub,k,v,seed)
                    out=src/'interventions'/k/level/f'seed_{seed:04d}.npz';out.parent.mkdir(parents=True,exist_ok=True)
                    yy=y.copy();yy[:]+=0.01
                    np.savez_compressed(out,x=bx,y=yy,adj=adj.astype('float32'))
        check.validate(root,[0,1])
        # Fail closed on non-paired pre-t0 X.
        p=src/'interventions'/'edge_removal'/'s=0.05'/'seed_0000.npz'
        with np.load(p) as z:x1,yy,a1=z['x'].copy(),z['y'].copy(),z['adj'].copy()
        x1[0,0,0,0]+=1
        np.savez_compressed(p,x=x1,y=yy,adj=a1)
        try:check.validate(root,[0,1])
        except RuntimeError:pass
        else:raise AssertionError('validator missed input mismatch')
    print('TESTS PASSED: connected subgraph, 12 intervention specs, coarse stats, 25-file mock source, negative paired-X test')
if __name__=='__main__': main()
