#!/usr/bin/env python3
"""Stage validated N-node source for the existing Track-B evaluator (not H1 inference)."""
import argparse, shutil, json
from pathlib import Path
import numpy as np
import pandas as pd
from config import ROOT,EXPERIMENT,LEVELS
from importlib import import_module
validate=import_module('04_validate_source').validate

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--experiment-root',type=Path,default=EXPERIMENT)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(5)));args=p.parse_args()
    validate(args.root,args.seeds)
    dst=args.experiment_root/'environments'/'sumo_metrla';dst.mkdir(parents=True,exist_ok=True)
    shutil.copy2(args.root/'subgraph.npz',dst/'subgraph.npz')
    shutil.copy2(args.root/'source'/'baseline.npz',dst/'baseline.npz')
    shutil.copy2(args.root/'source'/'baseline.json',dst/'baseline.json')
    rows=[]
    for kind,lvls in LEVELS.items():
        for j,(level,v) in enumerate(lvls):
            for seed in args.seeds:
                f=args.root/'source'/'interventions'/kind/level/f'seed_{seed:04d}.npz'
                t=dst/'interventions'/kind/level/f.name;t.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(f,t)
                meta=json.loads(f.with_suffix('.json').read_text())
                meta.update({'baseline_start':0,'baseline_stop':int(np.load(dst/'baseline.npz')['x'].shape[0]),
                             'seed':seed,'intervention_sampling_seed':seed,'severity':float(v)})
                t.with_suffix('.json').write_text(json.dumps(meta,indent=2))
                rows.append({'setting':'sumo_metrla','intervention_type':kind,'level':level,
                    'phase':'strong' if j==len(lvls)-1 else 'weak',
                    'strength':(1-float(v)) if kind=='weight_modification' else float(v),
                    'seed':seed,'path':str(t)})
    pd.DataFrame(rows).to_csv(dst/'manifest.csv',index=False)
    (dst/'EVIDENCE_ROLE.txt').write_text('ABSTRACT_SUMO_REGIME_REPLICATION_ONLY\nNOT_H1\nNOT_STEERING\n')
    print('[PASS] staged',len(rows),'replication intervention files ->',dst)
if __name__=='__main__': main()
