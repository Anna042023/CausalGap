#!/usr/bin/env python3
"""Paired abstract-SUMO counterfactual y; no H1 or steering in this script."""
from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
from config import ROOT, LEVELS, H, F
from importlib import import_module
coarse=import_module('02_coarse_calibrate')


def adj_for(a,kind,val,seed):
    b=np.array(a,copy=True)
    rng=np.random.default_rng(20000+int(seed))
    n=len(b)
    if kind=='node_isolation':
        order=rng.permutation(n);ids=sorted(order[:max(1,round(float(val)*n))].tolist())
        b[ids,:]=0;b[:,ids]=0
        links=[f'link_{i}_{j}' for i in range(n) for j in range(n) if i!=j and a[i,j]>0 and (i in ids or j in ids)]
        links +=[f'sensor_{i}' for i in ids]
        return b,list(dict.fromkeys(links)),ids
    pairs=[(i,j) for i in range(n) for j in range(i+1,n) if a[i,j]>0]
    rng.shuffle(pairs)
    count=max(1,round((float(val) if kind=='edge_removal' else (1.-float(val))*.25)*len(pairs)))
    chosen=pairs[:count]
    for i,j in chosen:
        if kind=='edge_removal':b[i,j]=b[j,i]=0.
        else:b[i,j]*=val;b[j,i]*=val
    links=[f'link_{x}_{y}' for i,j in chosen for x,y in ((i,j),(j,i))]
    return b,links,[]


def stable_digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save(path,x,y,a,meta):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,x=x.astype(np.float32),y=y.astype(np.float32),adj=a.astype(np.float32))
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--simulation-seeds',nargs='+',type=int,default=[100,101])
    p.add_argument('--intervention-seeds',nargs='+',type=int,default=list(range(5)))
    p.add_argument('--smoke',action='store_true');p.add_argument('--skip-existing',action='store_true')
    args=p.parse_args()
    choice_path=args.root/'coarse_choice.json'
    if not choice_path.exists():raise FileNotFoundError(f'{choice_path}: run 02_coarse_calibrate.py first')
    choice=json.loads(choice_path.read_text());a=np.load(args.root/'subgraph.npz',allow_pickle=False)['adj']
    source=args.root/'source';source.mkdir(parents=True,exist_ok=True)
    sim_seeds=args.simulation_seeds[:1] if args.smoke else args.simulation_seeds
    intervention_seeds=args.intervention_seeds[:1] if args.smoke else args.intervention_seeds
    routes={s:coarse.make_routes(args.root,s,choice['factor'],choice['peak_amp']) for s in sim_seeds}
    bx=[];by=[];diags=[]
    for s in sim_seeds:
        x,y,d=coarse.simulate(args.root,routes[s],s)
        bx.append(x);by.append(y);diags.append(d)
    bx=np.stack(bx);by=np.stack(by)
    base_meta={'protocol':'ABSTRACT_SUMO_REPLICATION_NOT_H1','simulator_seeds':sim_seeds,
      'paired_demand':'fixed synthetic trip file per simulator seed, NOT fitted real OD',
      'coarse_choice':choice,'diag':diags,'baseline_topology_digest':stable_digest(a)}
    save(source/'baseline.npz',bx,by,a,base_meta)
    print('[BASELINE]',bx.shape,by.shape,flush=True)
    for kind,specs in ([("edge_removal",LEVELS["edge_removal"])] if args.smoke else LEVELS.items()):
        for level,val in (specs[:1] if args.smoke else specs):
            for seed in intervention_seeds:
                out=source/'interventions'/kind/level/f'seed_{seed:04d}.npz'
                if args.skip_existing and out.exists():continue
                ak,links,isolated=adj_for(a,kind,val,seed)
                ys=[];dx=[];diagnostics=[]
                for si,s in enumerate(sim_seeds):
                    x,y,d=coarse.simulate(args.root,routes[s],s,kind,links,
                                         gamma=float(val) if kind=='weight_modification' else None)
                    dx.append(float(np.max(abs(x-bx[si]))))
                    if dx[-1]>1e-5:raise RuntimeError(f'Paired pre-intervention x differs: {kind}/{level}/{seed}: {dx[-1]}')
                    ys.append(y);diagnostics.append(d)
                yk=np.stack(ys)
                # A no-effect intervention is explicitly exposed and must never be called successful causality.
                effect=float(np.max(abs(yk-by)))
                meta={'generator':'ABSTRACT_METR_TOPOLOGY_SUMO','evidence_role':'regime_replication_only',
                  'not_for_h1':True,'not_for_steering':True,'causal_valid':True,'leakage_detected':False,'phase':'strong' if level==specs[-1][0] else 'weak',
                  'paper_intensity_level':level,'intervention_type':kind,'intervention_seed':seed,
                  'simulator_seeds':sim_seeds,'paired_input_maxdiff':max(dx),
                  'future_label_maxdiff':effect,'nontrivial_traffic_effect':effect>1e-5,
                  'closure_routes_fully_replanned':all(d['failures']==0 and d['unchanged_blocked_routes']==0 for d in diagnostics),
                  'isolated_nodes':isolated,'affected_abstract_edges':links,
                  'graph_digest':stable_digest(ak),'diagnostics':diagnostics}
                save(out,bx,yk,ak,meta)
                print('[DONE]',kind,level,seed,'affected_edges=',len(links),'max_delta_y=',effect,flush=True)
    print('[COMPLETE] source at',source,'NOTE: SUMO is excluded from held-out-setting H1 and steering.')
if __name__=='__main__':main()
