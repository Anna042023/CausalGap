"""H2 independent temporal calibration rollout and disjoint topology seeds.

The existing H1 bundle has train/val/test 7:1:2, but NO explicit Dcal:
val_x is Dval (forecasting early stopping), not Dcal. Do not silently relabel it.
We draw an independent synthetic trajectory with the SAME frozen graph/dynamics
but a declared NEW RNG seed, and use only calibration topology seeds 1000+.
Strong test data are read only after the model is selected.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from synthetic_env import simulate, windows, topology_only, selected_indices
from protocol import H1_LEVELS

CAL_OFFSET = 1_000_003


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def calibration_rollout(bundle, exp, output, max_windows):
    """Returns burn,x_init,u,eps,raw_base_x,raw_base_y,calibration_indices,seed.
    Does NOT read Dval or Dtest arrays; base trajectory is independently drawn.
    """
    metadata_path=exp/'environments'/'synthetic'/'synthetic_metadata.json'
    meta=json.loads(metadata_path.read_text())
    if meta.get('split')!='windows_then_chronological_7_1_2':
        raise RuntimeError('Unexpected synthetic temporal split; inspect metadata')
    seed=int(meta['dynamics_seed'])+CAL_OFFSET
    burn=int(meta['burn_in']); snapshots=int(meta['snapshots_after_burn_in'])
    n=int(bundle.num_nodes)
    if n!=int(meta['num_nodes']): raise RuntimeError('Graph node-count mismatch')
    rg=np.random.default_rng(seed)
    x_init=rg.normal(0., .2, size=n).astype(np.float32)
    u=rg.normal(0.,float(meta['u_std']),size=(burn+snapshots,n)).astype(np.float32)
    eps=rg.normal(0.,float(meta['eps_std']),size=(burn+snapshots,n)).astype(np.float32)
    original=simulate(np.asarray(bundle.adj,np.float32),x_init,u,eps)
    bx,by=windows(original[burn:],bundle.history_len,bundle.pred_len)
    idx=selected_indices(len(bx),max_windows)
    if len(idx)<2:raise RuntimeError('Calibration selection needs >=2 windows')
    record=dict(calibration_rollout_seed=seed,base_dynamics_seed=int(meta['dynamics_seed']),
                independent_temporal_calibration=True,calibration_indices=idx.tolist(),
                selected_windows=len(idx),generator_metadata_sha256=sha256(metadata_path),
                original_h1_training_or_early_stopping_arrays_used=False,
                original_h1_test_arrays_used=False)
    output.mkdir(parents=True,exist_ok=True)
    audit=output/'calibration_rollout_manifest.json'
    if audit.exists() and json.loads(audit.read_text())!=record:
        raise RuntimeError('Calibration manifest differs from existing output; use a NEW --output-root')
    audit.write_text(json.dumps(record,indent=2))
    return burn,x_init,u,eps,bx[idx],by[idx],idx,seed


def calibration_topologies(bundle, exp, output, chosen_seeds):
    refpath=exp/'environments'/'synthetic'/'calibration_distance_reference.csv'
    df=pd.read_csv(refpath)
    required={'intervention_type','level','seed','environment_distance'}
    if not required.issubset(df):raise RuntimeError(f'Invalid {refpath}: missing {required-set(df)}')
    chosen=set(map(int,chosen_seeds))
    selected=df[df.seed.astype(int).isin(chosen)].copy()
    if not chosen.issubset(set(df.seed.astype(int))):
        raise RuntimeError('Missing calibration seed(s) in distance reference')
    expected={(kind,spec['level'],seed)
              for kind,levels in H1_LEVELS.items()
              for spec in levels if spec['phase']=='weak'
              for seed in chosen}
    got={(str(r.intervention_type),str(r.level),int(r.seed)) for r in selected.itertuples()}
    if expected!=got or len(selected)!=len(expected):
        raise RuntimeError(f'Calibration topology coverage mismatch: missing={expected-got}, extras={got-expected}')
    result=[]
    root=output/'_calibration_topologies'
    root.mkdir(parents=True,exist_ok=True)
    for row in selected.itertuples(index=False):
        kind,level,seed=str(row.intervention_type),str(row.level),int(row.seed)
        spec=next(s for s in H1_LEVELS[kind] if s['level']==level and s['phase']=='weak')
        it=topology_only(bundle.adj,kind,spec,seed)
        dist=float(it.environment_distance)
        if not np.isclose(dist,float(row.environment_distance),rtol=2e-5,atol=1e-6):
            raise RuntimeError(f'Calibration topology drift for {kind}/{level}/{seed}: {dist} vs {row.environment_distance}')
        dest=root/kind/level.replace('/','_')/f'seed_{seed:04d}.npz'
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            with np.load(dest,allow_pickle=False) as old:
                if not np.allclose(old['adj'],it.adj,rtol=0,atol=0):
                    raise RuntimeError('Cached calibration topology mismatch '+str(dest))
        else:
            np.savez_compressed(dest,adj=np.asarray(it.adj,np.float32))
            dest.with_suffix('.json').write_text(json.dumps(it.meta,indent=2,default=str))
        result.append(dict(setting='synthetic',intervention_type=kind,level=level,
                           phase='weak',seed=seed,environment_distance=dist,
                           path=str(dest),strength=float(spec['strength'])))
    return pd.DataFrame(result)
