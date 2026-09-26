#!/usr/bin/env python3
"""Sequential, resumable launcher for the *frozen* H1 V2 representation replay.

Jobs are taken from the frozen V2 weak-row snapshot, not by scanning the mutable
H1 run directory.  This prevents later H2-only DLinear seeds from entering 5.6.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
PROJECT=Path('/data/wanganna/CausalGap')
RAW_REL=Path('experiment_results/incremental_value/analysis/h1/intermediate/01_h1_raw_calibrated_rows.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=PROJECT)
    p.add_argument('--stage',choices=['preflight','export'],default='preflight')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--foundation-cache-policy',choices=['auto','cached','fresh'],default='auto')
    p.add_argument('--models',nargs='*',default=None)
    p.add_argument('--seeds',nargs='*',type=int,default=None,help='Intervention seeds only')
    p.add_argument('--rep-rel-l2',type=float,default=2e-5)
    a=p.parse_args()
    raw=a.project_root/RAW_REL
    if not raw.is_file(): raise FileNotFoundError(raw)
    d=pd.read_csv(raw)
    need={'setting','h1_phase','model','model_seed'}
    if need-set(d.columns): raise RuntimeError(f'Frozen raw H1 missing {sorted(need-set(d.columns))}')
    d=d[(d.setting.astype(str)=='synthetic') & (d.h1_phase.astype(str)=='weak')].copy()
    d['model_seed']=pd.to_numeric(d['model_seed'],errors='raise').astype(int)
    jobs=sorted(set(zip(d.model.astype(str),d.model_seed.astype(int))))
    if a.models is not None:
        jobs=[x for x in jobs if x[0] in set(a.models)]
    if not jobs: raise RuntimeError('No requested frozen H1 jobs found')
    print(f'[JOBS] {len(jobs)} frozen model/seed jobs from V2 weak snapshot; stage={a.stage}',flush=True)
    for i,(model,seed) in enumerate(jobs,1):
        policy=a.foundation_cache_policy
        if policy=='auto' and model=='Moirai': policy='fresh'
        cmd=[sys.executable,'-u',str(HERE/'export_h1_pairs.py'),
             '--project-root',str(a.project_root),'--model',model,'--model-seed',str(seed),
             '--stage',a.stage,'--device',a.device,
             '--foundation-cache-policy',policy,
             '--rep-rel-l2',str(a.rep_rel_l2)]
        if a.models is None and a.seeds is None: cmd.append('--preflight-full')
        if a.seeds is not None: cmd+=['--seeds',*map(str,a.seeds)]
        print(f'[{i}/{len(jobs)}] {model} seed{seed} cache_policy={policy}',flush=True)
        rc=subprocess.run(cmd,check=False).returncode
        if rc!=0: raise RuntimeError(f'Stopped at {model} seed{seed} (rc={rc}); fix before proceeding')
    print('[DONE] requested frozen H1 pair replay jobs',flush=True)

if __name__=='__main__': main()
