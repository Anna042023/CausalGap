#!/usr/bin/env python3
"""Strictly verify complete model inference before descriptive replication analysis.
Does not compute/fabricate DMIS or RD and does not run H1 on SUMO.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'model_eval'))
from protocol import ALL_MODELS, H1_LEVELS, DEFAULT_EXPERIMENT_ROOT

def check(experiment_root, models, model_seeds, intervention_seeds):
    expected={(kind,item['level'],str(seed)) for kind,items in H1_LEVELS.items()
              for item in items for seed in intervention_seeds}
    errors=[]; rows=[]
    for model in models:
        for seed in model_seeds:
            name=f'{model}/seed{seed}'
            p=experiment_root/'runs'/'sumo_metrla'/model/'trackB'/f'seed{seed}'/'causalgap_results.csv'
            if not p.exists():
                errors.append(f'{name}: missing {p}');continue
            try: d=pd.read_csv(p)
            except Exception as e: errors.append(f'{name}: bad CSV {e}');continue
            needed={'intervention_type','paper_intensity_level','intervention_seed',
                    'RD','DMIS','representation_response_raw','evaluation_track'}
            if not needed<=set(d.columns):
                errors.append(f'{name}: missing columns {sorted(needed-set(d.columns))}');continue
            keys=list(zip(d.intervention_type.astype(str),d.paper_intensity_level.astype(str),
                          d.intervention_seed.astype(str)))
            obs=set(keys)
            if len(keys)!=len(expected) or len(obs)!=len(expected) or obs!=expected:
                errors.append(f'{name}: rows={len(keys)}, unique={len(obs)}, expected={len(expected)}; missing={len(expected-obs)}, extra={len(obs-expected)}')
                continue
            num=d[['RD','DMIS','representation_response_raw']].apply(pd.to_numeric,errors='coerce').to_numpy()
            if not np.isfinite(num).all():
                errors.append(f'{name}: nonfinite RD/DMIS/representation_response_raw');continue
            if not d.evaluation_track.astype(str).str.upper().eq('B').all():
                errors.append(f'{name}: not Track B');continue
            rows.append({'model':model,'model_seed':seed,'rows':len(d),'RD_mean':float(d.RD.mean()),
                         'DMIS_mean':float(d.DMIS.mean()),'result_csv':str(p)})
    print(f'[CHECK] models={len(models)} model_seeds={len(model_seeds)} expected_jobs={len(models)*len(model_seeds)} passed={len(rows)}')
    for e in errors:print('[FAIL]',e)
    if errors:raise SystemExit(f'[STOP] {len(errors)} incomplete/invalid job(s); do not analyze yet')
    out=experiment_root/'analysis'/'replication';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'model_job_completion.csv',index=False)
    print(f'[PASS] {len(rows)} complete Track-B replication jobs; results in {out}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment-root',type=Path,default=DEFAULT_EXPERIMENT_ROOT)
    p.add_argument('--models',nargs='+',choices=ALL_MODELS,default=ALL_MODELS)
    p.add_argument('--model-seeds',nargs='+',type=int,default=[0])
    p.add_argument('--eval-seeds',nargs='+',type=int,default=[0,1,2,3,4])
    a=p.parse_args();check(a.experiment_root,a.models,a.model_seeds,a.eval_seeds)
if __name__=='__main__':main()
