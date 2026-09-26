#!/usr/bin/env python3
"""Read the existing frozen H2 paired CSV and archived calibration logs.
No retraining; no selection; no rewriting original results.
"""
import argparse, re, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=['Matched_without_DMIS','Matched+DMIS','Shuffled_DMIS','Gradient','Random']

def parse_logs(zip_path):
    rows=[]
    if not zip_path:return pd.DataFrame()
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            match=re.search(r'H2_DLinear_seed(\d+)\.log$',name)
            if not match:continue
            seed=int(match.group(1))
            pattern=re.compile(r'^\[H2 TRAIN\] (\S+) epoch=(\d+) cal-val IPE=([0-9.eE+-]+)')
            for line in z.read(name).decode(errors='replace').splitlines():
                m=pattern.search(line)
                if m:rows.append(dict(model_seed=seed,method=m.group(1),epoch=int(m.group(2)),cal_val_IPE=float(m.group(3))))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--paired',type=Path,required=True)
    ap.add_argument('--logs-zip',type=Path,default=None)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(a.paired)
    must={'model_seed','intervention_type','Matched+DMIS','Matched_without_DMIS','Shuffled_DMIS'}
    if not must.issubset(p.columns):raise ValueError('Missing columns: '+str(must-set(p.columns)))
    p['dmis_minus_matched']=p['Matched+DMIS']-p['Matched_without_DMIS']
    p['dmis_minus_shuffled']=p['Matched+DMIS']-p['Shuffled_DMIS']
    by_seed=p.groupby('model_seed').agg(
        paired_units=('dmis_minus_matched','size'),
        mean_effect_pp=('dmis_minus_matched','mean'),
        fraction_positive=('dmis_minus_matched',lambda s:float((s>0).mean())),
        vs_shuffled_pp=('dmis_minus_shuffled','mean')).reset_index()
    by_seed.to_csv(a.output/'observed_seed_effects_DESCRIPTIVE.csv',index=False)
    p.groupby(['model_seed','intervention_type'],as_index=False)[['dmis_minus_matched','dmis_minus_shuffled']].mean().to_csv(
        a.output/'observed_family_effects_DESCRIPTIVE.csv',index=False)
    curves=parse_logs(a.logs_zip)
    if len(curves):
        curves.to_csv(a.output/'calibration_validation_curves.csv',index=False)
        ix=curves.groupby(['model_seed','method']).cal_val_IPE.idxmin()
        best=curves.loc[ix].sort_values(['model_seed','method']).reset_index(drop=True)
        best.to_csv(a.output/'calibration_best_epochs.csv',index=False)
        print('CALIBRATION BEST EPOCHS:')
        print(best.to_string(index=False))
    print('OBSERVED STRONG OUTCOMES (diagnostic only, never use for variant selection):')
    print(by_seed.to_string(index=False))
    print('OUTPUT:',a.output)

if __name__=='__main__':main()
