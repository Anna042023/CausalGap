#!/usr/bin/env python3
"""Pretest-style calibration audit across fixed variants, never consults strong-test CSVs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

VARIANTS=['mlp_teacher0','film_teacher0','film_teacher0.01']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/wanganna/CausalGap/experiment_results/representation_steering'))
    a=p.parse_args()
    rows=[]
    for v in VARIANTS:
        src=a.root/v/'calibration_comparison.csv'
        if not src.exists():raise FileNotFoundError('Finish calibration-only first: '+str(src))
        d=pd.read_csv(src)
        if sorted(d.model_seed.astype(int).tolist())!=list(range(5)):
            raise RuntimeError(f'{v}: expected five distinct checkpoint seeds')
        diff=d.matched_val_ipe-d.dmis_val_ipe
        rows.append(dict(variant=v,n_seeds=len(d),
            mean_matched_val_IPE=float(d.matched_val_ipe.mean()),
            mean_dmis_val_IPE=float(d.dmis_val_ipe.mean()),
            mean_dmis_extra_cal_IPE=float(diff.mean()),
            positive_cal_seeds=int((diff>0).sum()),
            mean_dmis_minus_shuffled_cal_IPE=float((d.shuffled_val_ipe-d.dmis_val_ipe).mean())))
    tab=pd.DataFrame(rows)
    tab.to_csv(a.root/'calibration_variant_summary.csv',index=False)
    print('CALIBRATION-ONLY VARIANT COMPARISON; old strong-test outcome NOT accessed:')
    print(tab.to_string(index=False))
    plausible=tab[(tab.mean_dmis_extra_cal_IPE>0)&(tab.positive_cal_seeds>=3)]
    if plausible.empty:
        print('NO VARIANT clears calibration-only stability gate. Keep original negative H2 result.')
    else:
        v=plausible.sort_values('mean_dmis_val_IPE',ascending=True).iloc[0]['variant']
        print('CAL-ONLY candidate for an explicitly exploratory follow-up:',v)
        print('This does NOT imply superiority on strong interventions or over label-oracle Gradient.')

if __name__=='__main__':main()
