#!/usr/bin/env python3
"""Synthetic-only model-family-out H1; replaces prior 'setting robustness' with
weak-history and intensity specification analyses. Never uses SUMO outcomes.
"""
import argparse,sys,json
from pathlib import Path
import numpy as np
import pandas as pd
from config import PROJECT
sys.path.insert(0,str(Path(__file__).parent/'model_eval'))
import analysis_core as h

SPECS=[
 ('primary_mean_slope_max_distance','weak_RD_mean|weak_RD_slope|weak_RD_max','distance'),
 ('history_mean_max_no_slope','weak_RD_mean|weak_RD_max','distance'),
 ('history_mean_only','weak_RD_mean','distance'),
 ('nominal_intensity_slope','weak_RD_mean|weak_RD_slope_nominal|weak_RD_max','nominal'),
]

def nominal_slope(units, configurations):
    cols={'setting','model','intervention_type','h1_phase','declared_strength','RD'}
    if not cols.issubset(configurations.columns):
        raise ValueError(f'Configuration seed-means lacks {sorted(cols-set(configurations.columns))}')
    wk=configurations[configurations.h1_phase.astype(str).eq('weak')].copy()
    s=[]
    for (setting,model,kind),g in wk.groupby(['setting','model','intervention_type']):
        if len(g)!=3:continue
        x=g.declared_strength.to_numpy(float)
        x=(x-x.min())/max(x.max()-x.min(),1e-12)
        y=g.RD.to_numpy(float)
        b=float(((x-x.mean())*(y-y.mean())).sum()/max(((x-x.mean())**2).sum(),1e-12))
        s.append(dict(setting=setting,model=model,intervention_type=kind,weak_RD_slope_nominal=b))
    result=units.merge(pd.DataFrame(s),on=['setting','model','intervention_type'],how='left',validate='many_to_one')
    if result.weak_RD_slope_nominal.isna().any():
        raise RuntimeError('Incomplete 3-level weak history: no imputation allowed')
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--units',type=Path,default=PROJECT/'experiment_results'/'incremental_value'/'analysis'/'h1'/'intermediate'/'04_h1_prediction_units.csv')
    p.add_argument('--configurations',type=Path,default=PROJECT/'experiment_results'/'incremental_value'/'analysis'/'h1'/'intermediate'/'02_configuration_seed_means.csv')
    p.add_argument('--out',type=Path,default=PROJECT/'experiment_results'/'sumo_h1_protocol'/'analysis'/'h1')
    args=p.parse_args()
    if not args.units.exists():raise FileNotFoundError(args.units)
    df=pd.read_csv(args.units)
    if df.setting.nunique()!=1 or str(df.setting.iloc[0])!='synthetic':
        raise ValueError('H1 MUST be synthetic-only under revised protocol; do not pool SUMO rows')
    if not args.configurations.exists():raise FileNotFoundError(args.configurations)
    df=nominal_slope(df,pd.read_csv(args.configurations))
    args.out.mkdir(parents=True,exist_ok=True)
    records=[];folds=[]
    for name,feature_string,intensity in SPECS:
        history=feature_string.split('|')
        h.FEATURE_SETS['spec_error']=history
        h.FEATURE_SETS['spec_plus']=history+['weak_DMIScal_mean']
        ra,fa,pa=h.run_outer(df,'model_family','spec_error',h.LAMBDA_GRID)
        rb,fb,pb=h.run_outer(df,'model_family','spec_plus',h.LAMBDA_GRID)
        a=fa.set_index('held_out').macro_fold_mae
        b=fb.set_index('held_out').macro_fold_mae
        shared=a.index.intersection(b.index)
        if len(shared)<3:raise RuntimeError('Insufficient held-out model families')
        changes=(a.loc[shared]-b.loc[shared]).to_numpy(float)
        # Descriptive fold resampling CI; NOT independent model-level statistical inference.
        rng=np.random.default_rng(2026)
        boot=rng.choice(changes,(2000,len(changes)),replace=True).mean(axis=1)
        records.append(dict(specification=name,weak_history=feature_string,intensity_coordinate=intensity,
          n_families=len(shared),risk_error_history=float(ra),risk_plus_dmis=float(rb),
          risk_reduction=float(ra-rb),ci95_lower=float(np.percentile(boot,2.5)),
          ci95_upper=float(np.percentile(boot,97.5)),positive_folds=int((changes>0).sum()),
          evidence='Synthetic only; model-family-out; supplementary specification sensitivity'))
        for held in shared:
            folds.append(dict(specification=name,held_out_family=held,error_history_mae=a[held],
                              plus_dmis_mae=b[held],risk_reduction=a[held]-b[held]))
    pd.DataFrame(records).to_csv(args.out/'Table2_H1_weak_history_intensity_specification.csv',index=False)
    pd.DataFrame(folds).to_csv(args.out/'model_family_out_specification_folds.csv',index=False)
    (args.out/'H1_SCOPE.txt').write_text('PRIMARY H1: Synthetic-only model-family-out. SUMO excluded. '+
         'Weak-history/intensity rows are sensitivity specifications, NOT held-out setting folds. '+
         'Synthetic frozen raw results are not modified.\n')
    print(pd.DataFrame(records).to_string(index=False));print('[PASS] H1 specification sensitivity at',args.out)
if __name__=='__main__':main()
