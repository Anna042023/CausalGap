#!/usr/bin/env python3
"""Frozen H1 V2 alternative-coordinate reruns for Section 5.6 / Appendix A.7.5.

The FULL H1 ridge code is imported from the user's EXISTING V2 analyze_h1.py.
No standalone surrogate predictor, silent imputation, dropping, or fold alteration.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
UNIFIED=HERE.parents[1]
V2_SRC=UNIFIED/'experiments/incremental_value'
for p in (str(V2_SRC),str(UNIFIED)):
    if p not in sys.path:sys.path.insert(0,p)

PROJECT=Path('/data/wanganna/CausalGap')
RAW_REL='experiment_results/incremental_value/analysis/h1/intermediate/01_h1_raw_calibrated_rows.csv'
UNITS_REL='experiment_results/incremental_value/analysis/h1/intermediate/04_h1_prediction_units.csv'
TABLE_REL='experiment_results/incremental_value/analysis/h1/tables/Table2_H1_incremental_diagnostic_value.csv'
OUT_REL='experiment_results/representation_robustness'

MEASURES={'1-CKA':'resp_1mcka','1-SVCCA':'resp_1msvcca','RSA distance':'resp_rsa_distance','1-PWCCA':'resp_1mpwcca'}
KEY=['setting','model','model_seed','intervention_type','paper_intensity_level','intervention_seed']
GROUP=['setting','model','intervention_type','paper_intensity_level']


def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''):h.update(c)
    return h.hexdigest()


def read(p):
    if not p.is_file():raise FileNotFoundError(p)
    return pd.read_csv(p)


def check(d,required,title):
    missing=set(required)-set(d)
    if missing:raise RuntimeError(f'{title} missing columns: {sorted(missing)}')
    if d.empty:raise RuntimeError(f'{title} is empty')
    return d


def h1_analyzer():
    p=V2_SRC/'analyze_h1.py'
    if not p.exists():raise FileNotFoundError(f'Required original H1 V2 analysis script: {p}')
    spec=importlib.util.spec_from_file_location('_original_h1_v2_analyze_for56',p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ('run_outer','comparison','fit_design','inner_lambda'):
        if not hasattr(mod,fn):raise RuntimeError(f'Original H1 analyzer lacks {fn}')
    return mod


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=PROJECT)
    ap.add_argument('--stage',choices=['preflight','all'],default='preflight')
    ap.add_argument('--allow-subset',action='store_true',help='Diagnostic ONLY; never produce paper Table A.7.5')
    a=ap.parse_args()
    raw_path=a.project_root/RAW_REL
    units_path=a.project_root/UNITS_REL
    table_path=a.project_root/TABLE_REL
    out=a.project_root/OUT_REL
    r=check(read(raw_path),KEY+['h1_phase','DMIS_cal','distance_calibrated'], 'H1 V2 raw calibrated rows')
    u=check(read(units_path),['unit_id','setting','model','model_family','intervention_type','weak_DMIScal_mean','strong_RD'], 'H1 V2 units')
    t=check(read(table_path),['split','risk_reduction','risk_error_history','risk_plus_dmis'], 'H1 V2 reference table')
    primary=t[t['split'].eq('model_family')]
    if len(primary)!=1:raise RuntimeError('Expected exactly one frozen H1 model_family result')
    primary=primary.iloc[0]
    r=r[r['h1_phase'].eq('weak') & r['setting'].eq('synthetic')].copy()
    u=u[u['setting'].eq('synthetic')].copy()
    if len(u)!=63 or u.model.nunique()!=21 or u.model_family.nunique()!=5:
        if not a.allow_subset:raise RuntimeError(f'Expected full frozen Synthetic H1 63 units/21 models/5 families; got {len(u)}, {u.model.nunique()}, {u.model_family.nunique()}')
    if r.duplicated(KEY).any():raise RuntimeError('Duplicate frozen H1 key')
    if r[KEY].isna().any().any():raise RuntimeError('Incomplete frozen H1 keys')
    print(f'[H1] {len(u)} frozen units, {u.model.nunique()} models, {u.model_family.nunique()} families; {len(r)} weak rows',flush=True)
    if a.stage=='preflight':
        print('[OK] frozen H1 units + raw calibration readable; run export_h1_pairs.py on all models next',flush=True)
        return
    candidate_files=sorted((out/'representation_pairs').glob('*/trackB/seed*/weak_representation_metrics.csv'))
    files=[p for p in candidate_files if re.fullmatch(r'seed\d+',p.parent.name)]
    if not files:raise RuntimeError('No 5.6 representation export files; run run_export_all.py --stage export')
    e=pd.concat([read(p) for p in files],ignore_index=True)
    e=check(e,KEY+list(MEASURES.values())+['raw_response_absdiff','raw_response_frozen'],'Section 5.6 exported pairs')
    if e.duplicated(KEY).any():raise RuntimeError('Duplicate exported representation keys, including stale reruns; resolve before analysis')
    if e[KEY].isna().any().any():raise RuntimeError('Missing exported keys')
    m=r.merge(e,on=KEY,how='outer',validate='one_to_one',indicator=True,suffixes=('','_export'))
    if (m['_merge']!='both').any():
        problem=m.loc[m['_merge']!='both',KEY+['_merge']]
        raise RuntimeError(f'Frozen vs replay coverage mismatch ({len(problem)} unmatched):\n{problem.head(30).to_string(index=False)}')
    m=m.drop(columns='_merge')
    metric_audit=[]
    for name,col in MEASURES.items():
        val=pd.to_numeric(m[col],errors='coerce')
        if not np.isfinite(val).all():raise RuntimeError(f'{name}: undefined metrics; no row may be silently dropped')
        arr=val.to_numpy(float)
        metric_audit.append({
            'Response measure':name, 'n':int(len(arr)),
            'raw_mean':float(np.mean(arr)), 'raw_std':float(np.std(arr,ddof=1)),
            'raw_min':float(np.min(arr)), 'raw_median':float(np.median(arr)), 'raw_max':float(np.max(arr)),
            'exact_zero':int(np.sum(arr==0)), 'near_zero_lt_1e-8':int(np.sum(np.abs(arr)<1e-8)),
            'near_zero_fraction':float(np.mean(np.abs(arr)<1e-8))
        })
        if float(np.std(arr,ddof=1)) < 1e-12:
            raise RuntimeError(f'{name}: response coordinate is numerically degenerate after the fixed metric protocol')
        m[f'{col}_cal']=arr/np.maximum(pd.to_numeric(m['distance_calibrated'],errors='raise').to_numpy(float),1e-6)
    if m['raw_response_absdiff'].max()>1e-3:raise RuntimeError('Representation replay response audit fails; check exporter logs')
    if not a.allow_subset:
        if set(m.model.astype(str))!=set(u.model.astype(str)):
            raise RuntimeError('Missing model(s) in representation export')
        if m[KEY].isna().any().any():raise RuntimeError('Missing pair key(s)')
    # H1 order: seed mean for each (model, family, weak level), then equal-weight mean over exactly 3 weak levels.
    measured=[f'{col}_cal' for col in MEASURES.values()]
    cfg=m.groupby(GROUP,as_index=False)[measured+['DMIS_cal']].mean()
    count=m.groupby(GROUP).size().rename('n_seed_rows').reset_index()
    cfg=cfg.merge(count,on=GROUP,validate='one_to_one')
    coverage=cfg.groupby(['setting','model','intervention_type']).size()
    if (coverage!=3).any():raise RuntimeError(f'Expected 3 weak levels per H1 unit; bad configs {coverage[coverage!=3]}')
    agg=cfg.groupby(['setting','model','intervention_type'],as_index=False)[measured+['DMIS_cal']].mean()
    merged=u.merge(agg,on=['setting','model','intervention_type'],how='left',validate='one_to_one')
    if merged[measured].isna().any().any():raise RuntimeError('Missing alternative H1 feature(s)')
    # Test that the frozen primary weak DMIS is reconstructed from the identical weak raw row population.
    dmis_delta=np.max(np.abs(merged['DMIS_cal'].astype(float)-merged['weak_DMIScal_mean'].astype(float)))
    if dmis_delta>1e-6:raise RuntimeError(f'Frozen DMIS aggregation / selected seed mismatch: max gap={dmis_delta}; do not claim frozen folds')
    for col in measured:
        if not np.isfinite(merged[col].to_numpy(float)).all():raise RuntimeError(f'Invalid {col}')
    h1=h1_analyzer()
    # Restore/freeze exactly the V2 columns and solver. Never modify its source or original artifacts.
    lambdas=list(getattr(h1,'LAMBDA_GRID',[1e-4,1e-3,1e-2,1e-1,1,10,100]))
    if list(lambdas)!=[1e-4,1e-3,1e-2,1e-1,1,10,100]:raise RuntimeError(f'Unexpected H1 lambda grid: {lambdas}')
    r_base,f_base,_=h1.run_outer(merged,'model_family','error_history',lambdas,group_col='model')
    r_dm,f_dm,_=h1.run_outer(merged,'model_family','error_history_plus_dmis',lambdas,group_col='model')
    if abs(r_base-float(primary['risk_error_history']))>1e-8 or abs(r_dm-float(primary['risk_plus_dmis']))>1e-8:
        raise RuntimeError(f'Frozen H1 baseline mismatch: {r_base} / {r_dm}; original={primary["risk_error_history"]} / {primary["risk_plus_dmis"]}. Do not report alternate risk until reproduced.')
    folds=[f_base.assign(measure='Error history'),f_dm.assign(measure='DMIScal (frozen)')]
    orig_feats=h1.FEATURE_SETS['error_history_plus_dmis'][:]
    results=[]
    for name,col in MEASURES.items():
        chosen=f'{col}_cal'
        h1.FEATURE_SETS['error_history_plus_dmis']=h1.FEATURE_SETS['error_history']+[chosen]
        try: alt_risk, fold, _=h1.run_outer(merged,'model_family','error_history_plus_dmis',lambdas,group_col='model')
        finally:h1.FEATURE_SETS['error_history_plus_dmis']=orig_feats
        reduction=float(r_base-alt_risk)
        rhos=spearmanr(merged['weak_DMIScal_mean'].to_numpy(float),merged[chosen].to_numpy(float))
        rho=float(rhos.statistic)
        if not np.isfinite(rho):raise RuntimeError(f'{name} response ordering undefined')
        p=float(primary['risk_reduction'])
        direction='Same' if np.sign(reduction)==np.sign(p) else 'Different'
        if reduction==0 or p==0: direction='Zero / boundary'
        results.append({'Response measure':name,'Rank agreement with DMIS':rho,
                        'H1 risk reduction':reduction,'Direction vs. primary':direction,
                        'H1 baseline risk':r_base,'H1 risk with alternative':alt_risk,
                        'Positive families / 5':int((f_base['macro_fold_mae'].to_numpy()-fold['macro_fold_mae'].to_numpy()>0).sum())})
        folds.append(fold.assign(measure=name))
        print(f'[{name}] Spearman={rho:.6g}, H1 risk reduction={reduction:.8g}, {direction}',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    m.to_csv(out/'01_alternative_response_rows.csv',index=False)
    pd.DataFrame(metric_audit).to_csv(out/'00_metric_dynamic_range_audit.csv',index=False)
    merged.to_csv(out/'02_alternative_response_units.csv',index=False)
    pd.concat(folds,ignore_index=True).to_csv(out/'03_h1_fold_risks.csv',index=False)
    report=pd.DataFrame(results)
    report.to_csv(out/'Table_A7_5_alternative_measures.csv',index=False)
    audit=dict(frozen_units=str(units_path),frozen_units_sha256=sha256(units_path),
               frozen_raw=str(raw_path),frozen_raw_sha256=sha256(raw_path),
               n_h1_units=len(merged),n_models=merged.model.nunique(),n_raw_weak_rows=len(m),
               n_export_files=len(files),max_DMIScal_mean_reconstruction_gap=float(dmis_delta),
               baseline_risk_reproduced=float(r_base),primary_risk_reproduced=float(r_dm),
               primary_h1_reduction=float(primary['risk_reduction']),
               alternate_measure_scaling='raw response divided by frozen calibration distance, then seed-mean and 3-weak-level-mean',
               cca_family_protocol='PCA 99% variance with centered-rank cap floor((n_rows-1)/2) per CCA side; n=32 -> 15',
               metric_dynamic_range_audit=metric_audit,
               original_H1_module=str(V2_SRC/'analyze_h1.py'),lambda_grid=lambdas)
    (out/'representation_robustness_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf8')
    print(report.to_string(index=False),flush=True)
    print(f'[SAVED] {out/"Table_A7_5_alternative_measures.csv"}',flush=True)

if __name__=='__main__':main()
