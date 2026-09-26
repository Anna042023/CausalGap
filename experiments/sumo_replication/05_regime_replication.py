#!/usr/bin/env python3
"""SUMO only: DMIS/RD quadrants, family distributions, cross-setting ranking & regime replication.
No H1 nested regression, no H2/steering, no held-out-setting generalization.
"""
import argparse, pathlib, json, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from config import EXPERIMENT
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'model_eval'))
from protocol import MODEL_FAMILIES, FAMILY_NAMES
REGIMES=['stable_invariance','adaptive_response','pathological_rigidity','over_reactive_instability']

def collect(root,setting):
    rows=[];counts=[]
    for model in MODEL_FAMILIES:
        folders=sorted((root/'runs'/setting/model/'trackB').glob('seed*'))
        for fold in folders:
            f=fold/'causalgap_results.csv'
            if not f.exists():continue
            d=pd.read_csv(f)
            d['setting']=setting;d['model']=model
            d['model_family']=MODEL_FAMILIES[model]
            rows.append(d);counts.append((model,len(d)))
    return (pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()),counts

def prepare(df):
    for c in ['DMIS','RD','representation_response_raw']:
        if c not in df.columns:raise ValueError(f'Missing evaluator column {c}')
    if 'h1_phase' not in df.columns:
        if 'paper_intensity_level' not in df.columns:
            raise ValueError('Missing paper_intensity_level; cannot assign weak/strong')
        df['h1_phase'] = df.apply(lambda r: 'strong' if str(r['paper_intensity_level']) in
            ('s=0.20','gamma=0.2','s=0.60') else 'weak',axis=1)
    df=df[df['evaluation_track'].astype(str).str.upper().eq('B')].copy()
    df['DMIS']=pd.to_numeric(df['DMIS'],errors='coerce')
    df['RD']=pd.to_numeric(df['RD'],errors='coerce')
    return df.replace([np.inf,-np.inf],np.nan).dropna(subset=['DMIS','RD'])

def assign(df,thresholds):
    a=df.copy();x=a.DMIS>=thresholds['dmis'];y=a.RD>=thresholds['rd']
    a['regime']=np.select([~x&~y,x&~y,~x&y,x&y],REGIMES,default='invalid')
    return a

def run(sumo_root,synthetic_root,out):
    out.mkdir(parents=True,exist_ok=True)
    s,scount=collect(sumo_root,'sumo_metrla')
    if s.empty:raise RuntimeError(f'No SUMO evaluator CSV in {sumo_root}')
    s=prepare(s)
    if not set(s.h1_phase.astype(str)).issuperset({'weak','strong'}):
        raise ValueError('Both weak and strong physically simulated outcomes required')
    # Thresholds ONLY from weak interventions, within setting. Strong RD never sets boundaries.
    thresholds={}
    for setting,raw in [('sumo_metrla',s)]:
        weak=raw[raw.h1_phase.eq('weak')]
        if weak.empty:raise ValueError('No weak interventions to fit thresholds')
        thresholds[setting]={'dmis':float(weak.DMIS.median()),'rd':float(weak.RD.median()),
                             'fit':'weak_only_median_in_same_setting'}
    sumo=assign(s,thresholds['sumo_metrla'])
    sumo.to_csv(out/'sumo_instance_regimes.csv',index=False)
    family=(sumo.groupby(['model_family','regime']).size().unstack(fill_value=0)
             .reindex(columns=REGIMES,fill_value=0))
    family.div(family.sum(axis=1),axis=0).to_csv(out/'sumo_four_quadrant_distribution.csv')
    bymodel=(sumo.groupby(['model_family','model'])[['DMIS','RD']].mean().reset_index())
    bymodel.to_csv(out/'sumo_model_response_error.csv',index=False)
    # Model ordering is descriptive; do not treat matched ranks as H1 prediction.
    modelrank=bymodel.copy()
    modelrank['RD_rank_descending']=modelrank.RD.rank(ascending=False,method='average')
    modelrank['DMIS_rank_descending']=modelrank.DMIS.rank(ascending=False,method='average')
    modelrank.to_csv(out/'sumo_model_rankings_descriptive.csv',index=False)
    if synthetic_root:
        t,tcount=collect(synthetic_root,'synthetic')
        if t.empty:raise RuntimeError(f'Synthetic raw CSV not found at {synthetic_root}')
        t=prepare(t)
        tw=t[t.h1_phase.eq('weak')]
        if tw.empty:raise RuntimeError('Missing Synthetic weak labels; cannot fit thresholds')
        thresholds['synthetic']={'dmis':float(tw.DMIS.median()),'rd':float(tw.RD.median()),
                                 'fit':'weak_only_median_in_same_setting'}
        t=assign(t,thresholds['synthetic'])
        tm=t.groupby('model')[['DMIS','RD']].mean().reset_index()
        joint=bymodel.merge(tm,on='model',suffixes=('_sumo','_synthetic'))
        joint.to_csv(out/'cross_setting_model_means.csv',index=False)
        ranking=[]
        for metric in ['DMIS','RD']:
            x=joint[f'{metric}_sumo'];y=joint[f'{metric}_synthetic']
            if len(joint)>2:
                sp=spearmanr(x,y);kt=kendalltau(x,y)
                ranking.append({'metric':metric,'n_models':len(joint),'spearman_rho':sp.statistic,
                                'spearman_p':sp.pvalue,'kendall_tau':kt.statistic,'kendall_p':kt.pvalue})
        pd.DataFrame(ranking).to_csv(out/'ranking_replication_sumo_vs_synthetic.csv',index=False)
        tf=t.groupby(['model_family','regime']).size().unstack(fill_value=0).reindex(columns=REGIMES,fill_value=0)
        tf.div(tf.sum(axis=1),axis=0).to_csv(out/'synthetic_four_quadrant_distribution.csv')
        # Report disagreement rather than assert replication.
        probs=(family.div(family.sum(axis=1),axis=0)-tf.div(tf.sum(axis=1),axis=0)).abs()
        probs.to_csv(out/'family_regime_absolute_fraction_difference.csv')
    (out/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
    (out/'EVIDENCE_SCOPE.txt').write_text(
      'Abstract-SUMO counterfactual replication ONLY. Not model-family-out H1; not held-out-setting H1; not steering.\n'
      'Thresholds fit on weak interventions separately per setting. No claim of real-road/OD/trajectory alignment.\n'
      'Compare per-setting fractions and rankings descriptively; no pooled cross-setting threshold.\n')
    print('[DONE] replication diagnostics ->',out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--sumo-root',type=pathlib.Path,default=EXPERIMENT)
    p.add_argument('--synthetic-root',type=pathlib.Path,default=None)
    p.add_argument('--out',type=pathlib.Path,default=EXPERIMENT/'analysis'/'replication')
    a=p.parse_args();run(a.sumo_root,a.synthetic_root,a.out)
if __name__=='__main__':main()
