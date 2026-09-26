#!/usr/bin/env python3
"""CausalGap §5.5: frozen, complete weak-RD-matched edge-weight cohort.

Primary endpoint: one row per *unordered model pair*, matched using mean weak
RD over the THREE declared weak edge-weight levels. The higher/lower responder
is determined by mean weak DMIS_cal. Strong RD is read ONLY AFTER the weak
selection has been written to disk, then compared at identical intervention
seed and gamma=0.2. Never match on, select by, or sort by strong outcomes.

No predictor training, inference, or synthetic regeneration is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DEFAULT = Path('/data/wanganna/CausalGap/experiment_results/incremental_value')
MODELS = ('STGCN','DCRNN','ASTGNN','GWNet','AGCRN','STEVE','DCGCN',
          'ST-Norm','CCHMM','CausalGRIT','TF-MVGNN','PatchTST','iTransformer',
          'TimesNet','FEDformer','DLinear','Autoformer','Chronos','TimesFM',
          'Moirai','Lag-Llama')
LEVELS = {
    'weight_modification': ('gamma=0.8', 'gamma=0.6', 'gamma=0.4', 'gamma=0.2'),
    'edge_removal': ('s=0.05', 's=0.10', 's=0.15', 's=0.20'),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--experiment-root', type=Path, default=ROOT_DEFAULT)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--source', choices=['h1-snapshot','runs'], default='h1-snapshot',
                   help='h1-snapshot is the frozen H1 calibrated intermediate; runs is a separately audited fallback.')
    p.add_argument('--stage', choices=['all','select','evaluate'], default='all')
    p.add_argument('--intervention-type', choices=list(LEVELS), default='weight_modification',
                   help='edge_removal is exploratory and NEVER replaces the primary Table 18.')
    p.add_argument('--eta', type=float, default=0.05)
    p.add_argument('--model-seeds', nargs='+', type=int, default=[0])
    p.add_argument('--eval-seeds', nargs='+', type=int, default=[0,1,2,3,4])
    p.add_argument('--expected-models', type=int, default=21)
    p.add_argument('--allow-incomplete', action='store_true',
                   help='Output diagnostic PARTIAL table rather than claiming the full cohort.')
    a = p.parse_args()
    if not (math.isfinite(a.eta) and a.eta >= 0):
        p.error('--eta must be nonnegative and finite')
    if a.expected_models < 2 or not a.eval_seeds or not a.model_seeds:
        p.error('Need >=2 models and nonempty model-seeds/eval-seeds')
    if len(set(a.model_seeds)) != len(a.model_seeds) or len(set(a.eval_seeds)) != len(a.eval_seeds):
        p.error('Duplicate seeds are not permitted')
    if a.output is None:
        a.output = a.experiment_root / 'analysis' / 'matched_error'
    if a.intervention_type != 'weight_modification':
        a.output = a.output / ('EXPLORATORY_' + a.intervention_type)
    return a


def is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(['true','1','yes'])


def file_hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda: f.read(1024*1024), b''):
            h.update(c)
    return h.hexdigest()


def json_save(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding='utf-8')


def calibration_rank(value: float, ref: np.ndarray) -> float:
    """Same mid-tie / +1 empirical distance calibration as H1 build_h1_units.py."""
    less = int(np.sum(ref < value))
    equal = int(np.sum(np.isclose(ref, value, rtol=1e-10, atol=1e-12)))
    return float(min((1. + less + 0.5 * equal) / (len(ref) + 1.), 1.))


def load_data(args, phase: str) -> tuple[pd.DataFrame, list[str]]:
    """Returns only requested phase. No strong data are used by selection."""
    levels = LEVELS[args.intervention_type]
    wanted = levels[:3] if phase=='weak' else levels[3:]
    paths=[]
    if args.source == 'h1-snapshot':
        paths=[args.experiment_root/'analysis/h1/intermediate/01_h1_raw_calibrated_rows.csv']
    else:
        for sd in sorted((args.experiment_root/'runs/synthetic').glob('*/trackB/seed*')):
            try: seed = int(sd.name[4:])
            except ValueError: continue
            if seed in args.model_seeds:
                paths.append(sd/'causalgap_results.csv')
    if not paths or any(not x.is_file() for x in paths):
        bad = [str(x) for x in paths if not x.is_file()]
        raise FileNotFoundError('Missing H1 file(s): '+repr(bad or [str(args.experiment_root)]))
    pieces=[]
    for path in paths:
        for chunk in pd.read_csv(path, chunksize=25000, low_memory=False):
            if 'model_seed' not in chunk.columns:
                if args.source == 'runs': chunk['model_seed'] = int(path.parent.name[4:])
                else: raise ValueError('Frozen H1 snapshot missing model_seed')
            need=['model','intervention_type','paper_intensity_level','intervention_seed',
                  'model_seed','evaluation_track','causal_valid','leakage_detected','RD']
            missing=[x for x in need if x not in chunk]
            if missing: raise ValueError(f'{path}: missing columns {missing}')
            mask=(chunk['intervention_type'].astype(str).eq(args.intervention_type)
                  & chunk['paper_intensity_level'].astype(str).isin(wanted)
                  & chunk['evaluation_track'].astype(str).str.upper().eq('B')
                  & is_true(chunk['causal_valid']) & ~is_true(chunk['leakage_detected'])
                  & pd.to_numeric(chunk['model_seed'],errors='coerce').isin(args.model_seeds)
                  & pd.to_numeric(chunk['intervention_seed'],errors='coerce').isin(args.eval_seeds))
            d=chunk.loc[mask].copy()
            if len(d): pieces.append(d)
    if not pieces: raise RuntimeError(f'No valid {phase} Track B synthetic rows for requested seeds')
    df=pd.concat(pieces,ignore_index=True)
    for c in ['intervention_seed','model_seed']:
        df[c]=pd.to_numeric(df[c],errors='raise').astype(int)
    df['RD']=pd.to_numeric(df['RD'],errors='raise')
    if not np.isfinite(df['RD']).all(): raise ValueError(f'Nonfinite {phase} RD')
    if phase=='weak':
        if args.source=='h1-snapshot':
            if 'DMIS_cal' not in df: raise ValueError('H1 snapshot does not contain DMIS_cal')
            df['DMIS_cal']=pd.to_numeric(df['DMIS_cal'],errors='raise')
        else:
            if 'representation_response_raw' not in df or 'environment_distance' not in df:
                raise ValueError('Runs source needs representation_response_raw and environment_distance')
            ref_file=args.experiment_root/'environments/synthetic/calibration_distance_reference.csv'
            ref_df=pd.read_csv(ref_file)
            ref= pd.to_numeric(ref_df.loc[ref_df.intervention_type.astype(str).eq(args.intervention_type),
                                          'environment_distance'],errors='coerce').dropna().to_numpy(float)
            if not len(ref): raise ValueError(f'Empty distance calibration for {args.intervention_type}')
            df['DMIS_cal']=[float(dz)/max(calibration_rank(float(dist),ref),1e-6)
                            for dz,dist in zip(df['representation_response_raw'],df['environment_distance'])]
        if not np.isfinite(df['DMIS_cal']).all(): raise ValueError('Nonfinite weak DMIS_cal')
    key=['model','model_seed','intervention_type','paper_intensity_level','intervention_seed']
    dup=df.duplicated(key,keep=False)
    if dup.any():
        raise ValueError(f'{phase} duplicate measurement rows: {df.loc[dup,key].head().to_dict("records")}')
    return df, [str(x) for x in paths]


def check_coverage(df: pd.DataFrame, args, phase: str) -> tuple[bool, dict]:
    expected_levels=list(LEVELS[args.intervention_type][:3 if phase=='weak' else 4])
    if phase=='strong': expected_levels=[expected_levels[-1]]
    models=sorted(df['model'].unique())
    actual=set(zip(df.model,df.model_seed,df.paper_intensity_level,df.intervention_seed))
    requested={(m,s,l,e) for m in models for s in args.model_seeds for l in expected_levels for e in args.eval_seeds}
    missing=sorted(requested-actual)
    unknown_models=sorted(set(models)-set(MODELS))
    missing_models=sorted(set(MODELS)-set(models)) if args.expected_models==21 else []
    passed=(len(models)==args.expected_models and not missing and not unknown_models and not missing_models)
    audit=dict(phase=phase,n_models=len(models),models=models,n_rows=len(df),expected_models=args.expected_models,
               missing_cells_count=len(missing),missing_cells_preview=missing[:30],
               unknown_models=unknown_models,missing_models=missing_models,
               requested_model_seeds=args.model_seeds,requested_intervention_seeds=args.eval_seeds)
    if not passed and not args.allow_incomplete:
        raise RuntimeError(f'INCOMPLETE {phase} coverage: '+json.dumps(audit,default=str))
    return passed,audit


def select(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    d,inputs=load_data(args,'weak')
    full,audit=check_coverage(d,args,'weak')
    # First average intervention seeds per model/checkpoint/weak intensity,
    # then equal-average checkpoint seeds, then 3 predefined weak levels.
    cfg=d.groupby(['model','model_seed','paper_intensity_level'],as_index=False).agg(
        weak_RD=('RD','mean'),weak_DMIScal=('DMIS_cal','mean'),n_eval_seeds=('intervention_seed','nunique'))
    cfg.to_csv(out/'01_weak_configuration_means.csv',index=False)
    model=cfg.groupby('model',as_index=False).agg(
        weak_RD_mean=('weak_RD','mean'),weak_DMIScal_mean=('weak_DMIScal','mean'),
        weak_level_checkpoint_count=('weak_RD','size'))
    model=model.sort_values('model').reset_index(drop=True)
    model.to_csv(out/'02_weak_model_summaries.csv',index=False)
    candidates=[];eligible=[];tied=[]
    for r1,r2 in itertools.combinations(model.to_dict('records'),2):
        m1,m2=r1['model'],r2['model']
        gap=abs(r1['weak_RD_mean']-r2['weak_RD_mean'])
        match=bool(gap <= args.eta + 1e-12)
        row=dict(model_a=m1,model_b=m2,weak_RD_a=float(r1['weak_RD_mean']),
                 weak_RD_b=float(r2['weak_RD_mean']),weak_RD_gap=float(gap),
                 weak_DMIScal_a=float(r1['weak_DMIScal_mean']),
                 weak_DMIScal_b=float(r2['weak_DMIScal_mean']),eligible_RD=match)
        candidates.append(row)
        if not match:continue
        if np.isclose(r1['weak_DMIScal_mean'],r2['weak_DMIScal_mean'],rtol=1e-12,atol=1e-12):
            tied.append(row);continue
        higher,lower=(r1,r2) if r1['weak_DMIScal_mean']>r2['weak_DMIScal_mean'] else (r2,r1)
        eligible.append({**row,'higher_DMIS_model':higher['model'],'lower_DMIS_model':lower['model'],
                         'higher_weak_RD':float(higher['weak_RD_mean']),
                         'lower_weak_RD':float(lower['weak_RD_mean']),
                         'higher_weak_DMIScal':float(higher['weak_DMIScal_mean']),
                         'lower_weak_DMIScal':float(lower['weak_DMIScal_mean'])})
    if not eligible: raise RuntimeError('No non-tied RD-matched model pairs. Do not invent a cohort.')
    pairs=pd.DataFrame(eligible)
    pairs.insert(0,'pair_id',[f'WM_{i:04d}' if args.intervention_type=='weight_modification'
                              else f'ER_{i:04d}' for i in range(1,len(pairs)+1)])
    pd.DataFrame(candidates).to_csv(out/'03_all_weak_model_pairs.csv',index=False)
    pd.DataFrame(tied).to_csv(out/'04_weak_DMIS_ties_excluded.csv',index=False)
    selected=out/'05_eligible_pairs_WEAK_ONLY.csv'
    pairs.to_csv(selected,index=False)
    json_save(out/'05_selection_frozen.json',dict(
        source=args.source, input_paths=inputs, source_sha256={p:file_hash(Path(p)) for p in inputs},
        stage='selection_frozen_before_strong_lookup', intervention_type=args.intervention_type,
        eta=args.eta, primary=(args.intervention_type=='weight_modification' and abs(args.eta-.05)<1e-12),
        model_seeds=args.model_seeds,eval_seeds=args.eval_seeds,expected_models=args.expected_models,
        full_cohort=full,coverage=audit,n_candidate_pairs=len(candidates),n_RD_matched=len(eligible)+len(tied),
        n_tied_DMIS_excluded=len(tied),n_ordered_eligible_pairs=len(pairs),
        weak_pair_file=str(selected),weak_pair_sha256=file_hash(selected),
        convention='mean of THREE weak weight-modification levels, mean across intervention seeds and model seeds; high vs low by weak DMIS_cal only'))
    print(f'[SELECT] {len(model)} models, {len(candidates)} candidate pairs, {len(pairs)} complete eligible pairs; selection frozen: {selected}',flush=True)


def evaluate(args):
    out=args.output
    lock_path=out/'05_selection_frozen.json'
    if not lock_path.exists():raise FileNotFoundError(f'Run --stage select first: {lock_path}')
    lock=json.loads(lock_path.read_text(encoding='utf8'))
    selected=out/'05_eligible_pairs_WEAK_ONLY.csv'
    if file_hash(selected)!=lock['weak_pair_sha256']:
        raise RuntimeError('Weak selection file changed after freezing; stop.')
    for key,value in [('eta',args.eta),('intervention_type',args.intervention_type),
                      ('source',args.source),('model_seeds',args.model_seeds),('eval_seeds',args.eval_seeds)]:
        if lock[key]!=value: raise RuntimeError(f'Selection lock mismatch for {key}: {lock[key]} != {value}')
    # A complete cohort in default mode is required BEFORE looking at strong RD.
    if not lock['full_cohort'] and not args.allow_incomplete:
        raise RuntimeError('Incomplete weak cohort is diagnostic only.')
    pairs=pd.read_csv(selected)
    strong,inputs=load_data(args,'strong')
    current_hashes={p:file_hash(Path(p)) for p in inputs}
    if current_hashes != lock['source_sha256']:
        raise RuntimeError('H1 source changed since weak cohort was frozen; do not combine different result snapshots.')
    full,strong_audit=check_coverage(strong,args,'strong')
    if not full and not args.allow_incomplete:raise RuntimeError('Incomplete strong cohort.')
    # Use same intervention seed on both sides; model checkpoints are equal
    # weighted when multiple were prospectively requested.
    ss=strong.groupby(['model','intervention_seed'],as_index=False).agg(strong_RD=('RD','mean'),
                                                                        model_checkpoint_count=('model_seed','nunique'))
    ss.to_csv(out/'06_strong_model_seed_means.csv',index=False)
    hi=ss.rename(columns={'model':'higher_DMIS_model','strong_RD':'strong_RD_higher_DMIS',
                          'model_checkpoint_count':'higher_model_checkpoint_count'})
    lo=ss.rename(columns={'model':'lower_DMIS_model','strong_RD':'strong_RD_lower_DMIS',
                          'model_checkpoint_count':'lower_model_checkpoint_count'})
    # Expand one weak-selected model pair onto the FROZEN strong intervention seeds.
    expanded=pairs.assign(_join=1).merge(pd.DataFrame({'intervention_seed':args.eval_seeds,'_join':1}),on='_join').drop(columns='_join')
    paired=expanded.merge(hi,on=['higher_DMIS_model','intervention_seed'],how='left',validate='many_to_one')
    paired=paired.merge(lo,on=['lower_DMIS_model','intervention_seed'],how='left',validate='many_to_one')
    if paired[['strong_RD_higher_DMIS','strong_RD_lower_DMIS']].isna().any().any():
        bad=paired.loc[paired[['strong_RD_higher_DMIS','strong_RD_lower_DMIS']].isna().any(axis=1),
                       ['pair_id','intervention_seed','higher_DMIS_model','lower_DMIS_model']]
        raise RuntimeError('Missing same-seed strong RD for selected pairs: '+str(bad.head(15).to_dict('records')))
    paired['strong_RD_difference_high_minus_low']=paired.strong_RD_higher_DMIS-paired.strong_RD_lower_DMIS
    paired['intervention_type']=args.intervention_type
    paired['strong_level']=LEVELS[args.intervention_type][-1]
    paired.to_csv(out/'07_matched_strong_seed_rows.csv',index=False)
    cohort=paired.groupby('pair_id',as_index=False).agg(
        higher_DMIS_model=('higher_DMIS_model','first'),lower_DMIS_model=('lower_DMIS_model','first'),
        weak_RD_gap=('weak_RD_gap','first'),higher_weak_RD=('higher_weak_RD','first'),
        lower_weak_RD=('lower_weak_RD','first'),
        higher_weak_DMIScal=('higher_weak_DMIScal','first'),lower_weak_DMIScal=('lower_weak_DMIScal','first'),
        strong_RD_higher_DMIS=('strong_RD_higher_DMIS','mean'),
        strong_RD_lower_DMIS=('strong_RD_lower_DMIS','mean'),
        strong_RD_difference_high_minus_low=('strong_RD_difference_high_minus_low','mean'),
        n_shared_intervention_seeds=('intervention_seed','nunique'))
    if not (cohort.n_shared_intervention_seeds==len(args.eval_seeds)).all():
        raise RuntimeError('Not all pairs have identical complete strong-seed coverage.')
    cohort.to_csv(out/'08_complete_matched_error_cohort.csv',index=False)
    tbl=pd.DataFrame([{
        'Tolerance':args.eta,
        'Eligible_pairs':len(cohort),
        'Weak_RD_gap':cohort.weak_RD_gap.mean(),
        'Strong_RD_higher_DMIS':cohort.strong_RD_higher_DMIS.mean(),
        'Strong_RD_lower_DMIS':cohort.strong_RD_lower_DMIS.mean(),
        'Difference_high_minus_low':cohort.strong_RD_difference_high_minus_low.mean(),
    }])
    paper=(full and lock['full_cohort'] and args.intervention_type=='weight_modification' and abs(args.eta-0.05)<1e-12)
    table_name='Table18_matched_error_cohort.csv' if paper else 'Table18_PARTIAL_OR_EXPLORATORY_NOT_FOR_PAPER.csv'
    tbl.to_csv(out/table_name,index=False)
    # GWNet–PatchTST is descriptive ONLY if it belongs to the complete cohort.
    illustration=cohort[cohort.apply(lambda r:set([r['higher_DMIS_model'],r['lower_DMIS_model']])=={'GWNet','PatchTST'},axis=1)]
    if len(illustration): illustration.to_csv(out/'09_GWNet_PatchTST_illustrative_pair.csv',index=False)
    result=dict(source=args.source,source_paths=inputs,
                selection_sha256=lock['weak_pair_sha256'],
                strong_source_sha256={p:file_hash(Path(p)) for p in inputs},
                primary_paper_ready=paper,weak_coverage=lock['coverage'],strong_coverage=strong_audit,
                eta=args.eta,intervention_type=args.intervention_type,weak_levels=list(LEVELS[args.intervention_type][:3]),
                strong_level=LEVELS[args.intervention_type][3],
                eligible_pairs=len(cohort),strong_paired_rows=len(paired),
                n_high_DMIS_better=int((cohort.strong_RD_difference_high_minus_low<0).sum()),
                n_low_DMIS_better=int((cohort.strong_RD_difference_high_minus_low>0).sum()),
                n_exact_strong_ties=int((cohort.strong_RD_difference_high_minus_low==0).sum()),
                mean_difference_high_minus_low=float(cohort.strong_RD_difference_high_minus_low.mean()),
                difference_sign='NEGATIVE means higher weak DMIS model has lower strong RD; POSITIVE means higher weak DMIS model has higher strong RD',
                caution='Model pairs share predictors; no naive pair-independence p-value or CI is claimed.')
    json_save(out/'10_matched_error_audit.json',result)
    diff=tbl.loc[0,'Difference_high_minus_low']
    print('\n[Table 18] '+str(out/table_name),flush=True)
    print(tbl.to_string(index=False,float_format=lambda x:f'{x:.7f}'),flush=True)
    print(f'[INTERPRET] high-minus-low strong RD = {diff:+.7f} (negative: higher weak DMIS -> lower strong RD); full={paper}',flush=True)


def main():
    args=parse_args()
    if args.stage in ('all','select'):select(args)
    if args.stage in ('all','evaluate'):evaluate(args)


if __name__=='__main__':main()
