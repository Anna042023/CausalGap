#!/usr/bin/env python3
"""Paper Table21 only for ALL prespecified seeds of calibration-selected model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

CONDITIONS=['Matched_without_DMIS','Matched+DMIS','Shuffled_DMIS','Gradient','Random']
FAMILIES=['edge_removal','weight_modification','node_isolation']
KEY=['model','model_seed','intervention_type','level','intervention_seed']


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--selection-file',type=Path,required=True)
    ap.add_argument('--bootstrap',type=int,default=2000)
    args=ap.parse_args()
    out=args.output_root;lock=json.loads(args.selection_file.read_text())
    model=lock['selected_model'];seeds=lock['model_seeds']
    raw=[];issues=[];run_settings=[]
    if seeds!=[0,1,2,3,4]:
        issues.append('Paper protocol requires selected-model checkpoint seeds 0,1,2,3,4; do not publish a one-seed smoke table')
    for seed in seeds:
        rd=out/model/f'seed{seed}'
        f=rd/'h2_paired_steering.csv';m=rd/'h2_manifest.json'
        if not f.exists() or not m.exists():
            issues.append(f'Missing seed{seed} result/manifest');continue
        try:
            md=json.loads(m.read_text());d=pd.read_csv(f)
            if md['model']!=model or md['model_seed']!=seed:raise RuntimeError('model/seed mismatch')
            if md.get('checkpoint_sha256')!=lock['checkpoint_sha256_by_seed'][str(seed)]:raise RuntimeError('checkpoint hash mismatch')
            if md.get('selected_model_lock_file')!=str(args.selection_file):raise RuntimeError('selection lock mismatch')
            if md.get('calibrated_on')!='independent_Dcal x Iweak only' or md.get('tested_on')!='Dtest x Istrong only':raise RuntimeError('wrong data splits')
            if not (md.get('verified_original_readout') and md.get('readout_identity_checked') and md.get('forecast_model_frozen')):raise RuntimeError('unverified frozen readout')
            if md.get('cal_seeds')!=lock['cal_topology_seeds']:raise RuntimeError('different calibration seeds')
            setting=(md.get('steering_config',{}),md.get('perturb_ratio'),
                     md.get('max_cal_windows'),md.get('max_strong_windows'),
                     md.get('independent_calibration_rollout_seed'))
            if run_settings and setting!=run_settings[0]:raise RuntimeError('Different steering settings across seeds')
            run_settings.append(setting)
            if d.duplicated(KEY+['condition']).any():raise RuntimeError('duplicated paired units')
            if not d.model.eq(model).all() or not d.model_seed.eq(seed).all():raise RuntimeError('row model/seed mismatch')
            units=d.groupby(KEY)['condition'].agg(lambda x:set(x))
            if len(units)!=15 or not all(x==set(CONDITIONS) for x in units):raise RuntimeError('incomplete five-condition coverage')
            expected={(kind,s) for kind in FAMILIES for s in range(5)}
            got={(r.intervention_type,int(r.intervention_seed)) for r in d[KEY].drop_duplicates().itertuples()}
            if got!=expected:raise RuntimeError('incomplete strong environment coverage')
            if not np.isfinite(d[['IPE_before','IPE_after','error_reduction_pct']].to_numpy(float)).all():raise RuntimeError('nonfinite metrics')
            values=100*(d.IPE_before-d.IPE_after)/d.IPE_before.clip(lower=1e-12)
            if not np.allclose(values,d.error_reduction_pct,rtol=1e-5,atol=1e-5):raise RuntimeError('incorrect delta IPE')
            before=d.groupby(KEY)['IPE_before'].agg(lambda x: x.max()-x.min())
            if before.max()>1e-5:raise RuntimeError('methods lack paired IPE_before')
            raw.append(d)
        except Exception as exc:
            issues.append(f'seed{seed}: {exc}')
    (out/'h2_selected_audit.json').write_text(json.dumps(dict(
        selected_model=model,model_seeds=seeds,issues=issues,
        manuscript_table_ready=not issues,original_calibration_only_lock=lock),indent=2,ensure_ascii=False))
    if issues:
        (out/'Table21_H2_NOT_READY.txt').write_text('\n'.join(issues))
        old=out/'Table21_H2_steering.csv'
        if old.exists():old.unlink()
        print('[H2 INCOMPLETE]',issues);return 2
    d=pd.concat(raw,ignore_index=True)
    d.to_csv(out/'h2_paired_steering_all_selected_seeds.csv',index=False)
    paired=d.pivot(index=KEY,columns='condition',values='error_reduction_pct').reset_index()
    paired['Matched+DMIS_minus_Matched']=paired['Matched+DMIS']-paired['Matched_without_DMIS']
    paired.to_csv(out/'h2_selected_paired_units.csv',index=False)
    # Equal weight for each model seed and intervention family (no treating overlapping windows as IID).
    macro=paired.groupby(['model_seed','intervention_type'],as_index=False)[CONDITIONS].mean()
    tab=[]
    for method in CONDITIONS:
        tab.append(dict(Direction=method,
                        Error_reduction_pct=float(macro[method].mean()),
                        Difference_vs_matched_pct=float((macro[method]-macro['Matched_without_DMIS']).mean()),
                        Role={'Matched_without_DMIS':'Confirmatory baseline','Matched+DMIS':'Confirmatory condition',
                              'Shuffled_DMIS':'Placebo','Gradient':'Auxiliary label-oracle',
                              'Random':'Auxiliary'}[method],
                        Model=model,Model_seeds=len(seeds),Paired_units=len(paired)))
    table=pd.DataFrame(tab)
    table.to_csv(out/'Table21_H2_steering.csv',index=False)
    # Seed-clustered descriptive CI, very small number of independent model seeds.
    rng=np.random.default_rng(2027)
    effects=macro.assign(effect=macro['Matched+DMIS']-macro['Matched_without_DMIS']).groupby('model_seed').effect.mean().to_numpy(float)
    draws=np.mean(rng.choice(effects,size=(args.bootstrap,len(effects)),replace=True),axis=1)
    ci=np.quantile(draws,[.025,.975])
    pd.DataFrame([dict(Model=model,paired_effect_pct=float(np.mean(effects)),
      seed_cluster_bootstrap_CI_low=float(ci[0]),seed_cluster_bootstrap_CI_high=float(ci[1]),
      n_model_seeds=len(effects),note='Descriptive seed-cluster bootstrap; not independent window CI')]).to_csv(
      out/'H2_primary_paired_effect.csv',index=False)
    print(table.to_string(index=False))
    print('[H2 READY]',out/'Table21_H2_steering.csv')
    return 0

if __name__=='__main__':raise SystemExit(main())
