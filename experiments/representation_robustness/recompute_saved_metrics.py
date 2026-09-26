#!/usr/bin/env python3
"""CPU-only recomputation of Section 5.6 metrics from already audited Z0/Zk pairs.

This script does NOT rerun forecasting models and does NOT touch frozen H1 files.
It replaces only the derived alternative-response columns in the Section 5.6
weak_representation_metrics.csv files, after backing up the previous CSV once.

SVCCA/PWCCA protocol:
  * paired row-centering, identical to the original 5.6 implementation;
  * 99% PCA variance target;
  * outcome-independent component cap floor((n_rows-1)/2) on each CCA side.
For n_rows=32 this cap is 15.  The cap is fixed from sample geometry, never from
strong-RD outcomes or H1 performance.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from response_metrics import compute_measures, PCA_VARIANCE

PROJECT = Path('/data/wanganna/CausalGap')
OUT_REL = Path('experiment_results/representation_robustness')
RAW_REL = Path('experiment_results/incremental_value/analysis/h1/intermediate/01_h1_raw_calibrated_rows.csv')
KEY = ['setting','model','model_seed','intervention_type','paper_intensity_level','intervention_seed']
METRIC_COLS = ['resp_1mcka','resp_1msvcca','resp_rsa_distance','resp_1mpwcca',
               'svcca_kx','svcca_ky','pwcca_kx','pwcca_ky','n_rows','n_features',
               'pca_variance','cca_dim_cap']


def exact_seed_metrics_files(root: Path):
    out=[]
    for p in sorted((root/'representation_pairs').glob('*/trackB/seed*/weak_representation_metrics.csv')):
        if re.fullmatch(r'seed\d+', p.parent.name):
            out.append(p)
    return out


def tuple_key(row):
    return (str(row['setting']), str(row['model']), int(row['model_seed']),
            str(row['intervention_type']), str(row['paper_intensity_level']), int(row['intervention_seed']))


def resolve_artifact(row, project_root: Path) -> Path:
    candidates=[]
    if 'artifact_path' in row and pd.notna(row['artifact_path']):
        candidates.append(str(row['artifact_path']))
    for c in ('z0_path','zk_path'):
        if c in row and pd.notna(row[c]):
            candidates.append(str(row[c]).split('::',1)[0])
    for x in candidates:
        p=Path(x)
        if not p.is_absolute():
            p=project_root/p
        if p.is_file():
            return p
    raise FileNotFoundError(f'Cannot resolve saved Z0/Zk artifact for row: {dict(row)}')


def desc(x: pd.Series):
    a=pd.to_numeric(x,errors='coerce').to_numpy(float)
    a=a[np.isfinite(a)]
    if not len(a): return {'n':0}
    q=np.quantile(a,[0,.01,.05,.25,.5,.75,.95,.99,1])
    return {
        'n':int(len(a)), 'mean':float(np.mean(a)), 'std':float(np.std(a,ddof=1)) if len(a)>1 else 0.0,
        'min':float(q[0]), 'p01':float(q[1]), 'p05':float(q[2]), 'p25':float(q[3]),
        'p50':float(q[4]), 'p75':float(q[5]), 'p95':float(q[6]), 'p99':float(q[7]), 'max':float(q[8]),
        'exact_zero':int(np.sum(a==0)), 'near_zero_lt_1e-8':int(np.sum(np.abs(a)<1e-8)),
        'unique':int(pd.Series(a).nunique())
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=PROJECT)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--no-backup',action='store_true',help='Not recommended; skip one-time CSV backup')
    a=ap.parse_args()
    out=a.project_root/OUT_REL
    raw_path=a.project_root/RAW_REL
    if not raw_path.is_file(): raise FileNotFoundError(raw_path)
    raw=pd.read_csv(raw_path)
    need=set(KEY+['h1_phase'])
    if need-set(raw.columns): raise RuntimeError(f'Frozen H1 raw file missing {sorted(need-set(raw.columns))}')
    frozen=raw[(raw['setting'].astype(str)=='synthetic') & (raw['h1_phase'].astype(str)=='weak')].copy()
    for c in ('model_seed','intervention_seed'):
        frozen[c]=pd.to_numeric(frozen[c],errors='raise').astype(int)
    if frozen.duplicated(KEY).any(): raise RuntimeError('Frozen H1 contains duplicate weak keys')
    frozen_keys={tuple(x) for x in frozen[KEY].itertuples(index=False,name=None)}
    files=exact_seed_metrics_files(out)
    if not files: raise FileNotFoundError(f'No exact seed*/weak_representation_metrics.csv under {out}')
    print(f'[FROZEN] {len(frozen_keys)} weak keys; [FILES] {len(files)} exact seed directories',flush=True)

    before_frames=[]; after_frames=[]; seen=set(); changed_files=0
    for fi,p in enumerate(files,1):
        d=pd.read_csv(p)
        missing=set(KEY)-set(d.columns)
        if missing: raise RuntimeError(f'{p}: missing keys {sorted(missing)}')
        for c in ('model_seed','intervention_seed'):
            d[c]=pd.to_numeric(d[c],errors='raise').astype(int)
        if d.duplicated(KEY).any(): raise RuntimeError(f'{p}: duplicate keys')
        keys=[tuple(x) for x in d[KEY].itertuples(index=False,name=None)]
        extras=[k for k in keys if k not in frozen_keys]
        if extras: raise RuntimeError(f'{p}: contains non-frozen H1 rows, e.g. {extras[:3]}')
        seen.update(keys)
        before_frames.append(d[[c for c in METRIC_COLS if c in d.columns]].copy())
        new=d.copy()
        for idx,row in d.iterrows():
            artifact=resolve_artifact(row,a.project_root)
            with np.load(artifact,allow_pickle=False) as z:
                if {'Z0','Zk'}-set(z.files): raise RuntimeError(f'{artifact}: missing Z0/Zk')
                z0=np.asarray(z['Z0'],dtype=np.float32)
                zk=np.asarray(z['Zk'],dtype=np.float32)
            m=compute_measures(z0,zk)
            for k,v in m.items(): new.at[idx,k]=v
            new.at[idx,'metric_protocol']='CCA_dimcap_v5_centered_rank_half'
            new.at[idx,'cca_dim_rule']='min(k99, floor((n_rows-1)/2))'
        after_frames.append(new[[c for c in METRIC_COLS if c in new.columns]].copy())
        if not a.dry_run:
            backup=p.with_name('weak_representation_metrics.pre_cca_dimcap_v5.csv')
            if not a.no_backup and not backup.exists():
                shutil.copy2(p,backup)
            tmp=p.with_suffix('.tmp')
            new.to_csv(tmp,index=False)
            os.replace(tmp,p)
        changed_files += 1
        print(f'[{fi}/{len(files)}] recomputed {len(new)} rows: {p}',flush=True)

    if seen != frozen_keys:
        missing=sorted(frozen_keys-seen)[:10]
        extra=sorted(seen-frozen_keys)[:10]
        raise RuntimeError(f'Recomputed coverage != frozen H1: missing={missing}, extra={extra}, seen={len(seen)}, frozen={len(frozen_keys)}')

    before=pd.concat(before_frames,ignore_index=True)
    after=pd.concat(after_frames,ignore_index=True)
    audit={
        'protocol':'CCA_dimcap_v5_centered_rank_half',
        'rule':'For each centered n-row pair, PCA keeps up to 99% variance but no more than floor((n-1)/2) components on either CCA side.',
        'pca_variance':float(PCA_VARIANCE),
        'n_frozen_weak_rows':int(len(frozen_keys)),
        'n_metric_files':int(len(files)),
        'dry_run':bool(a.dry_run),
        'before':{c:desc(before[c]) for c in ('resp_1msvcca','resp_1mpwcca') if c in before},
        'after':{c:desc(after[c]) for c in ('resp_1msvcca','resp_1mpwcca') if c in after},
        'after_dims':{c:desc(after[c]) for c in ('svcca_kx','svcca_ky','pwcca_kx','pwcca_ky','cca_dim_cap') if c in after}
    }
    out.mkdir(parents=True,exist_ok=True)
    audit_path=out/'cca_dimcap_v5_recompute_audit.json'
    if not a.dry_run:
        audit_path.write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(audit,indent=2),flush=True)
    print(f'[DONE] CPU-only metric recomputation; changed_files={changed_files}; audit={audit_path}',flush=True)

if __name__=='__main__':
    main()
