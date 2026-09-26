#!/usr/bin/env python3
"""CausalGap Section 5.7: invariance sanity checks.

Primary audit requested by the paper:
  1) 100 shared orthogonal rotations
  2) 100 common feature permutations
  3) 100 admissible shared positive scalings

The script reuses the frozen, audited Z0/Zk representation pairs exported by
Section 5.6.  It does NOT retrain or rerun forecasting models.

For each transformation it reports:
  * maximum absolute DMIS deviation
  * mean / 95th-percentile absolute DMIS deviation (audit only)
  * Spearman rank agreement with the original DMIS (audit only)
  * operational-regime agreement under the frozen Section 5.4 threshold rule

It also creates publication-ready PDF/SVG/PNG figures and a LaTeX table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# -----------------------------------------------------------------------------
# Paths / protocol constants
# -----------------------------------------------------------------------------
PROJECT_DEFAULT = Path('/data/wanganna/CausalGap')
REP_REL = Path('experiment_results/representation_robustness/representation_pairs')
H1_WEAK_REL = Path('experiment_results/incremental_value/analysis/h1/intermediate/01_h1_raw_calibrated_rows.csv')
OUT_REL = Path('experiment_results/invariance_checks')
KEY = ['setting','model','model_seed','intervention_type','paper_intensity_level','intervention_seed']
REGIME_KEY = ['model','intervention_type','paper_intensity_level','intervention_seed']
REGIMES = [
    'Stable Invariance',
    'Adaptive Response',
    'Over-Reactive Instability',
    'Pathological Rigidity',
]
EPS_Z = 1e-12
EPS_D = 1e-6
PRIMARY_ALPHA = 0.50


def _add_unified_to_path(project_root: Path) -> Path:
    here = Path(__file__).resolve()
    candidates = [project_root / 'unified', here.parents[2] if len(here.parents) >= 3 else None]
    for p in candidates:
        if p is not None and (p / 'causalgap').is_dir():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p
    raise RuntimeError('Cannot locate unified/causalgap. Run this script from the CausalGap unified repository.')


def stable_seed(global_seed: int, *parts: object) -> int:
    s = '|'.join([str(global_seed), *map(str, parts)]).encode('utf-8')
    return int.from_bytes(hashlib.sha256(s).digest()[:8], 'little') % (2**32 - 1)


def exact_seed_metric_files(rep_root: Path) -> List[Path]:
    out = []
    for p in sorted(rep_root.glob('*/trackB/seed*/weak_representation_metrics.csv')):
        if re.fullmatch(r'seed\d+', p.parent.name):
            out.append(p)
    return out


def tuple_key(row: pd.Series | dict) -> tuple:
    return (
        str(row['setting']), str(row['model']), int(row['model_seed']),
        str(row['intervention_type']), str(row['paper_intensity_level']), int(row['intervention_seed'])
    )


def resolve_pair_artifact(row: pd.Series, project_root: Path) -> Path:
    candidates: List[str] = []
    if 'artifact_path' in row.index and pd.notna(row['artifact_path']):
        candidates.append(str(row['artifact_path']))
    for c in ('z0_path','zk_path'):
        if c in row.index and pd.notna(row[c]):
            candidates.append(str(row[c]).split('::',1)[0])
    for x in candidates:
        p = Path(x)
        if not p.is_absolute():
            p = project_root / p
        if p.is_file():
            return p
    raise FileNotFoundError(f'Cannot resolve saved Z0/Zk artifact for {dict(row)}')


def centered_norm(z: np.ndarray) -> float:
    x = np.asarray(z, dtype=np.float64).reshape(len(z), -1)
    x = x - x.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(x))


def load_frozen_weak(project_root: Path) -> pd.DataFrame:
    p = project_root / H1_WEAK_REL
    if not p.is_file():
        raise FileNotFoundError(p)
    d = pd.read_csv(p, low_memory=False)
    need = set(KEY + ['h1_phase'])
    missing = need - set(d.columns)
    if missing:
        raise RuntimeError(f'Frozen H1 weak table missing columns: {sorted(missing)}')
    q = d[(d['setting'].astype(str) == 'synthetic') & (d['h1_phase'].astype(str) == 'weak')].copy()
    for c in ('model_seed','intervention_seed'):
        q[c] = pd.to_numeric(q[c], errors='raise').astype(int)
    if q.duplicated(KEY).any():
        raise RuntimeError('Frozen H1 weak table contains duplicate keys.')
    return q


def load_metric_manifest(project_root: Path, frozen: pd.DataFrame) -> pd.DataFrame:
    rep_root = project_root / REP_REL
    files = exact_seed_metric_files(rep_root)
    if not files:
        raise FileNotFoundError(f'No Section 5.6 weak_representation_metrics.csv files under {rep_root}')
    frames=[]
    for p in files:
        x = pd.read_csv(p)
        missing = set(KEY + ['environment_distance','DMIS_raw_frozen','raw_response_frozen']) - set(x.columns)
        if missing:
            raise RuntimeError(f'{p} missing columns: {sorted(missing)}')
        for c in ('model_seed','intervention_seed'):
            x[c] = pd.to_numeric(x[c], errors='raise').astype(int)
        x['_metric_csv'] = str(p)
        frames.append(x)
    m = pd.concat(frames, ignore_index=True)
    if m.duplicated(KEY).any():
        dup = m.loc[m.duplicated(KEY, keep=False), KEY].head(10)
        raise RuntimeError('Duplicate Section 5.6 keys:\n' + dup.to_string(index=False))
    # Restrict exactly to the frozen H1 weak cohort; never ingest H2-added seeds.
    if 'RD' not in frozen.columns:
        raise RuntimeError('Frozen H1 weak table is missing RD, which is required for the Section 5.7 regime audit.')
    # Preserve the frozen weak-intervention RD from H1.  Section 5.7 should
    # classify the exact frozen 5.6 representation pair, not an ambiguous
    # historical duplicate row from the Section 5.4 export.
    fkeys = frozen[KEY + ['RD']].copy()
    fkeys['RD'] = pd.to_numeric(fkeys['RD'], errors='raise').astype(float)
    out = fkeys.merge(m, on=KEY, how='left', validate='one_to_one', indicator=True)
    bad = out[out['_merge'] != 'both']
    if len(bad):
        raise RuntimeError(f'Section 5.6 representation coverage does not match frozen H1; missing {len(bad)} rows.\n'
                           + bad[KEY+['_merge']].head(20).to_string(index=False))
    out = out.drop(columns='_merge')
    if len(out) != len(frozen):
        raise RuntimeError(f'Coverage mismatch: manifest={len(out)} frozen={len(frozen)}')
    return out


def discover_regime_csv(project_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit if explicit.is_absolute() else project_root / explicit
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    candidates = list((project_root/'experiment_results').glob('**/Controlled_Synthetic_regime_assignments.csv'))
    root_candidate = project_root / 'Controlled_Synthetic_regime_assignments.csv'
    if root_candidate.is_file():
        candidates.append(root_candidate)
    # Prefer paths that look final; then newest mtime.
    candidates = list(dict.fromkeys([p.resolve() for p in candidates if p.is_file()]))
    if not candidates:
        raise FileNotFoundError(
            'Could not find Controlled_Synthetic_regime_assignments.csv. '
            'Pass --regime-csv explicitly to the completed Section 5.4 output.'
        )
    candidates.sort(key=lambda p: (('final' in str(p).lower()), p.stat().st_mtime), reverse=True)
    return candidates[0]


def calc_synthetic_dmis(d: pd.DataFrame) -> np.ndarray:
    if 'DMIS' in d.columns:
        return pd.to_numeric(d['DMIS'], errors='raise').to_numpy(float)
    if {'representation_response_raw','environment_distance'} <= set(d.columns):
        rr = pd.to_numeric(d['representation_response_raw'], errors='raise').to_numpy(float)
        dist = pd.to_numeric(d['environment_distance'], errors='raise').to_numpy(float)
        return rr / np.maximum(dist, EPS_D)
    raise RuntimeError('Regime assignment CSV has neither DMIS nor response/distance fields.')


def classify(dmis: np.ndarray, rd: np.ndarray, dthr: np.ndarray | float, rthr: np.ndarray | float) -> np.ndarray:
    dmis = np.asarray(dmis, float); rd = np.asarray(rd, float)
    dh = dmis >= dthr; rh = rd >= rthr
    out = np.empty(len(dmis), dtype=object)
    out[(~dh) & (~rh)] = 'Stable Invariance'
    out[dh & (~rh)] = 'Adaptive Response'
    out[dh & rh] = 'Over-Reactive Instability'
    out[(~dh) & rh] = 'Pathological Rigidity'
    return out


def recover_regime_protocol(reg: pd.DataFrame) -> Tuple[str, Dict[str, Tuple[float,float]], float]:
    """Recover the already-used Section 5.4 q=.50 thresholding rule from saved labels.

    We test the known protocol variants without peeking at Section 5.7 transformed values.
    The first exact match is used.  This prevents silently changing the already-finalized 5.4 labels.
    """
    if 'regime_q050' not in reg.columns or 'RD' not in reg.columns:
        raise RuntimeError('Regime CSV must contain regime_q050 and RD.')
    x = reg.copy()
    x['_dmis'] = calc_synthetic_dmis(x)
    x['_rd'] = pd.to_numeric(x['RD'], errors='raise').astype(float)
    true = x['regime_q050'].astype(str).to_numpy()

    # Candidate 1: global q=.50, exactly matching the finalized 5.4 script currently used.
    dthr=float(np.quantile(x['_dmis'], .5)); rthr=float(np.quantile(x['_rd'], .5))
    pred=classify(x['_dmis'].to_numpy(), x['_rd'].to_numpy(), dthr, rthr)
    acc=float(np.mean(pred == true))
    if acc >= 1.0 - 1e-12:
        return 'global_q050', {'__global__':(dthr,rthr)}, acc

    # Candidate 2: intervention-type-specific q=.50, retained only for compatibility
    # with older manuscript variants.  We accept it only if it exactly reproduces
    # the already-saved labels.
    if 'intervention_type' in x.columns:
        th={}
        pred2=np.empty(len(x),dtype=object)
        for g,ix in x.groupby('intervention_type').groups.items():
            sub=x.loc[ix]
            dt=float(np.quantile(sub['_dmis'],.5)); rt=float(np.quantile(sub['_rd'],.5))
            th[str(g)]=(dt,rt)
            pred2[np.asarray(list(ix),dtype=int)] = classify(sub['_dmis'].to_numpy(),sub['_rd'].to_numpy(),dt,rt)
        acc2=float(np.mean(pred2 == true))
        if acc2 >= 1.0 - 1e-12:
            return 'intervention_type_q050', th, acc2

    raise RuntimeError(
        f'Could not reproduce saved Section 5.4 regime_q050 labels from the known q=.50 rules. '
        f'Global agreement={acc:.6f}. Pass the exact finalized regime-assignment CSV or align the 5.4 protocol first.'
    )


def attach_regime_data(manifest: pd.DataFrame, reg_csv: Path) -> Tuple[pd.DataFrame, str, Dict[str,Tuple[float,float]]]:
    """Attach the frozen Section 5.4 threshold protocol to the frozen H1 cohort.

    Important: the historical Section 5.4 CSV can contain stale repeated exports
    for the same model/intervention key (for example multiple Moirai reruns).
    Those rows can legitimately disagree in RD/DMIS, so choosing one duplicate
    would be arbitrary.  We therefore do NOT map labels by duplicate key.

    Instead we:
      1) recover the exact q=.50 threshold rule that generated the saved 5.4 labels;
      2) verify that every frozen H1 key is represented in the 5.4 source; and
      3) apply those frozen thresholds to the exact frozen H1 DMIS/RD carried by
         the Section 5.6 manifest.

    This preserves the finalized Section 5.4 decision rule while avoiding any
    post-hoc selection among conflicting historical reruns.
    """
    r = pd.read_csv(reg_csv, low_memory=False)

    # Section 5.6 representation pairs are controlled-access Track B only.
    if 'evaluation_track' in r.columns:
        track = (r['evaluation_track'].astype(str).str.strip().str.upper()
                 .str.replace('TRACK', '', regex=False).str.replace('_', '', regex=False))
        mask_b = track.eq('B')
        if not mask_b.any():
            vals = sorted(r['evaluation_track'].dropna().astype(str).unique().tolist())
            raise RuntimeError(
                f'{reg_csv} contains evaluation_track but no Track B rows. Found: {vals}'
            )
        before = len(r)
        r = r.loc[mask_b].copy()
        print(f'[REGIME] selected controlled-access Track B: {len(r)}/{before} rows', flush=True)

    # If a setting column is present, keep only controlled synthetic rows.
    if 'setting' in r.columns:
        ss = r['setting'].astype(str).str.strip().str.lower()
        mask_syn = ss.str.contains('synthetic', regex=False)
        if mask_syn.any():
            r = r.loc[mask_syn].copy()

    missing=set(REGIME_KEY+['RD','regime_q050'])-set(r.columns)
    if missing:
        raise RuntimeError(f'{reg_csv} missing columns {sorted(missing)}')

    # Normalize source columns.
    r['model'] = r['model'].astype(str)
    r['intervention_type'] = r['intervention_type'].astype(str)
    r['paper_intensity_level'] = r['paper_intensity_level'].astype(str)
    r['intervention_seed'] = pd.to_numeric(r['intervention_seed'], errors='raise').astype(int)
    r['RD'] = pd.to_numeric(r['RD'], errors='raise').astype(float)

    # Recover the exact Section 5.4 threshold protocol BEFORE touching duplicates.
    # The saved labels were generated from this complete CSV, so this is the only
    # faithful way to reuse the already-finalized decision boundary.
    mode, thresholds, agreement = recover_regime_protocol(r)
    print(f'[REGIME] {reg_csv}', flush=True)
    print(f'[REGIME] recovered protocol={mode}, saved-label reproduction={agreement:.6f}', flush=True)

    # Audit historical duplicate keys.  Conflicting reruns are reported, but they
    # are no longer used to choose the label for a frozen H1 pair.
    dup_mask = r.duplicated(REGIME_KEY, keep=False)
    duplicate_rows = int(dup_mask.sum())
    duplicate_groups = 0
    conflicting_groups = 0
    if duplicate_rows:
        duplicate_groups = int(r.loc[dup_mask].groupby(REGIME_KEY, dropna=False).ngroups)
        for _, g in r.loc[dup_mask].groupby(REGIME_KEY, dropna=False):
            labels = g['regime_q050'].astype(str).dropna().unique()
            dm = calc_synthetic_dmis(g)
            rd = g['RD'].to_numpy(float)
            dm_span = float(np.nanmax(dm)-np.nanmin(dm)) if len(dm) else 0.0
            rd_span = float(np.nanmax(rd)-np.nanmin(rd)) if len(rd) else 0.0
            if len(labels) > 1 or dm_span > 1e-12 or rd_span > 1e-12:
                conflicting_groups += 1
        print(
            f'[REGIME AUDIT] historical duplicate rows={duplicate_rows}, '
            f'duplicate groups={duplicate_groups}, conflicting groups={conflicting_groups}. '
            'No duplicate row is selected; frozen H1 DMIS/RD are classified directly.',
            flush=True
        )

    # Key coverage only: every frozen H1 weak pair must exist in the 5.4 source,
    # but multiple historical rows for the same key are harmless here.
    source_keys = r[REGIME_KEY].drop_duplicates().copy()
    cov = manifest.merge(source_keys, on=REGIME_KEY, how='left', validate='many_to_one', indicator=True)
    miss = cov[cov['_merge'] != 'both']
    if len(miss):
        raise RuntimeError(
            f'{len(miss)} frozen H1 weak pairs are absent from the Section 5.4 Track-B source.\n'
            + miss[KEY+['_merge']].head(20).to_string(index=False)
        )
    cov = cov.drop(columns='_merge')

    # Classify the exact frozen H1 pair using the recovered 5.4 threshold(s).
    if 'RD' not in cov.columns:
        raise RuntimeError('Frozen H1 RD was not preserved in the Section 5.7 manifest.')
    if 'DMIS_raw_frozen' not in cov.columns:
        raise RuntimeError('Section 5.6 manifest is missing DMIS_raw_frozen.')

    regimes=[]
    for _, row in cov.iterrows():
        dthr, rthr = threshold_for(row, mode, thresholds)
        regimes.append(assign_one(float(row['DMIS_raw_frozen']), float(row['RD']), dthr, rthr))
    cov['regime_q050_frozen'] = regimes
    cov.attrs['regime_source_duplicate_rows'] = duplicate_rows
    cov.attrs['regime_source_duplicate_groups'] = duplicate_groups
    cov.attrs['regime_source_conflicting_groups'] = conflicting_groups
    return cov, mode, thresholds

def threshold_for(row: pd.Series, mode: str, thresholds: Dict[str,Tuple[float,float]]) -> Tuple[float,float]:
    if mode == 'global_q050':
        return thresholds['__global__']
    if mode == 'intervention_type_q050':
        return thresholds[str(row['intervention_type'])]
    raise KeyError(mode)


def assign_one(dmis: float, rd: float, dthr: float, rthr: float) -> str:
    return str(classify(np.asarray([dmis]),np.asarray([rd]),dthr,rthr)[0])


def preflight_representation(row: pd.Series, project_root: Path, representation_response) -> Tuple[Path,np.ndarray,np.ndarray,float,float,float]:
    p=resolve_pair_artifact(row,project_root)
    with np.load(p,allow_pickle=False) as z:
        if {'Z0','Zk'}-set(z.files):
            raise RuntimeError(f'{p} missing Z0/Zk')
        z0=np.asarray(z['Z0'],dtype=np.float64)
        zk=np.asarray(z['Zk'],dtype=np.float64)
    if z0.shape != zk.shape or z0.ndim < 2 or z0.shape[0] < 2:
        raise RuntimeError(f'Invalid aligned representation pair {p}: {z0.shape} vs {zk.shape}')
    n0=centered_norm(z0); nk=centered_norm(zk)
    raw=float(representation_response(z0,zk,alpha=PRIMARY_ALPHA,eps=EPS_Z)[0])
    frozen=float(row['raw_response_frozen'])
    allowed=5e-7 + 1e-4*abs(frozen)
    if not np.isfinite(raw) or abs(raw-frozen)>allowed:
        raise RuntimeError(
            f'Frozen 5.6 pair no longer reproduces H1 raw response: {p}\n'
            f'fresh={raw:.12g}, frozen={frozen:.12g}, absdiff={abs(raw-frozen):.3g}, tol={allowed:.3g}'
        )
    dist=float(row['environment_distance'])
    dm=raw/max(dist,EPS_D)
    fdm=float(row['DMIS_raw_frozen'])
    if abs(dm-fdm) > (5e-7 + 1e-4*abs(fdm)):
        raise RuntimeError(f'DMIS mismatch for {p}: recomputed={dm}, frozen={fdm}')
    return p,z0,zk,n0,nk,dm


def stratified_candidates(df: pd.DataFrame, n: int, seed: int) -> List[int]:
    """Deterministic random order prioritizing coverage of all models/types."""
    rng=np.random.default_rng(seed)
    chosen=[]; used=set()
    # First cover each model x intervention type once when possible.
    for model in sorted(df['model'].astype(str).unique()):
        dm=df[df['model'].astype(str)==model]
        for kind in sorted(dm['intervention_type'].astype(str).unique()):
            inds=dm.index[dm['intervention_type'].astype(str)==kind].to_numpy()
            if len(inds):
                i=int(rng.choice(inds));
                if i not in used: chosen.append(i); used.add(i)
    # Then give each model one extra row; for 21 models and 3 types this yields 84.
    for model in sorted(df['model'].astype(str).unique()):
        inds=np.array([i for i in df.index[df['model'].astype(str)==model] if i not in used],dtype=int)
        if len(inds):
            i=int(rng.choice(inds)); chosen.append(i); used.add(i)
    # Fill from all remaining rows.
    remaining=np.array([i for i in df.index if i not in used],dtype=int)
    rng.shuffle(remaining)
    chosen.extend(map(int,remaining.tolist()))
    return chosen[:max(n,1)] if len(chosen)>=n else chosen


def givens_rotate_pair(z0: np.ndarray, zk: np.ndarray, rng: np.random.Generator, n_givens: int) -> Tuple[np.ndarray,np.ndarray,dict]:
    a=np.asarray(z0,dtype=np.float64).reshape(z0.shape[0],-1).copy()
    b=np.asarray(zk,dtype=np.float64).reshape(zk.shape[0],-1).copy()
    d=a.shape[1]
    if d < 2:
        raise ValueError('Need >=2 features for a rotation audit.')
    ng=min(max(1,int(n_givens)), max(1,d*2))
    angles=[]
    for _ in range(ng):
        i,j=rng.choice(d,size=2,replace=False)
        theta=float(rng.uniform(-math.pi,math.pi))
        c,s=math.cos(theta),math.sin(theta)
        ai=a[:,i].copy(); aj=a[:,j].copy()
        bi=b[:,i].copy(); bj=b[:,j].copy()
        a[:,i]=c*ai-s*aj; a[:,j]=s*ai+c*aj
        b[:,i]=c*bi-s*bj; b[:,j]=s*bi+c*bj
        angles.append(theta)
    return a,b,{'givens':ng,'mean_abs_angle':float(np.mean(np.abs(angles)))}


def permute_pair(z0: np.ndarray, zk: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray,np.ndarray,dict]:
    a=np.asarray(z0,dtype=np.float64).reshape(z0.shape[0],-1)
    b=np.asarray(zk,dtype=np.float64).reshape(zk.shape[0],-1)
    d=a.shape[1]
    perm=rng.permutation(d)
    # Advanced indexing materializes one pair at a time; safe even for high-d models.
    return a[:,perm], b[:,perm], {'features':int(d)}


def scale_pair(z0: np.ndarray, zk: np.ndarray, rng: np.random.Generator,
               scale_min: float, scale_max: float) -> Tuple[np.ndarray,np.ndarray,dict]:
    if not (0 < scale_min < scale_max):
        raise ValueError('Require 0 < scale_min < scale_max.')
    c=float(np.exp(rng.uniform(np.log(scale_min),np.log(scale_max))))
    return np.asarray(z0,dtype=np.float64)*c, np.asarray(zk,dtype=np.float64)*c, {'scale':c}


def rank_agreement(a: Iterable[float], b: Iterable[float]) -> float:
    x=np.asarray(list(a),float); y=np.asarray(list(b),float)
    if len(x)<2: return float('nan')
    r=spearmanr(x,y).statistic
    return float(r) if np.isfinite(r) else float('nan')


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    display={'rotation':'Shared Rotation','permutation':'Feature Permutation','scaling':'Admissible Positive Scaling'}
    for check,g in trials.groupby('check',sort=False):
        dev=pd.to_numeric(g['abs_dmis_deviation'],errors='raise').to_numpy(float)
        rows.append({
            'Check':display.get(check,check),
            'Valid_instances':int(len(g)),
            'Maximum_DMIS_deviation':float(np.max(dev)),
            'Mean_DMIS_deviation':float(np.mean(dev)),
            'P95_DMIS_deviation':float(np.quantile(dev,.95)),
            'Rank_agreement':rank_agreement(g['DMIS_original'],g['DMIS_transformed']),
            'Regime_agreement':float(np.mean(g['regime_original'].astype(str)==g['regime_transformed'].astype(str))),
            'Regime_changes':int(np.sum(g['regime_original'].astype(str)!=g['regime_transformed'].astype(str))),
        })
    return pd.DataFrame(rows)


def fmt_sci(x: float) -> str:
    if x == 0: return '0'
    return f'{x:.3e}'


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    lines=[
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{DMIS invariance sanity checks under shared representation transformations.}',
        r'\label{tab:dmis_invariance}',
        r'\setlength{\tabcolsep}{5pt}',
        r'\renewcommand{\arraystretch}{0.9}',
        r'\footnotesize',
        r'\begin{tabular}{lcc}',
        r'\toprule',
        r'Check & Maximum DMIS Deviation / Rank Agreement & Regime Agreement \\',
        r'\midrule',
    ]
    for _,r in summary.iterrows():
        dev=fmt_sci(float(r['Maximum_DMIS_deviation']))
        rho=float(r['Rank_agreement'])
        agr=100*float(r['Regime_agreement'])
        lines.append(f"{r['Check']} & {dev} / {rho:.6f} & {agr:.1f}\\% \\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def choose_font():
    import matplotlib.font_manager as fm
    try:
        fm.findfont('Times New Roman',fallback_to_default=False)
        return 'Times New Roman'
    except Exception:
        return 'DejaVu Serif'


def save_fig(fig, stem: Path):
    for ext,dpi in [('pdf',None),('svg',None),('png',600)]:
        kw={'bbox_inches':'tight','facecolor':'white'}
        if dpi is not None: kw['dpi']=dpi
        fig.savefig(stem.with_suffix('.'+ext),**kw)


def make_figures(trials: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    font=choose_font()
    plt.rcParams.update({
        'font.family':font,'font.size':12,'axes.labelsize':13,'axes.titlesize':13,
        'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10,
        'pdf.fonttype':42,'ps.fonttype':42,
    })
    order=['rotation','permutation','scaling']
    names={'rotation':'Shared Rotation','permutation':'Feature Permutation','scaling':'Positive Scaling'}

    # Figure 1: deviation distribution + regime agreement.
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.8,4.6),gridspec_kw={'width_ratios':[1.35,1.0]})
    rng=np.random.default_rng(2027)
    floor=1e-18
    for xi,k in enumerate(order):
        g=trials[trials['check']==k]
        y=np.maximum(pd.to_numeric(g['abs_dmis_deviation']).to_numpy(float),floor)
        x=xi+rng.uniform(-0.16,0.16,size=len(y))
        ax1.scatter(x,y,s=18,alpha=.58,edgecolors='none')
        mx=float(np.max(pd.to_numeric(g['abs_dmis_deviation']).to_numpy(float)))
        ax1.scatter([xi],[max(mx,floor)],s=72,marker='D',edgecolors='black',linewidths=.6,zorder=4)
        ax1.text(xi,max(mx,floor)*1.35 if mx>0 else floor*1.35,fmt_sci(mx),ha='center',va='bottom',fontsize=9)
    ax1.set_yscale('log')
    ax1.set_xticks(range(3),[names[k] for k in order],rotation=12,ha='right')
    ax1.set_ylabel('Absolute DMIS Deviation')
    ax1.set_title('(a) Numerical Invariance')
    ax1.grid(axis='y',alpha=.22,linewidth=.7)
    ax1.set_axisbelow(True)

    sidx=summary.set_index('Check')
    vals=[]; rhos=[]
    check_names=[names[k] if k!='scaling' else 'Admissible Positive Scaling' for k in order]
    # map plotting names to summary names
    sum_names=['Shared Rotation','Feature Permutation','Admissible Positive Scaling']
    for nm in sum_names:
        vals.append(100*float(sidx.loc[nm,'Regime_agreement']))
        rhos.append(float(sidx.loc[nm,'Rank_agreement']))
    xx=np.arange(3)
    bars=ax2.bar(xx,vals,width=.62,edgecolor='black',linewidth=.6)
    ax2.set_ylim(0,105)
    ax2.set_ylabel('Regime Agreement (%)')
    ax2.set_xticks(xx,[names[k] for k in order],rotation=12,ha='right')
    ax2.set_title('(b) Operational-Label Stability')
    ax2.grid(axis='y',alpha=.22,linewidth=.7)
    ax2.set_axisbelow(True)
    for b,v,rho in zip(bars,vals,rhos):
        ax2.text(b.get_x()+b.get_width()/2,min(v+1.3,103.5),f'{v:.1f}%\n$\\rho$={rho:.3f}',ha='center',va='bottom',fontsize=9)
    fig.subplots_adjust(left=.08,right=.985,bottom=.22,top=.88,wspace=.30)
    save_fig(fig,out/'Figure5_7_invariance_audit')
    plt.close(fig)

    # Figure 2: original vs transformed DMIS identity plots.
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.1))
    for ax,k in zip(axes,order):
        g=trials[trials['check']==k]
        x=pd.to_numeric(g['DMIS_original']).to_numpy(float)
        y=pd.to_numeric(g['DMIS_transformed']).to_numpy(float)
        lo=float(min(x.min(),y.min())); hi=float(max(x.max(),y.max()))
        pad=max((hi-lo)*.04,1e-9)
        ax.scatter(x,y,s=24,alpha=.68,edgecolors='none')
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],linestyle='--',linewidth=1.0)
        ax.set_xlim(lo-pad,hi+pad); ax.set_ylim(lo-pad,hi+pad)
        ax.set_xlabel('Original DMIS')
        ax.set_ylabel('Transformed DMIS')
        rho=rank_agreement(x,y)
        mx=float(np.max(np.abs(x-y)))
        ax.set_title(f'{names[k]}\n$\\rho$={rho:.3f}, max $|\\Delta|$={fmt_sci(mx)}')
        ax.grid(alpha=.20,linewidth=.7)
    fig.subplots_adjust(left=.065,right=.99,bottom=.18,top=.82,wspace=.30)
    save_fig(fig,out/'Figure5_7_dmis_identity')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=PROJECT_DEFAULT)
    ap.add_argument('--regime-csv',type=Path,default=None,help='Completed Section 5.4 Controlled_Synthetic_regime_assignments.csv')
    ap.add_argument('--output-dir',type=Path,default=None)
    ap.add_argument('--trials',type=int,default=100,help='Valid random instances per transformation; paper primary = 100')
    ap.add_argument('--random-seed',type=int,default=5700)
    ap.add_argument('--givens',type=int,default=64,help='Number of random Givens planes per shared rotation')
    ap.add_argument('--scale-min',type=float,default=0.25)
    ap.add_argument('--scale-max',type=float,default=4.0)
    ap.add_argument('--no-plots',action='store_true')
    a=ap.parse_args()
    if a.trials != 100:
        print(f'[WARNING] Paper protocol specifies 100 valid instances; requested {a.trials}.',flush=True)
    project=a.project_root.resolve()
    unified=_add_unified_to_path(project)
    from causalgap.metrics import representation_response

    out=(a.output_dir if a.output_dir is not None else project/OUT_REL)
    if not out.is_absolute(): out=project/out
    out.mkdir(parents=True,exist_ok=True)

    frozen=load_frozen_weak(project)
    manifest=load_metric_manifest(project,frozen)
    regime_csv=discover_regime_csv(project,a.regime_csv)
    data,regime_mode,thresholds=attach_regime_data(manifest,regime_csv)
    print(f'[H1] frozen weak rows={len(frozen)}, models={frozen.model.nunique()}, families={frozen.model_family.nunique() if "model_family" in frozen else "NA"}',flush=True)
    print(f'[5.6] audited representation rows={len(data)}',flush=True)

    # Candidate order prioritizes all model/intervention-type coverage, then random fill.
    candidate_order=stratified_candidates(data,a.trials*3,a.random_seed)
    if len(candidate_order)<a.trials:
        raise RuntimeError(f'Not enough representation pairs for {a.trials} trials.')

    selected=[]; cache={}; rejected=[]
    for idx in candidate_order:
        row=data.loc[idx]
        try:
            p,z0,zk,n0,nk,dm=preflight_representation(row,project,representation_response)
        except Exception as e:
            rejected.append({'index':int(idx),'model':str(row['model']),'reason':str(e)})
            continue
        if n0 <= EPS_Z or nk <= EPS_Z:
            rejected.append({'index':int(idx),'model':str(row['model']),'reason':f'degenerate centered norm {n0},{nk}'})
            continue
        cache[int(idx)]=(p,z0,zk,n0,nk,dm)
        selected.append(int(idx))
        if len(selected)>=a.trials: break
    if len(selected)<a.trials:
        raise RuntimeError(f'Only {len(selected)} nondegenerate audited pairs available; need {a.trials}.')

    sel=data.loc[selected].copy().reset_index(drop=False).rename(columns={'index':'manifest_index'})
    sel['artifact_path_resolved']=[str(cache[int(i)][0]) for i in selected]
    sel['centered_norm_Z0']=[cache[int(i)][3] for i in selected]
    sel['centered_norm_Zk']=[cache[int(i)][4] for i in selected]
    sel.to_csv(out/'00_selected_pairs.csv',index=False)
    if rejected:
        pd.DataFrame(rejected).to_csv(out/'00_rejected_candidates.csv',index=False)

    print(f'[SELECT] {len(sel)} nondegenerate pairs; models={sel.model.nunique()}, intervention_types={sel.intervention_type.nunique()}',flush=True)

    trial_rows=[]
    for pos,row in sel.iterrows():
        idx=int(row['manifest_index'])
        p,z0,zk,n0,nk,dm0=cache[idx]
        rd=float(row['RD'])
        dthr,rthr=threshold_for(row,regime_mode,thresholds)
        original_regime=str(row['regime_q050_frozen'])
        # Verify that the exact frozen H1 DMIS/RD pair reproduces the regime
        # obtained from the recovered Section 5.4 threshold protocol.
        audit_regime=assign_one(dm0,rd,dthr,rthr)
        if audit_regime != original_regime:
            raise RuntimeError(
                f'Regime protocol mismatch on selected row {tuple_key(row)}: '
                f'frozen={original_regime}, recomputed={audit_regime}'
            )

        for check in ('rotation','permutation','scaling'):
            rng=np.random.default_rng(stable_seed(a.random_seed,*tuple_key(row),check))
            if check=='rotation':
                ta,tb,meta=givens_rotate_pair(z0,zk,rng,a.givens)
            elif check=='permutation':
                ta,tb,meta=permute_pair(z0,zk,rng)
            else:
                ta,tb,meta=scale_pair(z0,zk,rng,a.scale_min,a.scale_max)
                # Admissibility required by the paper: stabilizer inactive before/after scaling.
                if centered_norm(ta) <= EPS_Z or centered_norm(tb) <= EPS_Z:
                    raise RuntimeError(f'Generated inadmissible scaling for {tuple_key(row)}: {meta}')
            raw_t=float(representation_response(ta,tb,alpha=PRIMARY_ALPHA,eps=EPS_Z)[0])
            dmis_t=raw_t/max(float(row['environment_distance']),EPS_D)
            reg_t=assign_one(dmis_t,rd,dthr,rthr)
            rec={
                **{k:row[k] for k in KEY},
                'check':check,
                'trial_index':int(pos),
                'artifact_path':str(p),
                'DMIS_original':float(dm0),
                'DMIS_transformed':float(dmis_t),
                'signed_dmis_deviation':float(dmis_t-dm0),
                'abs_dmis_deviation':float(abs(dmis_t-dm0)),
                'raw_response_original':float(row['raw_response_frozen']),
                'raw_response_transformed':float(raw_t),
                'RD':rd,
                'dmis_threshold':float(dthr),
                'rd_threshold':float(rthr),
                'regime_original':original_regime,
                'regime_transformed':reg_t,
                'regime_unchanged':bool(reg_t==original_regime),
                'centered_norm_Z0':float(n0),
                'centered_norm_Zk':float(nk),
                'alpha':PRIMARY_ALPHA,
                'eps_Z':EPS_Z,
                'eps_D':EPS_D,
                'transform_metadata':json.dumps(meta,sort_keys=True),
            }
            trial_rows.append(rec)

    trials=pd.DataFrame(trial_rows)
    trials.to_csv(out/'01_invariance_trials.csv',index=False)
    summary=summarize(trials)
    summary.to_csv(out/'Table_A7_6_invariance.csv',index=False)
    write_latex_table(summary,out/'Table_A7_6_invariance.tex')

    audit={
        'protocol':'CausalGap Section 5.7 invariance sanity checks',
        'paper_trials_per_transformation':100,
        'actual_trials_per_transformation':int(a.trials),
        'random_seed':int(a.random_seed),
        'alpha':PRIMARY_ALPHA,
        'eps_Z':EPS_Z,
        'eps_D':EPS_D,
        'shared_rotation':f'composition of {a.givens} random Givens rotations in feature space',
        'feature_permutation':'one common full random feature permutation applied to Z0 and Zk',
        'positive_scaling':f'one common log-uniform scalar c in [{a.scale_min},{a.scale_max}] with stabilizer-inactive admissibility check',
        'source_representation_root':str(project/REP_REL),
        'frozen_h1_weak_table':str(project/H1_WEAK_REL),
        'regime_assignment_csv':str(regime_csv),
        'regime_threshold_protocol':regime_mode,
        'regime_thresholds':{k:[float(v[0]),float(v[1])] for k,v in thresholds.items()},
        'regime_source_duplicate_rows':int(data.attrs.get('regime_source_duplicate_rows',0)),
        'regime_source_duplicate_groups':int(data.attrs.get('regime_source_duplicate_groups',0)),
        'regime_source_conflicting_groups':int(data.attrs.get('regime_source_conflicting_groups',0)),
        'selected_pairs':int(len(sel)),
        'selected_models':int(sel['model'].nunique()),
        'selected_intervention_types':sorted(sel['intervention_type'].astype(str).unique().tolist()),
        'rejected_candidate_count':int(len(rejected)),
        'summary':summary.to_dict('records'),
        'degenerate_policy':'centered pairs with norm <= eps_Z are excluded from primary invariance claims; scaling must remain above eps_Z after transformation',
    }
    (out/'02_invariance_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')

    if not a.no_plots:
        make_figures(trials,summary,out)

    print('\n================ Section 5.7 Summary ================',flush=True)
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.9g}'),flush=True)
    print('\n[SAVED]',out/'Table_A7_6_invariance.csv',flush=True)
    print('[SAVED]',out/'Table_A7_6_invariance.tex',flush=True)
    print('[SAVED]',out/'01_invariance_trials.csv',flush=True)
    print('[SAVED]',out/'02_invariance_audit.json',flush=True)
    if not a.no_plots:
        print('[SAVED]',out/'Figure5_7_invariance_audit.[pdf|svg|png]',flush=True)
        print('[SAVED]',out/'Figure5_7_dmis_identity.[pdf|svg|png]',flush=True)


if __name__=='__main__':
    main()
