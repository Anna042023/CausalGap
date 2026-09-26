#!/usr/bin/env python3
"""Section 5.6: *audited* replay of frozen H1 V1 pairs. No training or H1 mutation.

Corrections over prior V2 exporter:
 - Replay Zk via the SAME evaluator._collect as original H1 (full intervention
   x/y/mask and selective representation rows); then Z0 via _collect_representations.
 - Read original H1 saved intervention.z and rep_local_indices; check Zk *itself*,
   not only the scalar response. Report worker/DSAI/code drift if different.
 - Optionally reproduce original full baseline representation before replaying
   weak pairs; on Moirai this is automatic because the previous replay was invalid.
 - Reject substantive mismatch, while allowing only tiny audited GPU replay noise (relative-L2 default 2e-5).
 - Audit previously saved 5.6 files on resume, to prevent stale rows.
 - Moirai fresh-worker mode bypasses content-addressed foundation cache, with
   no cache deletion or H1 mutation. Historical hook drift still fails safely.

This cannot regenerate an original missing Z0 from a single frozen scalar or an
original Zk; a mismatching worker/cache/DSAI version must be restored or audited.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
UNIFIED = HERE.parents[1]
V2_SRC = UNIFIED / 'experiments/incremental_value'
for p in (str(V2_SRC), str(UNIFIED), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
from response_metrics import compute_measures

PROJECT = Path('/data/wanganna/CausalGap')
V1_REL = 'experiment_results/incremental_value'
OUT_REL = 'experiment_results/representation_robustness'
KEY = ('intervention_type', 'paper_intensity_level', 'intervention_seed')


def must(path: Path, label='file') -> Path:
    if not path.is_file():
        raise FileNotFoundError(f'Missing {label}: {path}')
    return path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def yes(x):
    return str(x).strip().lower() in ('true', '1', 'yes')


def json_atomic(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    os.replace(tmp, path)


def json_safe(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else str(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (Path,)):
        return str(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def diff(x, y):
    x, y = np.asarray(x), np.asarray(y)
    d = {'shape_a': list(x.shape), 'shape_b': list(y.shape), 'same_shape': x.shape == y.shape}
    if x.shape != y.shape:
        return d
    xa, ya = x.astype(np.float64), y.astype(np.float64)
    delta = xa - ya
    d.update(exact_equal=bool(np.array_equal(x, y)),
             norm_a=float(np.linalg.norm(xa)), norm_b=float(np.linalg.norm(ya)),
             diff_l2=float(np.linalg.norm(delta)),
             diff_max=float(np.max(np.abs(delta))) if delta.size else 0.0,
             relative_l2=float(np.linalg.norm(delta) / max(float(np.linalg.norm(xa)), 1e-12)))
    return d


def check_close(a, b, atol, rtol):
    return np.asarray(a).shape == np.asarray(b).shape and bool(
        np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False))


def representation_replay_ok(a, b, atol, rtol, rel_l2_tol):
    """Numerical replay audit for a frozen representation tensor.

    Deep-learning GPU kernels can produce tiny elementwise differences across
    executions even when the representation and downstream response are
    numerically the same.  We therefore accept either ordinary elementwise
    allclose OR a very small global relative-L2 discrepancy.  Shape and
    finiteness are always required.  This is an audit tolerance only; it never
    changes the saved frozen H1 result or the response metric.
    """
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape:
        return False
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return False
    if np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=False):
        return True
    xa, ya = x.astype(np.float64), y.astype(np.float64)
    rel = float(np.linalg.norm(xa - ya) / max(float(np.linalg.norm(xa)), 1e-12))
    return rel <= float(rel_l2_tol)


def preflight(a):
    from protocol import H1_LEVELS, MODEL_FAMILIES, FOUNDATION_MODELS
    from causalgap import evaluator
    for name in ('_collect', '_collect_representations', 'uniform_sample_indices'):
        if not hasattr(evaluator, name):
            raise RuntimeError(f'Missing installed causalgap.evaluator.{name}; no guesswork permitted')
    ps = inspect.signature(evaluator._collect)
    if not {'rep_indices','max_rep_samples','y_mask_arr'}.issubset(ps.parameters):
        raise RuntimeError(f'Original _collect signature unexpectedly changed: {ps}')
    v1 = a.project_root / V1_REL
    run = v1 / 'runs/synthetic' / a.model / 'trackB' / f'seed{a.model_seed}'
    cfg = json.loads(must(run / 'h1_job_config.json', 'frozen job config').read_text())
    if (str(cfg['model']) != a.model or int(cfg['model_seed']) != a.model_seed or
        cfg['setting'] != 'synthetic' or cfg.get('profile') != 'paper' or
        cfg.get('evaluation_track') != 'B'):
        raise RuntimeError(f'Frozen job config mismatch: {cfg}')
    df = pd.read_csv(must(run / 'causalgap_results.csv', 'frozen H1 results'))
    fields = {'intervention_type','paper_intensity_level','intervention_seed',
              'representation_response_raw','environment_distance','DMIS',
              'causal_valid','leakage_detected','artifact'}
    if fields - set(df.columns):
        raise RuntimeError(f'Missing frozen fields: {fields - set(df.columns)}')
    df = df[df.intervention_type.isin(H1_LEVELS)].copy()
    df = df[df.apply(lambda r: next((v['phase'] for v in H1_LEVELS[str(r.intervention_type)]
               if v['level'] == str(r.paper_intensity_level)), None) == 'weak', axis=1)]
    if df.empty or df.duplicated(list(KEY)).any():
        raise RuntimeError('Missing or duplicate weak rows')
    if not df.causal_valid.map(yes).all() or df.leakage_detected.map(yes).any():
        raise RuntimeError('Invalid or leaking original H1 weak row')
    em = pd.read_csv(must(v1 / 'environments/synthetic/manifest.csv'))
    em = em.rename(columns={'level':'paper_intensity_level','seed':'intervention_seed'})
    needed = {'intervention_type','paper_intensity_level','intervention_seed','path'}
    if needed - set(em.columns):
        raise RuntimeError(f'Manifest missing columns: {needed - set(em.columns)}')
    df = df.merge(em[list(needed)], on=list(KEY), validate='one_to_one')
    if a.seeds is not None:
        df = df[df.intervention_seed.astype(int).isin(set(a.seeds))]
        if df.empty:
            raise RuntimeError('No requested intervention seeds')
    if a.preflight_full:
        actual = {(str(t),str(l),int(s)) for t,l,s in df[list(KEY)].itertuples(index=False,name=None)}
        seeds = set(df.intervention_seed.astype(int))
        expected = {(kind, spec['level'], seed) for kind, plan in H1_LEVELS.items()
                    for spec in plan if spec['phase']=='weak' for seed in seeds}
        if actual != expected:
            raise RuntimeError(f'Incomplete H1 weak grid: missing={sorted(expected-actual)[:10]}, extra={sorted(actual-expected)[:10]}')
    for p in df.path:
        p = Path(str(p)); must(p if p.is_absolute() else v1 / p, 'source intervention')
    for p in df.artifact:
        p = Path(str(p)); must(p if p.is_absolute() else run / p, 'original H1 intervention NPZ')
    base = must(run / 'representations_original.npz')
    with np.load(base, allow_pickle=False) as z:
        if {'z','rep_indices'} - set(z.files):
            raise RuntimeError(f'Original baseline artifact missing z/rep_indices: {base}')
        original_base = np.asarray(z['z'], dtype=np.float32)
        base_idx = np.asarray(z['rep_indices'], dtype=np.int64)
    if a.model not in MODEL_FAMILIES:
        raise RuntimeError(f'Not in frozen 21-model registry: {a.model}')
    if a.model not in FOUNDATION_MODELS:
        must(run / 'checkpoint.pt', 'frozen non-foundation checkpoint')
    return v1, run, cfg, df, original_base, base_idx


def make_model(a, cfg, run, v1):
    from evaluate_h1_job import synthetic_bundle, add_model_args
    from protocol import FOUNDATION_MODELS
    from causalgap.models.factory import build_model
    from causalgap.evaluator import load_checkpoint
    from causalgap.profiles import resolve as resolve_profile
    from causalgap.utils import seed_everything, device_from_arg
    ap = argparse.ArgumentParser(add_help=False)
    add_model_args(ap)
    original = ap.parse_args(['--model', a.model, '--setting', 'synthetic'])
    original.project_root = a.project_root
    original.experiment_root = v1
    original.model_seed = original.seed = a.model_seed
    original.track, original.dataset = 'B', 'SYNTHETIC-H1'
    original.profile = cfg['profile']
    original.alpha, original.max_rep_samples = float(cfg['alpha']), int(cfg['max_rep_samples'])
    # Frozen H1 used cached foundation inference.  A cache hit proves only that
    # the CURRENT cache returns a result, not that it is the original H1 tensor.
    # In particular, the Moirai worker/hook may have changed without invalidating
    # v3.2 cache keys, which do not contain the worker source hash.
    policy = a.foundation_cache_policy
    if policy == 'auto':
        policy = 'fresh' if a.model == 'Moirai' else 'cached'
    original.foundation_cache = (policy == 'cached')
    original.foundation_cache_readonly = True  # no modifications to old cache
    original.foundation_cache_dir = a.project_root / 'cache/foundation_inference'
    print(f'[FOUNDATION REPLAY] {a.model} policy={policy} '
          f'cache_enabled={original.foundation_cache} cache_readonly=True', flush=True)
    os.environ.setdefault('CAUSALGAP_ROOT', str(a.project_root))
    seed_everything(int(a.model_seed))
    device = device_from_arg(a.device)
    bundle = synthetic_bundle(v1)
    original = resolve_profile(original, original.model,
                               history_len=bundle.history_len, num_nodes=bundle.num_nodes)
    model = build_model(a.model, bundle, original)
    if a.model not in FOUNDATION_MODELS and not bool(getattr(model, 'is_zero_shot', False)):
        model = load_checkpoint(model.to(device), run / 'checkpoint.pt', device,
                                bundle=bundle, allow_legacy=False)
    else:
        model = model.to(device)
    model.eval()
    return model, bundle, original, device


def original_npz(run, row):
    p = Path(str(row['artifact']))
    p = p if p.is_absolute() else run / p
    with np.load(must(p, 'frozen per-intervention artifact'), allow_pickle=False) as z:
        if {'z', 'rep_local_indices'} - set(z.files):
            raise RuntimeError(f'Original H1 intervention missing z/rep_local_indices: {p}')
        zk = np.asarray(z['z'], dtype=np.float32)
        ix = np.asarray(z['rep_local_indices'], dtype=np.int64)
        target_idx = (np.asarray(z['rep_target_indices'], dtype=np.int64)
                      if 'rep_target_indices' in z.files else None)
    return p, zk, ix, target_idx


def diagnostic(a, output, payload, cause):
    p = output / 'diagnostics' / f'{a.model}_seed{a.model_seed}_replay_mismatch.json'
    json_atomic(p, json_safe(payload))
    raise RuntimeError(f'{cause}\nDiagnostic saved: {p}\n'
                       'Do not loosen tolerance or mix frozen H1 Zk with a mismatching new Z0.')


def baseline_audit(a, model, bundle, original, device, original_base, base_idx, output, run):
    from causalgap import evaluator
    # Original evaluate_original -> _collect(all baseline windows), not rep-only.
    ymask = np.asarray(bundle.test_y_mask, dtype=bool)
    _, _, fresh, got_ix, _ = evaluator._collect(
        model, bundle.test_x, bundle.test_y, bundle.adj, bundle, device,
        original.batch_size, rep_indices=base_idx,
        max_rep_samples=int(original.max_rep_samples), y_mask_arr=ymask)
    d = diff(original_base, fresh)
    print(f'[BASELINE AUDIT] {a.model}: original norm={d.get("norm_a")} '
          f'fresh norm={d.get("norm_b")} relative_l2={d.get("relative_l2")}', flush=True)
    if not np.array_equal(got_ix, base_idx) or not representation_replay_ok(
            original_base, fresh, a.rep_atol, a.rep_rtol, a.rep_rel_l2):
        diagnostic(a, output, {
            'failure':'Original baseline cannot be reproduced',
            'foundation_cache_policy': a.foundation_cache_policy,
            'frozen_run':str(run),
            'frozen_baseline_npz':str(run/'representations_original.npz'),
            'baseline_diff':d,
            'original_indices':base_idx.tolist(),
            'fresh_indices':np.asarray(got_ix).tolist(),
            'core_evaluator':str(Path(evaluator.__file__).resolve()),
            'core_evaluator_sha256':digest(Path(evaluator.__file__).resolve()),
            'foundation_cache_stats':model.cache_stats() if hasattr(model,'cache_stats') else {},
            'foundation_worker_sha256': (
                digest(Path(model.base.worker))
                if hasattr(model,'base') and hasattr(model.base,'worker')
                and Path(model.base.worker).is_file() else None),
            'hint':'Original and current representation implementations, worker weights/cache, or DSAI settings differ. Restoring a verified original run is required.'
        }, 'Frozen H1 baseline representation changed BEFORE weak replay')


def export(a):
    v1, run, cfg, df, original_base, base_idx = preflight(a)
    print(f'[PREFLIGHT] {a.model} seed{a.model_seed}: {len(df)} weak rows; frozen H1 untouched', flush=True)
    if a.stage == 'preflight':
        return
    from protocol import MODEL_FAMILIES
    from causalgap import evaluator
    from causalgap.interventions import load_external_intervention
    from causalgap.metrics import representation_response
    model, bundle, original, device = make_model(a, cfg, run, v1)
    output = a.project_root / OUT_REL / 'representation_pairs' / a.model / 'trackB' / f'seed{a.model_seed}'
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / 'weak_representation_metrics.csv'
    prior = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    recorded = {(str(r['intervention_type']), str(r['paper_intensity_level']), int(r['intervention_seed'])):r
                for r in prior.to_dict('records')}
    try:
        if a.audit_baseline or a.model == 'Moirai':
            baseline_audit(a, model, bundle, original, device, original_base, base_idx, output, run)
        for row in df.sort_values(list(KEY)).to_dict('records'):
            kind, level, seed = str(row['intervention_type']), str(row['paper_intensity_level']), int(row['intervention_seed'])
            key = (kind, level, seed)
            safe_level = level.replace('=', '_').replace('/', '_')
            pairpath = output / kind / safe_level / f'eval_seed_{seed:04d}.npz'
            frozen_path, original_zk, original_ix, original_target_ix = original_npz(run, row)
            if key in recorded and pairpath.exists() and not a.force:
                old = recorded[key]
                with np.load(pairpath, allow_pickle=False) as saved:
                    if {'Z0','Zk','rep_indices'} - set(saved.files):
                        raise RuntimeError(f'Corrupt existing 5.6 pair: {pairpath}')
                    saved_zk = np.asarray(saved['Zk'], dtype=np.float32)
                    d_zk = diff(original_zk, saved_zk)
                    ix_ok = np.array_equal(original_ix, saved['rep_indices'])
                if abs(float(old['raw_response_frozen']) - float(row['representation_response_raw'])) > 1e-9:
                    raise RuntimeError(f'Frozen H1 row changed: {key}')
                if not ix_ok or not representation_replay_ok(
                        original_zk, saved_zk, a.rep_atol, a.rep_rtol, a.rep_rel_l2):
                    diagnostic(a, output, {'failure':'Previously exported 5.6 Zk differs from frozen original',
                       'key':key,'original_npz':str(frozen_path), 'exported_npz':str(pairpath),
                       'zk_diff':d_zk}, 'Prior 5.6 file did not preserve original H1 representation')
                print(f'[SKIP AUDITED] {a.model} {key}', flush=True)
                continue
            source = Path(str(row['path'])); source = source if source.is_absolute() else v1 / source
            intervention = load_external_intervention(source, bundle, kind=kind)
            n = len(intervention.x)
            if len(original_ix) == 0 or np.min(original_ix) < 0 or np.max(original_ix) >= n:
                diagnostic(a, output, {'key':key, 'original_ix':original_ix.tolist(), 'n':n},
                           'Frozen representation indices out of bounds')
            if original_target_ix is not None and intervention.baseline_indices is not None:
                expected = np.asarray(intervention.baseline_indices, dtype=np.int64)[original_ix]
                if not np.array_equal(expected, original_target_ix):
                    diagnostic(a, output, {'key':key,'original_target_ix':original_target_ix.tolist(),
                       'expected_target_ix':expected.tolist()}, 'Intervention baseline alignment changed')
            if str(intervention.kind) != kind:
                raise RuntimeError(f'Intervention kind mismatch: {intervention.kind} != {kind}')
            # EXACT H1 order: (1) _collect full intervention -> Zk selective,
            # (2) _collect_representations on SAME Xk at baseline adjacency -> Z0.
            if getattr(intervention, 'y_mask', None) is not None:
                mask = np.asarray(intervention.y_mask, dtype=bool)
            else:
                mask = bundle.infer_external_target_mask(bundle.scaler.inverse(intervention.y))
            _, _, fresh_zk, fresh_ix, _ = evaluator._collect(
                model, intervention.x, intervention.y, intervention.adj, bundle, device,
                original.batch_size, rep_indices=original_ix,
                max_rep_samples=int(cfg['max_rep_samples']), y_mask_arr=mask)
            z0 = evaluator._collect_representations(
                model, np.asarray(intervention.x)[original_ix], bundle.adj,
                bundle, device, original.batch_size)
            z0, fresh_zk = np.asarray(z0, dtype=np.float32), np.asarray(fresh_zk, dtype=np.float32)
            if not np.array_equal(fresh_ix, original_ix) or z0.shape != original_zk.shape:
                diagnostic(a, output, {'key':key,'original_ix':original_ix.tolist(),
                  'fresh_ix':np.asarray(fresh_ix).tolist(), 'z0_shape':z0.shape,
                  'frozen_zk_shape':original_zk.shape}, 'Replay sample/shape alignment failed')
            raw = float(representation_response(z0, fresh_zk, alpha=float(cfg['alpha']))[0])
            frozen = float(row['representation_response_raw'])
            abs_err = abs(raw - frozen)
            zk_diff = diff(original_zk, fresh_zk)
            raw_allowed = a.audit_atol + a.audit_rtol * abs(frozen)
            rep_ok = representation_replay_ok(
                original_zk, fresh_zk, a.rep_atol, a.rep_rtol, a.rep_rel_l2)
            if not rep_ok or not np.isfinite(raw) or abs_err > raw_allowed:
                cache_info = model.cache_stats() if hasattr(model, 'cache_stats') else {}
                dsai_key = None
                if hasattr(model, '_adj_key'):
                    k0, kk = model._adj_key(bundle.adj), model._adj_key(intervention.adj)
                    dsai_key = {'equal':k0 == kk,'A0_key':str(k0),'Ak_key':str(kk)}
                diagnostic(a, output, {
                    'failure':'Frozen representation does not match current core replay',
                    'key':key, 'frozen_artifact':str(frozen_path), 'source':str(source),
                    'frozen_zk_vs_fresh_zk':zk_diff,
                    'z0_shape':list(z0.shape), 'z0_norm':float(np.linalg.norm(z0.astype(np.float64))),
                    'frozen_raw_response':frozen,'fresh_raw_response':raw,
                    'raw_absdiff':abs_err,'raw_tolerance':raw_allowed,
                    'intervention_input_shape':list(intervention.x.shape),
                    'intervention_adj_diff':diff(bundle.adj, intervention.adj),
                    'dsai_cache_key':dsai_key, 'cache_stats':cache_info,
                    'core_evaluator':str(Path(evaluator.__file__).resolve()),
                    'core_evaluator_sha256':digest(Path(evaluator.__file__).resolve()),
                    'hint':'Replay failed the numerical representation/response audit. Inspect model checkpoint, evaluator/DSAI path, foundation worker/cache (when applicable), or representation hook.'
                }, 'Current replay does not reproduce the frozen H1 pair within the audited numerical tolerance')
            measures = compute_measures(z0, fresh_zk)
            if any(not np.isfinite(measures[k]) for k in
                   ('resp_1mcka','resp_1msvcca','resp_rsa_distance','resp_1mpwcca')):
                raise RuntimeError(f'Undefined alternative metric: {a.model} {key}: {measures}')
            pairpath.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=pairpath.parent,suffix='.npz',delete=False) as tmp:
                temppath = Path(tmp.name)
            try:
                np.savez_compressed(temppath, Z0=z0, Zk=fresh_zk, rep_indices=original_ix)
                os.replace(temppath, pairpath)
            finally:
                temppath.unlink(missing_ok=True)
            rec = dict(setting='synthetic',model=a.model,
                model_family=int(MODEL_FAMILIES[a.model]),model_seed=a.model_seed,
                intervention_type=kind,paper_intensity_level=level,intervention_seed=seed,
                phase='weak',n_rep_rows=len(original_ix),raw_response_replay=raw,
                raw_response_frozen=frozen,raw_response_absdiff=abs_err,
                environment_distance=float(row['environment_distance']),
                DMIS_raw_frozen=float(row['DMIS']),
                z0_path=f'{pairpath}::Z0',zk_path=f'{pairpath}::Zk',
                artifact_path=str(pairpath), frozen_zk_diff_max=float(zk_diff['diff_max']),
                frozen_zk_relative_l2=float(zk_diff['relative_l2']),
                **measures)
            recorded[key] = rec
            tbl = pd.DataFrame(list(recorded.values())).sort_values(list(KEY))
            tmpcsv = metrics_path.with_suffix('.tmp')
            tbl.to_csv(tmpcsv,index=False)
            os.replace(tmpcsv,metrics_path)
            print(f'[PASS] {a.model} {key}, frozen-Zk relL2={zk_diff["relative_l2"]:.3g}, '
                  f'maxdiff={zk_diff["diff_max"]:.3g}, raw-response absdiff={abs_err:.3g}',flush=True)
    finally:
        if hasattr(model,'close'):
            try:
                model.close()
            except Exception:
                pass
    print(f'[SAVED] {metrics_path}',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=PROJECT)
    p.add_argument('--model',required=True)
    p.add_argument('--model-seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--stage',choices=['preflight','export'],default='preflight')
    p.add_argument('--seeds',nargs='*',type=int,default=None)
    p.add_argument('--preflight-full',action='store_true',
                   help='Require all three weak levels for every intervention family and observed evaluation seed')
    p.add_argument('--force',action='store_true',help='Only overwrite 5.6 derivatives; NEVER H1 inputs')
    p.add_argument('--foundation-cache-policy',choices=['auto','cached','fresh'],default='auto',
                   help='auto bypasses foundation cache for Moirai; fresh never deletes/writes old cache')
    p.add_argument('--audit-baseline',action='store_true',
                   help='Replay full frozen H1 baseline; automatically enabled for Moirai')
    p.add_argument('--audit-atol',type=float,default=1e-5)
    p.add_argument('--audit-rtol',type=float,default=1e-3)
    p.add_argument('--rep-atol',type=float,default=1e-5)
    p.add_argument('--rep-rtol',type=float,default=1e-4)
    p.add_argument('--rep-rel-l2',type=float,default=2e-5,
                   help='Audit-only maximum relative-L2 replay discrepancy when elementwise allclose fails')
    a = p.parse_args()
    if min(a.audit_atol,a.audit_rtol,a.rep_atol,a.rep_rtol,a.rep_rel_l2) < 0:
        p.error('Audit tolerances must be nonnegative')
    export(a)

if __name__ == '__main__':
    main()
