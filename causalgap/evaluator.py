from pathlib import Path
import csv
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import TrafficTensorDataset
from .metrics import prediction_metrics, dmis_graph_formula, dmis_generalized, relative_degradation, mean_std_se, masked_std, METRIC_PROTOCOL
from .utils import ensure_dir, save_json
from .interventions import GRAPH_INTERVENTIONS


def load_checkpoint(model, path, device, bundle=None, allow_legacy=False):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if bundle is not None:
        sig = ckpt.get('data_signature') if isinstance(ckpt, dict) else None
        if sig is None and not allow_legacy:
            raise RuntimeError(
                f'Checkpoint {path} predates the v3.0 paper protocol signature. '
                'Retrain it with v3.0 for paper experiments, or use --allow-legacy-checkpoint only for debugging.'
            )
        if sig is not None:
            expected = {
                'dataset': bundle.name, 'history_len': bundle.history_len, 'pred_len': bundle.pred_len,
                'num_nodes': bundle.num_nodes, 'scaler_fit': bundle.scaler_fit,
                'split_protocol': bundle.split_protocol,
                'evaluation_track': str(getattr(model, 'evaluation_track', 'A')),
                'model_protocol': str(getattr(model, 'model_protocol', 'v3.0-native')),
                'forecast_protocol': 'direct_horizon_matched',
            }
            for k, v in expected.items():
                if sig.get(k) != v:
                    raise RuntimeError(f'Checkpoint/data protocol mismatch for {k}: {sig.get(k)!r} != {v!r}. Retrain.')
            if abs(float(sig.get('scaler_mean')) - float(bundle.scaler.mean)) > 1e-6 or \
               abs(float(sig.get('scaler_std')) - float(bundle.scaler.std)) > 1e-6:
                raise RuntimeError('Checkpoint scaler statistics differ from current v2.4 preprocessing. Retrain.')
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    return model


def uniform_sample_indices(n, max_samples=256):
    """Deterministic, whole-period representation sampling."""
    n = int(n)
    if n <= 0:
        raise ValueError('Cannot sample representation indices from an empty array.')
    k = min(int(max_samples), n)
    if k <= 0:
        raise ValueError(f'max_samples must be positive, got {max_samples}')
    if k == n:
        return np.arange(n, dtype=np.int64)
    # Integer arithmetic avoids duplicated rounded linspace points.
    idx = np.floor(np.arange(k, dtype=np.float64) * (n - 1) / (k - 1)).astype(np.int64)
    idx[-1] = n - 1
    if len(np.unique(idx)) != len(idx):
        raise RuntimeError('Uniform representation sampling unexpectedly produced duplicate indices.')
    return idx


def _canonicalize_representation_rows(z):
    """Paper extraction interface: aligned sample rows + fixed flattening, no learned pooling."""
    a = np.asarray(z, dtype=np.float32)
    if a.ndim == 0:
        raise ValueError('Representation must have a sample dimension.')
    if a.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)
    return a.reshape(a.shape[0], -1)


def _collect(model, x_arr, y_arr, adj_arr, bundle, device, batch_size=32, rep_indices=None,
             max_rep_samples=256, y_mask_arr=None):
    n = len(x_arr)
    if len(y_arr) != n:
        raise ValueError(f'x/y sample mismatch: {n} vs {len(y_arr)}')
    if y_mask_arr is None:
        y_mask_arr = np.ones_like(y_arr, dtype=bool)
    y_mask_arr = np.asarray(y_mask_arr, dtype=bool)
    if y_mask_arr.shape != y_arr.shape:
        raise ValueError(f'y_mask shape mismatch: {y_mask_arr.shape} vs {y_arr.shape}')
    if rep_indices is None:
        rep_indices = uniform_sample_indices(n, max_rep_samples)
    rep_indices = np.asarray(rep_indices, dtype=np.int64)
    if np.any(rep_indices < 0) or np.any(rep_indices >= n):
        raise ValueError('rep_indices out of range.')
    rep_indices = np.sort(np.unique(rep_indices))

    loader = DataLoader(
        TrafficTensorDataset(x_arr, y_arr, y_mask_arr),
        batch_size=batch_size, shuffle=False,
    )
    is_foundation = bool(getattr(model, 'is_zero_shot', False))
    track = str(getattr(model, 'evaluation_track', 'A')).upper()
    parent_device = torch.device('cpu') if (is_foundation and track == 'A') else device
    adj_device = parent_device if (is_foundation and track == 'A') else device
    adj = torch.from_numpy(adj_arr).float().to(adj_device)
    preds, ys, ymasks, reps = [], [], [], []
    cursor = 0
    total_batches = len(loader)
    progress_every = max(1, total_batches // 20) if total_batches else 1
    if is_foundation:
        cache_info = model.cache_stats() if hasattr(model, 'cache_stats') else {}
        print(
            f'[Foundation full] model={model.model_name} track={track} test_windows={n} '
            f'outer_batch={batch_size} batches={total_batches} rep_windows={len(rep_indices)} '
            f'cache={cache_info.get("foundation_cache_enabled", False)}', flush=True,
        )

    model.eval()
    t0 = time.monotonic()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            x, y, ymask = batch
            b = len(x)
            start, stop = cursor, cursor + b
            pick = (rep_indices >= start) & (rep_indices < stop)
            need_rep = bool(np.any(pick))
            x = x.to(parent_device, non_blocking=False)
            if hasattr(model, 'inference_forward_selective_rep'):
                out = model.inference_forward_selective_rep(x, adj, bundle.pred_len, need_rep=need_rep)
            else:
                out = model.inference_forward(x, adj, bundle.pred_len)
            preds.append(out.pred.detach().cpu().numpy())
            ys.append(y.numpy())
            ymasks.append(ymask.numpy().astype(bool))
            if need_rep:
                local = rep_indices[pick] - start
                r = out.representation.detach().cpu().float().numpy()
                if r.shape[-1] == 0:
                    raise RuntimeError('Representation was requested but the model returned an empty tensor.')
                reps.append(r[local])
            cursor = stop

            if is_foundation and (batch_idx == 1 or batch_idx == total_batches or batch_idx % progress_every == 0):
                elapsed = max(time.monotonic() - t0, 1e-6)
                done = min(stop, n)
                rate = done / elapsed
                remain = (n - done) / max(rate, 1e-9)
                cache_info = model.cache_stats() if hasattr(model, 'cache_stats') else {}
                print(
                    f'[Foundation full] {model.model_name} {done}/{n} '
                    f'({100.0*done/max(n,1):.1f}%) elapsed={elapsed/60:.1f}m '
                    f'eta={remain/60:.1f}m cache_hit={cache_info.get("foundation_cache_hits",0)} '
                    f'cache_miss={cache_info.get("foundation_cache_misses",0)}', flush=True,
                )

    p_norm = np.concatenate(preds)
    y_norm = np.concatenate(ys)
    y_mask = np.concatenate(ymasks).astype(bool)
    z = _canonicalize_representation_rows(np.concatenate(reps)) if reps else np.zeros((0, 1), dtype=np.float32)
    if len(z) != len(rep_indices):
        raise RuntimeError(f'Collected {len(z)} representations for {len(rep_indices)} requested indices.')

    output_space = str(getattr(model, 'output_space', 'normalized')).lower()
    if output_space == 'normalized':
        p_eval = bundle.scaler.inverse(p_norm)
    elif output_space == 'original':
        p_eval = p_norm
    else:
        raise RuntimeError(f'Unknown output_space={output_space} for {model.model_name}')
    y_eval = bundle.scaler.inverse(y_norm)
    return p_eval, y_eval, z, rep_indices, y_mask


def _collect_representations(model, x_arr, adj_arr, bundle, device, batch_size=32):
    """Collect representations for every row in a small, already-selected x array."""
    tx = torch.from_numpy(np.asarray(x_arr, dtype=np.float32))
    loader = DataLoader(TensorDataset(tx), batch_size=batch_size, shuffle=False)
    is_foundation = bool(getattr(model, 'is_zero_shot', False))
    track = str(getattr(model, 'evaluation_track', 'A')).upper()
    parent_device = torch.device('cpu') if (is_foundation and track == 'A') else device
    adj = torch.from_numpy(adj_arr).float().to(parent_device if track == 'A' and is_foundation else device)
    reps = []
    model.eval()
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(parent_device)
            if hasattr(model, 'inference_forward_selective_rep'):
                out = model.inference_forward_selective_rep(x, adj, bundle.pred_len, need_rep=True)
            else:
                out = model.inference_forward(x, adj, bundle.pred_len)
            reps.append(out.representation.detach().cpu().float().numpy())
    return _canonicalize_representation_rows(np.concatenate(reps)) if reps else np.zeros((0, 1), dtype=np.float32)


def _metric_kwargs(bundle, error_mask=None, mape_mask=None):
    return dict(
        report_primary_mape=bundle.report_primary_mape,
        error_mask=error_mask,
        mape_mask=mape_mask,
        mape_min_abs=float(bundle.mape_min_abs),
    )


def evaluate_original(model, bundle, device, run_dir, batch_size=32, max_rep_samples=256):
    run_dir = ensure_dir(run_dir)
    t0 = time.monotonic()
    rep_indices = uniform_sample_indices(len(bundle.test_x), max_rep_samples)
    pred, y, z, rep_indices, y_mask = _collect(
        model, bundle.test_x, bundle.test_y, bundle.adj, bundle, device,
        batch_size, rep_indices=rep_indices, max_rep_samples=max_rep_samples,
        y_mask_arr=bundle.test_y_mask,
    )
    evaluation_seconds = float(time.monotonic() - t0)
    metrics = prediction_metrics(y, pred, **_metric_kwargs(bundle, y_mask, y_mask))
    metrics['IPE'] = metrics['MAE']
    metrics['IPE_definition'] = 'masked_MAE_on_observed_targets'
    metrics['primary_MAPE_reported'] = bool(bundle.report_primary_mape)
    metrics['missing_value_policy'] = str(bundle.missing_value_policy)
    save_json(Path(run_dir) / 'metrics_original.json', metrics)
    np.savez_compressed(Path(run_dir) / 'predictions_original.npz', pred=pred, y=y, y_mask=y_mask)
    np.savez_compressed(Path(run_dir) / 'representations_original.npz', z=z, rep_indices=rep_indices)
    cache_info = model.cache_stats() if hasattr(model, 'cache_stats') else {}
    meta = {
        'dataset': bundle.name,
        'model': model.model_name,
        'uses_graph': bool(model.uses_graph),
        'native_uses_graph': bool(getattr(model, 'native_uses_graph', model.uses_graph)),
        'evaluation_track': str(getattr(model, 'evaluation_track', 'A')),
        'coordinate_source': str(getattr(bundle, 'coordinate_source', 'missing_zero_filled')),
        'coordinate_coverage': float(getattr(bundle, 'coordinate_coverage', 0.0)),
        'history_len': bundle.history_len,
        'pred_len': bundle.pred_len,
        'interval_minutes': bundle.interval_minutes,
        'temporal_protocol': bundle.temporal_protocol,
        'data_source': bundle.data_source,
        'prediction_shape_saved': list(pred.shape),
        'target_shape_saved': list(y.shape),
        'target_mask_shape_saved': list(y_mask.shape),
        'representation_shape_saved': list(z.shape),
        'representation_sampling': 'deterministic_uniform_rows_fixed_flattening',
        'representation_interface': 'immediately_preceding_final_prediction_head',
        'representation_pooling': 'fixed_flattening_only_no_learned_pooling',
        'representation_sample_count': int(len(rep_indices)),
        'representation_index_first': int(rep_indices[0]),
        'representation_index_last': int(rep_indices[-1]),
        'max_rep_samples': int(max_rep_samples),
        'scaler_fit': bundle.scaler_fit,
        'split_protocol': bundle.split_protocol,
        'model_protocol': str(getattr(model, 'model_protocol', 'v3.3-native')),
        'forecast_protocol': 'direct_horizon_matched',
        'output_space': str(getattr(model, 'output_space', 'normalized')),
        'metric_protocol': METRIC_PROTOCOL,
        'missing_value_policy': str(bundle.missing_value_policy),
        'speed_missing_policy': str(bundle.speed_missing_policy),
        'speed_missing_input_fill': str(bundle.speed_missing_input_fill),
        'valid_error_fraction': float(metrics['valid_error_fraction']),
        'valid_mape_fraction': metrics['valid_mape_fraction'],
        'mape_min_abs': float(bundle.mape_min_abs),
        'evaluation_seconds': evaluation_seconds,
        'batch_size': int(batch_size),
    }
    meta.update(cache_info)
    save_json(Path(run_dir) / 'evaluation_metadata.json', meta)
    print(f'Original: {metrics} | evaluation_time={evaluation_seconds/60:.2f} min', flush=True)
    return metrics, pred, y, y_mask, z, rep_indices


def append_result_csv(path, row):
    path = Path(path); ensure_dir(path.parent)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _validity(model, intervention, force_include_adjacency_only=False, observability=None):
    """Scientific validity flags for native, information-equivalent and physical tracks."""
    if intervention.leakage_detected or not intervention.causal_valid:
        return False, False, False, False, False, 'environment failed causal-leakage validity check'
    if intervention.environment_source == 'external_physical':
        physical = intervention.kind in GRAPH_INTERVENTIONS
        return True, False, False, physical, physical, (
            'external physical/simulated environment with aligned Xk/Yk/Ak; valid for architecture-complete physical structural comparison'
        )
    if intervention.kind in GRAPH_INTERVENTIONS and not model.uses_graph:
        if force_include_adjacency_only:
            return True, False, False, False, False, (
                'FORCED diagnostic row: graph-free Track-A model cannot observe adjacency-only intervention; excluded from structural aggregates'
            )
        return False, False, False, False, False, (
            'graph-free Track-A model cannot observe adjacency-only edge intervention; use Track B (DSAI) or external physical environments'
        )
    native_structural = bool(intervention.family == 'structural' and str(getattr(model, 'evaluation_track', 'A')) == 'A')
    info_equiv = bool(intervention.family == 'structural' and str(getattr(model, 'evaluation_track', 'A')) == 'B')
    if intervention.family == 'temporal':
        return True, False, False, False, False, 'leakage-safe matched past-only temporal lag; separated from structural aggregates'
    if intervention.family == 'sensor':
        return True, False, False, False, False, 'sensor intervention; separated from structural aggregates'
    if info_equiv:
        if observability is None or not bool(observability.get('dsai_observable', False)):
            return False, False, False, False, False, (
                'INVALID Track-B row: graph changed but the DSAI-conditioned model input did not measurably change; '
                'do not interpret DMIS=0 as representation rigidity'
            )
        return True, False, True, False, True, (
            'Track-B weight-aware static + dynamic DSAI makes the CSG topology intervention observably available to all architectures; '
            'controlled-input diagnostic, not physical target-shift'
        )
    if native_structural:
        return True, True, False, False, False, 'Track-A native topology-observable CSG structural intervention'
    return True, False, False, False, False, 'observable intervention under current implementation'

def _matched_baseline(base_metrics, base_pred, base_y, base_y_mask, bundle, baseline_indices, evaluation_mask=None):
    """Matched IPE0, optionally restricted to the exact intervention support.

    Node isolation requires IPE0 and IPEk to be evaluated on the same unisolated
    node set.  `evaluation_mask` carries that support without changing the data split.
    """
    idx = np.asarray(baseline_indices, dtype=np.int64)
    bp = base_pred[idx]
    by = base_y[idx]
    bm = np.asarray(base_y_mask, dtype=bool)[idx]
    if evaluation_mask is not None:
        em = np.asarray(evaluation_mask, dtype=bool)
        if em.shape != bm.shape:
            raise ValueError(f'evaluation_mask shape mismatch: {em.shape} vs {bm.shape}')
        bm = bm & em
    full = len(idx) == len(base_y) and np.array_equal(idx, np.arange(len(base_y)))
    if full and evaluation_mask is None:
        return base_metrics, bp, by, bm
    met = prediction_metrics(by, bp, **_metric_kwargs(bundle, bm, bm))
    met['IPE'] = met['MAE']
    met['IPE_definition'] = 'masked_MAE_on_matched_targets_and_support'
    met['primary_MAPE_reported'] = bool(bundle.report_primary_mape)
    met['missing_value_policy'] = str(bundle.missing_value_policy)
    return met, bp, by, bm



def _trackb_observability(model, bundle, intervention, baseline_indices, device, max_samples=64):
    """Check DSAI *input* observability independently of latent/model response."""
    is_trackb = str(getattr(model, 'evaluation_track', 'A')) == 'B'
    if not (is_trackb and intervention.kind in GRAPH_INTERVENTIONS and intervention.environment_source != 'external_physical'):
        return {
            'dsai_observability_checked': False,
            'dsai_static_delta': float('nan'),
            'dsai_dynamic_message_delta': float('nan'),
            'dsai_conditioned_input_delta': float('nan'),
            'dsai_conditioned_input_relative_delta': float('nan'),
            'dsai_observable': True,
        }
    if not hasattr(model, 'observability_diagnostics'):
        return {
            'dsai_observability_checked': True,
            'dsai_static_delta': 0.0,
            'dsai_dynamic_message_delta': 0.0,
            'dsai_conditioned_input_delta': 0.0,
            'dsai_conditioned_input_relative_delta': 0.0,
            'dsai_observable': False,
        }
    local = uniform_sample_indices(len(intervention.x), min(max_samples, len(intervention.x)))
    target = np.asarray(baseline_indices, dtype=np.int64)[local]
    xb = torch.from_numpy(np.asarray(bundle.test_x[target], dtype=np.float32)).to(device)
    xk = torch.from_numpy(np.asarray(intervention.x[local], dtype=np.float32)).to(device)
    a0 = torch.from_numpy(np.asarray(bundle.adj, dtype=np.float32)).to(device)
    ak = torch.from_numpy(np.asarray(intervention.adj, dtype=np.float32)).to(device)
    d = model.observability_diagnostics(xb, a0, xk, ak)
    d['dsai_observability_checked'] = True
    return d

def evaluate_intervention(model, bundle, intervention, device, run_dir,
                          base_metrics, base_pred, base_y, base_y_mask, z0_global, base_rep_indices,
                          batch_size=32, max_rep_samples=256, alpha=0.5, idx=0,
                          force_include_adjacency_only=False, artifact_level='minimal',
                          fail_on_unobservable_trackb=True):
    baseline_indices = (
        np.asarray(intervention.baseline_indices, dtype=np.int64)
        if intervention.baseline_indices is not None
        else np.arange(len(bundle.test_x), dtype=np.int64)
    )
    if np.any(baseline_indices < 0) or np.any(baseline_indices >= len(bundle.test_x)):
        raise ValueError('Intervention baseline_indices out of test-set range.')
    if len(intervention.x) != len(baseline_indices) or len(intervention.y) != len(baseline_indices):
        raise ValueError(
            f'Intervention/aligned baseline length mismatch: x={len(intervention.x)}, '
            f'y={len(intervention.y)}, baseline={len(baseline_indices)}'
        )

    intervention_mask = (
        np.asarray(intervention.y_mask, dtype=bool)
        if getattr(intervention, 'y_mask', None) is not None
        else bundle.infer_external_target_mask(bundle.scaler.inverse(intervention.y))
    )
    # Appendix A.1.1: node-isolation RD uses exactly the same unisolated-node
    # support for IPE0 and IPEk.  Other interventions retain the ordinary matched mask.
    same_support_mask = intervention_mask if intervention.kind == 'node_isolation' else None
    base_match_metrics, _, base_y_matched, base_mask_matched = _matched_baseline(
        base_metrics, base_pred, base_y, base_y_mask, bundle, baseline_indices,
        evaluation_mask=same_support_mask,
    )

    full_alignment = (
        len(baseline_indices) == len(bundle.test_x)
        and np.array_equal(baseline_indices, np.arange(len(bundle.test_x)))
    )
    if full_alignment:
        rep_local_indices = np.asarray(base_rep_indices, dtype=np.int64)
        rep_target_indices = rep_local_indices.copy()
    else:
        # Deterministic uniform positions over the ALIGNED intervention subset.
        rep_local_indices = uniform_sample_indices(len(intervention.x), max_rep_samples)
        rep_target_indices = baseline_indices[rep_local_indices]

    # For non-structural observational controls, preserve the matched baseline
    # representation definition.  Structural DMIS is set below using X_k for both
    # topology arguments, as required by the paper.
    if intervention.family not in ('structural', 'physical'):
        if full_alignment:
            z0 = z0_global
        else:
            z0 = _collect_representations(
                model, bundle.test_x[rep_target_indices], bundle.adj, bundle, device, batch_size
            )
    pred, y, zk, used_rep_indices, y_mask = _collect(
        model, intervention.x, intervention.y, intervention.adj, bundle, device,
        batch_size, rep_indices=rep_local_indices, max_rep_samples=max_rep_samples,
        y_mask_arr=intervention_mask,
    )
    if not np.array_equal(used_rep_indices, rep_local_indices):
        raise RuntimeError('Representation index alignment changed during intervention collection.')

    if intervention.family in ('structural', 'physical'):
        # Paired DMIS protocol: hold X_k fixed and vary only topology,
        # Z_k^(0)=z(X_k,A0), Z_k^(T)=z(X_k,Ak).
        # Internal CSG structural interventions leave X unchanged.  Under full
        # alignment the already-computed baseline representation therefore IS
        # exactly z(X_k,A0), so reuse it instead of repeating an expensive
        # foundation-model representation pass for every intervention.
        can_reuse_baseline_z0 = (
            intervention.environment_source != 'external_physical'
            and full_alignment
            and len(z0_global) == len(rep_local_indices)
        )
        if can_reuse_baseline_z0:
            z0 = z0_global
        else:
            z0 = _collect_representations(
                model, np.asarray(intervention.x)[rep_local_indices], bundle.adj,
                bundle, device, batch_size
            )

    met = prediction_metrics(y, pred, **_metric_kwargs(bundle, y_mask, y_mask))
    ipe0 = float(base_match_metrics['IPE'])
    ipek = float(met['MAE'])
    baseline_y_std = masked_std(base_y_matched, base_mask_matched)

    if intervention.kind in GRAPH_INTERVENTIONS:
        dm_graph, raw_graph, cd, mg, graph_sd = dmis_graph_formula(
            z0, zk, bundle.adj, intervention.adj, alpha=alpha
        )
    else:
        dm_graph = raw_graph = cd = mg = graph_sd = float('nan')
    dm_gen, raw_gen, cd2, mg2, env_dist = dmis_generalized(
        z0, zk, intervention.environment_distance, alpha=alpha
    )
    # environment_distance is the paper family-specific d_T: normalized Frobenius
    # for edge/weight interventions and |V_iso|/N for node isolation.
    dm_eval = dm_gen
    raw_response = raw_gen
    rd_k = relative_degradation(ipe0, ipek, baseline_y_std)
    observability = _trackb_observability(model, bundle, intervention, baseline_indices, device)
    include_main, include_structural, include_info_equiv, include_physical_structural, include_arch_complete, validity_note = _validity(
        model, intervention, force_include_adjacency_only, observability=observability
    )
    if (fail_on_unobservable_trackb and observability.get('dsai_observability_checked', False)
            and not observability.get('dsai_observable', False)):
        raise RuntimeError(
            f'Track-B observability gate failed for {intervention.kind} id={idx}: {observability}. '
            'The graph changed but the DSAI-conditioned input did not. This is an implementation-invalid row, not rigidity.'
        )

    out_dir = ensure_dir(Path(run_dir)/'interventions')
    stem = f'{intervention.kind}_{idx:03d}'
    source_indices = (
        np.asarray(intervention.source_indices, dtype=np.int64)
        if intervention.source_indices is not None else np.array([], dtype=np.int64)
    )
    artifact_npz = out_dir/f'{stem}.npz'
    if artifact_level == 'full':
        np.savez_compressed(
            artifact_npz, pred=pred, y=y, y_mask=y_mask, z=zk, adj=intervention.adj,
            baseline_indices=baseline_indices, source_indices=source_indices,
            rep_local_indices=rep_local_indices, rep_target_indices=rep_target_indices
        )
    elif artifact_level != 'minimal':
        raise ValueError(f'Unknown artifact_level={artifact_level}')
    artifact_json = out_dir/f'{stem}.json'
    save_json(artifact_json, {
        'kind': intervention.kind,
        'intervention_family': intervention.family,
        'severity': intervention.severity,
        'environment_distance': intervention.environment_distance,
        'distance_kind': intervention.distance_kind,
        'environment_source': intervention.environment_source,
        'causal_valid': bool(intervention.causal_valid),
        'leakage_detected': bool(intervention.leakage_detected),
        'include_in_main': include_main,
        'include_in_main_structural': include_structural,
        'include_in_information_equivalent_structural': include_info_equiv,
        'include_in_physical_structural': include_physical_structural,
        'include_in_architecture_complete_structural': include_arch_complete,
        'evaluation_track': str(getattr(model, 'evaluation_track', 'A')),
        'validity_note': validity_note,
        'dsai_observability': observability,
        'aligned_baseline_sample_count': int(len(baseline_indices)),
        'representation_sampling': 'deterministic_uniform_rows_fixed_flattening',
        'dmis_pairing_protocol': 'Z0=z(Xk,A0), ZT=z(Xk,Ak) for structural/physical interventions',
        'representation_centering': 'none_paper_raw_aligned_representations',
        'representation_sample_count': int(len(rep_local_indices)),
        'rep_target_index_first': int(rep_target_indices[0]),
        'rep_target_index_last': int(rep_target_indices[-1]),
        'artifact_level': artifact_level,
        'diagnostic_subset_active': bool(getattr(bundle, 'diagnostic_subset_active', False)),
        'diagnostic_original_test_windows': int(getattr(bundle, 'diagnostic_original_test_windows', len(bundle.test_x))),
        'diagnostic_selected_test_windows': int(len(bundle.test_x)),
        'diagnostic_selected_node_series': int(getattr(bundle, 'diagnostic_selected_node_series', len(bundle.test_x) * bundle.num_nodes)),
        'diagnostic_node_series_budget': getattr(bundle, 'diagnostic_node_series_budget', None),
        'metric_protocol': METRIC_PROTOCOL,
        'missing_value_policy': str(bundle.missing_value_policy),
        'valid_error_fraction_k': met.get('valid_error_fraction'),
        'valid_mape_fraction_k': met.get('valid_mape_fraction'),
        'full_array_artifact': str(artifact_npz) if artifact_level == 'full' else None,
        'meta': intervention.meta,
    })

    row = {
        'dataset': bundle.name,
        'model': model.model_name,
        'model_uses_graph': bool(model.uses_graph),
        'native_model_uses_graph': bool(getattr(model, 'native_uses_graph', model.uses_graph)),
        'evaluation_track': str(getattr(model, 'evaluation_track', 'A')),
        'intervention_type': intervention.kind,
        'intervention_family': intervention.family,
        'intervention_id': int(idx),
        'intervention_seed': int(intervention.meta.get('seed', -1)),
        'paper_intensity_level': str(intervention.meta.get('paper_intensity_level', '')),
        'intervention_strength': float(intervention.severity),
        'environment_source': intervention.environment_source,
        'causal_valid': bool(intervention.causal_valid),
        'leakage_detected': bool(intervention.leakage_detected),
        'include_in_main': bool(include_main),
        'include_in_main_structural': bool(include_structural),
        'include_in_information_equivalent_structural': bool(include_info_equiv),
        'include_in_physical_structural': bool(include_physical_structural),
        'include_in_architecture_complete_structural': bool(include_arch_complete),
        'validity_note': validity_note,
        'dsai_observability_checked': bool(observability.get('dsai_observability_checked', False)),
        'dsai_observable': bool(observability.get('dsai_observable', True)),
        'dsai_static_delta': observability.get('dsai_static_delta', float('nan')),
        'dsai_dynamic_message_delta': observability.get('dsai_dynamic_message_delta', float('nan')),
        'dsai_conditioned_input_delta': observability.get('dsai_conditioned_input_delta', float('nan')),
        'dsai_conditioned_input_relative_delta': observability.get('dsai_conditioned_input_relative_delta', float('nan')),
        # Global baseline is kept for traceability; matched baseline is used by ARD.
        'MAE0_global': base_metrics['MAE'], 'RMSE0_global': base_metrics['RMSE'],
        'MAPE0_global': base_metrics['MAPE'], 'IPE0_global': base_metrics['IPE'],
        'MAE0': base_match_metrics['MAE'], 'RMSE0': base_match_metrics['RMSE'],
        'MAPE0': base_match_metrics['MAPE'], 'MAPE0_raw': base_match_metrics['MAPE_raw'],
        'IPE0': ipe0,
        'MAEk': met['MAE'], 'RMSEk': met['RMSE'], 'MAPEk': met['MAPE'], 'MAPEk_raw': met['MAPE_raw'],
        'IPEk': ipek,
        'valid_error_fraction_k': met.get('valid_error_fraction'),
        'valid_mape_fraction_k': met.get('valid_mape_fraction'),
        'metric_protocol': met.get('metric_protocol', METRIC_PROTOCOL),
        'RD': rd_k,
        'ARD_k': rd_k,  # backwards-compatible alias; ARD is the mean of RD over interventions
        'representation_response_raw': raw_response,
        'DMIS': dm_eval,
        'DMIS_graph_formula': dm_graph,
        'DMIS_generalized': dm_gen,
        'cosine_change': max(0.0, float(cd if np.isfinite(cd) else cd2)),
        'magnitude_change': max(0.0, float(mg if np.isfinite(mg) else mg2)),
        'graph_structural_distance': graph_sd,
        'environment_distance': env_dist,
        'distance_kind': intervention.distance_kind,
        'baseline_eval_count': int(len(base_y_matched)),
        'intervention_eval_count': int(len(y)),
        'representation_sampling': 'uniform_aligned_rows_fixed_flattening',
        'representation_sample_count': int(len(rep_local_indices)),
        'rep_target_index_first': int(rep_target_indices[0]),
        'rep_target_index_last': int(rep_target_indices[-1]),
        'artifact_level': artifact_level,
        'diagnostic_subset_active': bool(getattr(bundle, 'diagnostic_subset_active', False)),
        'diagnostic_original_test_windows': int(getattr(bundle, 'diagnostic_original_test_windows', len(bundle.test_x))),
        'diagnostic_selected_test_windows': int(len(bundle.test_x)),
        'diagnostic_selected_node_series': int(getattr(bundle, 'diagnostic_selected_node_series', len(bundle.test_x) * bundle.num_nodes)),
        'diagnostic_node_series_budget': getattr(bundle, 'diagnostic_node_series_budget', None),
        'artifact': str(artifact_npz) if artifact_level == 'full' else str(artifact_json),
    }
    append_result_csv(Path(run_dir)/'causalgap_results.csv', row)
    print(row)
    return row


def _bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(['true', '1', 'yes'])


def _summarize_group(g):
    rd_col = 'RD' if 'RD' in g.columns else 'ARD_k'
    ard_mean, ard_std, ard_se, n_ard = mean_std_se(g[rd_col].to_numpy())
    dm_mean, dm_std, dm_se, n_dm = mean_std_se(g['DMIS'].to_numpy())
    rr_mean, rr_std, rr_se, n_rr = mean_std_se(g['representation_response_raw'].to_numpy())
    return {
        'dataset': g['dataset'].iloc[0],
        'model': g['model'].iloc[0],
        'intervention_type': g['intervention_type'].iloc[0],
        'intervention_family': g['intervention_family'].iloc[0],
        'environment_source': g['environment_source'].iloc[0],
        'include_in_main': bool(_bool_series(g['include_in_main']).iloc[0]),
        'include_in_main_structural': bool(_bool_series(g['include_in_main_structural']).iloc[0]),
        'include_in_information_equivalent_structural': bool(_bool_series(g['include_in_information_equivalent_structural']).iloc[0]),
        'include_in_physical_structural': bool(_bool_series(g['include_in_physical_structural']).iloc[0]),
        'include_in_architecture_complete_structural': bool(_bool_series(g['include_in_architecture_complete_structural']).iloc[0]),
        'evaluation_track': str(g['evaluation_track'].iloc[0]),
        'K': int(len(g)),
        'ARD': ard_mean, 'ARD_std': ard_std, 'ARD_se': ard_se,
        'DMIS_mean': dm_mean, 'DMIS_std': dm_std, 'DMIS_se': dm_se,
        'representation_response_raw_mean': rr_mean,
        'representation_response_raw_std': rr_std,
        'representation_response_raw_se': rr_se,
        'MAE0_matched_mean': float(g['MAE0'].mean()),
        'MAEk_mean': float(g['MAEk'].mean()),
        'RMSEk_mean': float(g['RMSEk'].mean()),
        'n_finite_ARD': n_ard,
        'n_finite_DMIS': n_dm,
        'n_finite_raw_response': n_rr,
        'n_causal_valid': int(_bool_series(g['causal_valid']).sum()),
        'n_leakage_detected': int(_bool_series(g['leakage_detected']).sum()),
    }


def aggregate_run_results(run_dir):
    run_dir = Path(run_dir)
    csv_path = run_dir / 'causalgap_results.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    rows = []
    keys = [
        'intervention_type', 'intervention_family', 'environment_source', 'evaluation_track',
        'include_in_main', 'include_in_main_structural', 'include_in_information_equivalent_structural',
        'include_in_physical_structural', 'include_in_architecture_complete_structural'
    ]
    for _, g in df.groupby(keys, dropna=False):
        rows.append(_summarize_group(g))
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir/'intervention_summary.csv', index=False)

    main_mask = _bool_series(summary['include_in_main'])
    summary[main_mask].to_csv(run_dir/'intervention_summary_main_valid.csv', index=False)
    structural_mask = _bool_series(summary['include_in_main_structural'])
    summary[structural_mask].to_csv(run_dir/'intervention_summary_main_structural.csv', index=False)
    info_mask = _bool_series(summary['include_in_information_equivalent_structural'])
    summary[info_mask].to_csv(run_dir/'intervention_summary_information_equivalent_structural.csv', index=False)
    physical_mask = _bool_series(summary['include_in_physical_structural'])
    summary[physical_mask].to_csv(run_dir/'intervention_summary_physical_structural.csv', index=False)
    arch_mask = _bool_series(summary['include_in_architecture_complete_structural'])
    summary[arch_mask].to_csv(run_dir/'intervention_summary_architecture_complete_structural.csv', index=False)
    for family in ('structural', 'sensor', 'temporal', 'physical'):
        summary[summary['intervention_family'] == family].to_csv(
            run_dir/f'intervention_summary_family_{family}.csv', index=False
        )
    print(f'Saved intervention summary -> {run_dir / "intervention_summary.csv"}')
    print(f'Saved core structural summary -> {run_dir / "intervention_summary_main_structural.csv"}')
    print(f'Saved information-equivalent Track-B structural summary -> {run_dir / "intervention_summary_information_equivalent_structural.csv"}')
    print(f'Saved physical structural summary -> {run_dir / "intervention_summary_physical_structural.csv"}')
    print(f'Saved architecture-complete structural summary -> {run_dir / "intervention_summary_architecture_complete_structural.csv"}')
    return summary
