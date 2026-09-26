from __future__ import annotations

import math
import numpy as np


METRIC_PROTOCOL = 'v3.3_masked_error_primary_mape'


def _as_float_arrays(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f'Metric shape mismatch: y={y.shape}, pred={p.shape}')
    return y, p


def _coerce_mask(mask, shape, default=True):
    if mask is None:
        return np.ones(shape, dtype=bool) if default else np.zeros(shape, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if m.shape != shape:
        raise ValueError(f'Mask shape mismatch: mask={m.shape}, expected={shape}')
    return m


def masked_mean(values, mask):
    x = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if x.shape != m.shape:
        raise ValueError((x.shape, m.shape))
    if not np.any(m):
        return float('nan')
    return float(np.mean(x[m]))


def mae(y, p, mask=None):
    y, p = _as_float_arrays(y, p)
    finite = np.isfinite(y) & np.isfinite(p)
    finite &= _coerce_mask(mask, y.shape)
    return masked_mean(np.abs(p - y), finite)


def rmse(y, p, mask=None):
    y, p = _as_float_arrays(y, p)
    finite = np.isfinite(y) & np.isfinite(p)
    finite &= _coerce_mask(mask, y.shape)
    mse = masked_mean((p - y) ** 2, finite)
    return float(np.sqrt(mse)) if np.isfinite(mse) else float('nan')


def mape(y, p, mask=None, min_abs=1.0):
    y, p = _as_float_arrays(y, p)
    valid = np.isfinite(y) & np.isfinite(p)
    valid &= _coerce_mask(mask, y.shape)
    valid &= np.abs(y) >= float(min_abs)
    if not np.any(valid):
        return float('nan')
    return float(np.mean(np.abs((p[valid] - y[valid]) / y[valid])) * 100.0)


def mape_unmasked_diagnostic(y, p, eps=1e-5):
    """Deliberately unstable raw diagnostic.

    This quantity is never the primary paper MAPE. It keeps every finite target
    and clips only the denominator so users can see why ordinary MAPE explodes on
    zero/near-zero traffic counts.
    """
    y, p = _as_float_arrays(y, p)
    valid = np.isfinite(y) & np.isfinite(p)
    if not np.any(valid):
        return float('nan')
    den = np.maximum(np.abs(y[valid]), float(eps))
    return float(np.mean(np.abs(p[valid] - y[valid]) / den) * 100.0)


def prediction_metrics(
    y,
    p,
    report_primary_mape=True,
    error_mask=None,
    mape_mask=None,
    mape_min_abs=1.0,
):
    y, p = _as_float_arrays(y, p)
    finite = np.isfinite(y) & np.isfinite(p)
    err_mask = finite & _coerce_mask(error_mask, y.shape)
    primary_mape_mask = err_mask & _coerce_mask(mape_mask, y.shape)

    primary = (
        mape(y, p, mask=primary_mape_mask, min_abs=mape_min_abs)
        if report_primary_mape else float('nan')
    )
    raw = mape_unmasked_diagnostic(y, p, eps=1e-5)

    return {
        'MAE': mae(y, p, mask=err_mask),
        'RMSE': rmse(y, p, mask=err_mask),
        'MAPE': primary,
        # Backwards-compatible field name. From v3.3 this explicitly means an
        # unmasked diagnostic and may legitimately be extremely large.
        'MAPE_raw': raw,
        'MAPE_unmasked_diagnostic': raw,
        'MAE_unmasked_diagnostic': mae(y, p, mask=finite),
        'RMSE_unmasked_diagnostic': rmse(y, p, mask=finite),
        'valid_error_fraction': float(np.mean(err_mask.astype(np.float64))),
        'valid_mape_fraction': float(np.mean(primary_mape_mask & (np.abs(y) >= float(mape_min_abs)))) if report_primary_mape else float('nan'),
        'mape_min_abs': float(mape_min_abs),
        'metric_protocol': METRIC_PROTOCOL,
    }


def masked_std(y, mask=None):
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(y) & _coerce_mask(mask, y.shape)
    if not np.any(valid):
        return float('nan')
    return float(np.std(y[valid]))


def _representation_matrix(z):
    """Canonical paper interface: one aligned sample per row, fixed flattening within a row."""
    a = np.asarray(z, dtype=np.float64)
    if a.ndim == 0:
        raise ValueError('Representation must have a sample/row dimension.')
    if a.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float64)
    return a.reshape(a.shape[0], -1)


def _paired_centered_representations(z0, zk):
    a = _representation_matrix(z0)
    b = _representation_matrix(zk)
    if a.shape != b.shape:
        raise ValueError(
            f'Representation shape mismatch after fixed flattening: {a.shape} vs {b.shape}. '
            'CausalGap requires row-aligned paired representations.'
        )
    if a.shape[0] == 0:
        return a, b
    # Center each feature over the aligned rows.  This removes a shared translation
    # while preserving the row pairing used by the paper's Frobenius metrics.
    return a - np.mean(a, axis=0, keepdims=True), b - np.mean(b, axis=0, keepdims=True)


def cosine_distance(z0, zk, eps=1e-12):
    a, b = _paired_centered_representations(z0, zk)
    if a.size == 0:
        return float('nan')
    av = a.reshape(-1)
    bv = b.reshape(-1)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom <= eps:
        return 0.0 if np.linalg.norm(av - bv) <= eps else 1.0
    cos = float(np.dot(av, bv) / denom)
    cos = float(np.clip(cos, -1.0, 1.0))
    out = max(0.0, 1.0 - cos)
    return 0.0 if out < 1e-12 else out


def magnitude_change(z0, zk, eps=1e-12):
    a, b = _paired_centered_representations(z0, zk)
    if a.size == 0:
        return float('nan')
    denom = np.linalg.norm(a)
    diff = np.linalg.norm(b - a)
    if denom <= eps:
        return 0.0 if diff <= eps else float('inf')
    out = max(0.0, float(diff / denom))
    return 0.0 if out < 1e-12 else out


def representation_response(z0, zk, alpha=0.5, eps=1e-12):
    """D_Z = alpha*(1-cos_F) + (1-alpha)*relative Frobenius change."""
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f'alpha must be in [0,1], got {alpha}')
    cd = cosine_distance(z0, zk, eps)
    mg = magnitude_change(z0, zk, eps)
    if not np.isfinite(cd) or not np.isfinite(mg):
        return float('nan'), cd, mg
    raw = max(0.0, float(alpha * cd + (1.0 - alpha) * mg))
    return (0.0 if raw < 1e-12 else raw), cd, mg


def graph_frobenius_distance(a0, ak, eps_a=1e-6):
    """Paper d_T for edge removal / edge-weight modification."""
    a0 = np.asarray(a0, dtype=np.float64)
    ak = np.asarray(ak, dtype=np.float64)
    if a0.shape != ak.shape:
        raise ValueError(f'Adjacency shape mismatch: {a0.shape} vs {ak.shape}')
    return max(0.0, float(np.linalg.norm(ak - a0) / (np.linalg.norm(a0) + float(eps_a))))


def _dmis_from_distance(z0, zk, distance, alpha=0.5, eps_a=1e-6, rep_eps=1e-12):
    raw, cd, mg = representation_response(z0, zk, alpha, rep_eps)
    dist = max(0.0, float(distance))
    if not np.isfinite(raw):
        return float('nan'), raw, cd, mg
    dm = max(0.0, float(raw / max(dist, float(eps_a))))
    dm = 0.0 if dm < 1e-12 else dm
    return dm, raw, cd, mg


def dmis_graph_formula(z0, zk, a0, ak, alpha=0.5, eps=1e-6):
    sd = graph_frobenius_distance(a0, ak, eps_a=eps)
    dm, raw, cd, mg = _dmis_from_distance(z0, zk, sd, alpha, eps_a=eps)
    return dm, raw, cd, mg, sd


def dmis_generalized(z0, zk, environment_distance, alpha=0.5, eps=1e-6):
    """DMIS using the intervention-family d_T supplied by the generator.

    For edge/weight interventions this is normalized Frobenius distance; for node
    isolation it is |V_iso|/N.  Temporal values remain diagnostic controls only.
    """
    dist = max(0.0, float(environment_distance))
    dm, raw, cd, mg = _dmis_from_distance(z0, zk, dist, alpha, eps_a=eps)
    return dm, raw, cd, mg, dist


def relative_degradation(ipe0, ipek, y_std, eps=1e-12):
    """RD_k=(IPE_k-IPE_0)/(IPE_0+tau), tau=0.05*std(Y_0)."""
    tau = 0.05 * float(y_std)
    denom = float(ipe0) + tau
    if not np.isfinite(denom) or denom <= eps:
        return float('nan')
    return float((float(ipek) - float(ipe0)) / denom)


def ard_k(ipe0, ipek, y_std, eps=1e-12):
    """Backward-compatible name for one RD_k value (ARD is the mean over k)."""
    return relative_degradation(ipe0, ipek, y_std, eps=eps)


def mean_std_se(values):
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(x) == 0:
        return float('nan'), float('nan'), float('nan'), 0
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, float('nan'), float('nan'), 1
    std = float(np.std(x, ddof=1))
    se = float(std / math.sqrt(len(x)))
    return mean, std, se, int(len(x))
