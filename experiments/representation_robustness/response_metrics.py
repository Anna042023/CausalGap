#!/usr/bin/env python3
"""Alternative representation-response measures on exactly aligned rows.

CKA uses the small n-by-n Gram formulation; SVCCA and PWCCA use the
small sample-side PCA eigendecomposition. This is efficient for d >> n.
SVCCA/PWCCA use an outcome-independent centered-rank cap of floor((n-1)/2)
per side to avoid CCA saturation when d >> n. For n=32 this cap is 15.
"""
from __future__ import annotations
import math
from typing import Tuple, Dict
import numpy as np
from scipy.stats import spearmanr
EPS=1e-12
PCA_VARIANCE=0.99
CCA_DIM_FRACTION=0.5


def canonical_matrix(z: np.ndarray) -> np.ndarray:
    a = np.asarray(z, dtype=np.float64)
    if a.ndim == 0:
        raise ValueError("Representation must have a row/sample dimension")
    if a.shape[0] < 2:
        raise ValueError(f"Need at least two aligned rows; got {a.shape}")
    a = a.reshape(a.shape[0], -1)
    if not np.isfinite(a).all():
        raise ValueError("Representation contains NaN/inf")
    return a


def paired_center(z0: np.ndarray, zk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = canonical_matrix(z0)
    b = canonical_matrix(zk)
    if a.shape != b.shape:
        raise ValueError(f"Row-aligned representation shape mismatch: {a.shape} vs {b.shape}")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    return a, b


def linear_cka_similarity(x: np.ndarray, y: np.ndarray, eps: float = EPS) -> float:
    # Feature-space linear CKA. Equivalent to centered Gram CKA for centered X,Y.
    # n-by-n Gram formulation avoids allocating d-by-d matrices for STGNNs.
    xx = x @ x.T
    yy = y @ y.T
    num = float(np.sum(xx * yy))
    den = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    if den <= eps:
        return 1.0 if np.linalg.norm(x - y) <= eps else 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def cca_dim_cap(n_rows: int) -> int:
    """Outcome-independent cap for CCA-family metrics.

    After column centering, an n-row representation has sample-space rank at
    most n-1.  Letting both CCA sides retain nearly n-1 directions makes CCA
    almost tautological in the d >> n regime.  We therefore cap each side at
    floor((n-1)/2), which guarantees k_x + k_y <= n-1 when both sides hit the
    cap.  For the paper protocol n=32, the fixed cap is 15.
    """
    if n_rows < 4:
        return 1
    return max(1, int(np.floor((n_rows - 1) * CCA_DIM_FRACTION)))


def pca_scores(x: np.ndarray, variance: float = PCA_VARIANCE,
               max_components: int | None = None) -> Tuple[np.ndarray, int]:
    # x is already centered. Scores U*S preserve the retained sample-space geometry.
    # Exact sample-side eigendecomposition of X X^T is far cheaper when d >> n.
    gram = x @ x.T
    eig, u = np.linalg.eigh((gram + gram.T) * 0.5)
    order = np.argsort(eig)[::-1]
    eig, u = np.maximum(eig[order], 0), u[:, order]
    total = float(eig.sum())
    if total <= EPS:
        return np.zeros((x.shape[0], 1), dtype=np.float64), 1
    k = int(np.searchsorted(np.cumsum(eig) / total, variance, side="left") + 1)
    k = max(1, min(k, len(eig)))
    if max_components is not None:
        k = min(k, max(1, int(max_components)))
    return u[:, :k] * np.sqrt(eig[:k]), k


def _invsqrt_psd(c: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    vals, vecs = np.linalg.eigh((c + c.T) * 0.5)
    keep = vals > eps
    if not np.any(keep):
        return np.zeros_like(c)
    return (vecs[:, keep] / np.sqrt(vals[keep])) @ vecs[:, keep].T


def cca_from_scores(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # x,y are centered PCA score matrices [n,kx]/[n,ky]
    n = x.shape[0]
    denom = max(n - 1, 1)
    cxx = (x.T @ x) / denom
    cyy = (y.T @ y) / denom
    cxy = (x.T @ y) / denom
    wx = _invsqrt_psd(cxx)
    wy = _invsqrt_psd(cyy)
    t = wx @ cxy @ wy
    u, s, vt = np.linalg.svd(t, full_matrices=False)
    s = np.clip(s, 0.0, 1.0)
    ax = wx @ u
    ay = wy @ vt.T
    return s, ax, ay


def svcca_similarity(x: np.ndarray, y: np.ndarray, variance: float = PCA_VARIANCE) -> Tuple[float, int, int]:
    cap = cca_dim_cap(x.shape[0])
    xs, kx = pca_scores(x, variance, max_components=cap)
    ys, ky = pca_scores(y, variance, max_components=cap)
    corrs, _, _ = cca_from_scores(xs, ys)
    if corrs.size == 0:
        return float("nan"), kx, ky
    return float(np.mean(corrs)), kx, ky


def pwcca_similarity(x: np.ndarray, y: np.ndarray, variance: float = PCA_VARIANCE) -> Tuple[float, int, int]:
    cap = cca_dim_cap(x.shape[0])
    xs, kx = pca_scores(x, variance, max_components=cap)
    ys, ky = pca_scores(y, variance, max_components=cap)
    corrs, ax, _ = cca_from_scores(xs, ys)
    if corrs.size == 0:
        return float("nan"), kx, ky

    # Projection-weighted CCA: weight canonical X variates by how much of the
    # retained representation they explain. This follows the PWCCA principle;
    # weights are determined from X only and never from the strong-RD target.
    h = xs @ ax[:, : len(corrs)]
    if h.ndim == 1:
        h = h[:, None]
    q, _ = np.linalg.qr(h)
    proj = np.abs(q.T @ xs)
    weights = proj.sum(axis=1)
    if weights.sum() <= EPS:
        return float(np.mean(corrs)), kx, ky
    weights = weights / weights.sum()
    m = min(len(weights), len(corrs))
    return float(np.sum(weights[:m] * corrs[:m]) / np.sum(weights[:m])), kx, ky


def row_correlation_rdm(x: np.ndarray) -> np.ndarray:
    # Correlation distance between rows; return upper triangle vector.
    xc = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(xc, axis=1)
    denom = np.outer(norms, norms)
    sim = np.zeros((x.shape[0], x.shape[0]), dtype=np.float64)
    valid = denom > EPS
    sim[valid] = (xc @ xc.T)[valid] / denom[valid]
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    iu = np.triu_indices(x.shape[0], 1)
    return dist[iu]


def rsa_response(x: np.ndarray, y: np.ndarray) -> float:
    a = row_correlation_rdm(x)
    b = row_correlation_rdm(y)
    if a.size < 2:
        return float("nan")
    rho = spearmanr(a, b).statistic
    if not math.isfinite(float(rho)):
        return float("nan")
    return float(1.0 - np.clip(rho, -1.0, 1.0))


def compute_measures(z0: np.ndarray, zk: np.ndarray) -> Dict[str, float]:
    x, y = paired_center(z0, zk)
    cka = linear_cka_similarity(x, y)
    sv, kx_sv, ky_sv = svcca_similarity(x, y)
    pw, kx_pw, ky_pw = pwcca_similarity(x, y)
    rsa = rsa_response(x, y)
    return {
        "resp_1mcka": 1.0 - cka,
        "resp_1msvcca": 1.0 - sv if math.isfinite(sv) else float("nan"),
        "resp_rsa_distance": rsa,
        "resp_1mpwcca": 1.0 - pw if math.isfinite(pw) else float("nan"),
        "svcca_kx": kx_sv,
        "svcca_ky": ky_sv,
        "pwcca_kx": kx_pw,
        "pwcca_ky": ky_pw,
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "pca_variance": float(PCA_VARIANCE),
        "cca_dim_cap": int(cca_dim_cap(x.shape[0])),
    }


