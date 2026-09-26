"""Training-only graph preprocessing for paper-reimplemented baselines.

This module intentionally separates *paper-specified* ingredients from the few
estimator details that the publications leave unspecified.  The latter are
saved in cache metadata so experiments remain auditable.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import hashlib
import json
import math
import numpy as np

from ...config import PROJECT_ROOT

CACHE_ROOT = PROJECT_ROOT / 'cache' / 'paper_reimplemented_graphs'


def _adj_hash(adj: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(adj, dtype=np.float32))
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]


def _uniform_indices(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, cap).round().astype(np.int64)


def _transitions(train_x: np.ndarray, max_samples: int):
    """Use two last *input* steps only; never uses validation/test/future targets."""
    if train_x.ndim != 4 or train_x.shape[1] < 2:
        raise ValueError(f'Need train_x [S,T,N,1] with T>=2, got {train_x.shape}')
    ix = _uniform_indices(len(train_x), int(max_samples))
    prev = np.asarray(train_x[ix, -2, :, 0], dtype=np.float32)
    curr = np.asarray(train_x[ix, -1, :, 0], dtype=np.float32)
    return prev, curr


def _row_max_normalize(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    m = np.max(np.abs(y), axis=1, keepdims=True)
    good = m[:, 0] > 1e-12
    y[good] = y[good] / m[good]
    y[~good] = 0.0
    return y


def _bfs_nodes(adj_bool: np.ndarray, root: int, k_hop: int, limit: int) -> list[int]:
    """Nearest nodes in the undirected support, deterministic BFS order."""
    n = adj_bool.shape[0]
    und = np.logical_or(adj_bool, adj_bool.T)
    seen = {int(root)}
    out = [int(root)]
    q = deque([(int(root), 0)])
    while q and len(out) < limit:
        u, d = q.popleft()
        if d >= k_hop:
            continue
        neigh = np.flatnonzero(und[u])
        for v in neigh.tolist():
            if v in seen:
                continue
            seen.add(v); out.append(v); q.append((v, d + 1))
            if len(out) >= limit:
                break
    return out


def _cache_file(kind: str, dataset: str, params: dict, adj: np.ndarray) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    safe = dataset.replace('/', '_')
    key_obj = dict(params)
    key_obj['adj_hash'] = _adj_hash(adj)
    key = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    d = CACHE_ROOT / kind / safe
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{kind}_{key}.npz'


def build_causalgrit_prior(
    dataset: str,
    train_x: np.ndarray,
    adj: np.ndarray,
    scaler,
    *,
    rf_trees: int = 32,
    rf_max_samples: int = 4096,
    rf_eval_samples: int = 256,
    k_hop: int = 2,
    max_adjustment_nodes: int = 32,
    force_rebuild: bool = False,
    seed: int = 2026,
    verbose: bool = True,
) -> tuple[np.ndarray, Path, dict]:
    """Paper-faithful scalable CATE prior for CausalGRIT.

    Paper-specified:
      * treatment candidates respect physical graph connectivity;
      * adjustment variables are restricted to a k-hop neighborhood;
      * nonlinear backdoor regression uses Random Forest;
      * speed CATE compares do(X=70) against do(X=0).

    Publication-unspecified choices made explicit here:
      * one RF per outcome node (rather than per edge) for scalability;
      * 32 trees, max depth 8, min leaf 5 by default;
      * empirical 5/95% interventions for non-speed benchmark datasets;
      * absolute CATE is row-max normalized to form nonnegative edge strength.
    """
    from sklearn.ensemble import RandomForestRegressor

    adj = np.asarray(adj, dtype=np.float32)
    params = dict(rf_trees=int(rf_trees), rf_max_samples=int(rf_max_samples),
                  rf_eval_samples=int(rf_eval_samples), k_hop=int(k_hop),
                  max_adjustment_nodes=int(max_adjustment_nodes), seed=int(seed),
                  estimator='RandomForestRegressor_one_per_outcome',
                  speed_intervention='0_to_70_mph', non_speed_intervention='q05_to_q95',
                  cate_postprocess='absolute_row_max_normalization')
    path = _cache_file('causalgrit', dataset, params, adj)
    if path.exists() and not force_rebuild:
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z['meta'].item()))
        return z['graph'].astype(np.float32), path, meta

    prev, curr = _transitions(train_x, rf_max_samples)
    n_samples, n = prev.shape
    support = np.abs(adj) > 1e-12
    np.fill_diagonal(support, False)
    graph = np.zeros((n, n), dtype=np.float32)  # row=outcome/target, col=treatment/source

    speed_dataset = dataset in ('METR-LA', 'PEMS-BAY')
    if speed_dataset:
        low_global = float(scaler.raw_value_to_normalized(0.0))
        high_global = float(scaler.raw_value_to_normalized(70.0))
    else:
        low_global = high_global = None

    eval_ix = _uniform_indices(n_samples, min(int(rf_eval_samples), n_samples))
    report_every = max(1, n // 10)
    if verbose:
        print(f'[CausalGRIT graph] building RF-CATE prior for {dataset}: N={n}, samples={n_samples}, trees={rf_trees}', flush=True)

    for target in range(n):
        sources = np.flatnonzero(support[target])
        if len(sources) == 0:
            continue
        nodes = _bfs_nodes(support, target, int(k_hop), int(max_adjustment_nodes))
        # A treatment must be available to the regression even if BFS limit clipped it.
        for s in sources.tolist():
            if s not in nodes:
                if len(nodes) < max_adjustment_nodes:
                    nodes.append(int(s))
        nodes = nodes[:max_adjustment_nodes]
        node_to_col = {v: i for i, v in enumerate(nodes)}
        sources = np.asarray([s for s in sources if int(s) in node_to_col], dtype=np.int64)
        if len(sources) == 0:
            continue

        X = prev[:, nodes]
        y = curr[:, target]
        rf = RandomForestRegressor(
            n_estimators=int(rf_trees), max_depth=8, min_samples_leaf=5,
            max_features=1.0, bootstrap=True, n_jobs=1, random_state=int(seed + target),
        )
        rf.fit(X, y)
        Xe = X[eval_ix].copy()
        for source in sources.tolist():
            col = node_to_col[int(source)]
            if speed_dataset:
                lo, hi = low_global, high_global
            else:
                vals = prev[:, source]
                lo, hi = np.quantile(vals, [0.05, 0.95]).astype(np.float64)
                lo, hi = float(lo), float(hi)
            Xlo = Xe.copy(); Xhi = Xe.copy()
            Xlo[:, col] = lo; Xhi[:, col] = hi
            cate = float(np.mean(rf.predict(Xhi) - rf.predict(Xlo)))
            graph[target, source] = abs(cate)

        if verbose and ((target + 1) % report_every == 0 or target + 1 == n):
            print(f'[CausalGRIT graph] {target+1}/{n} outcome nodes', flush=True)

    graph = _row_max_normalize(graph)
    meta = dict(params)
    meta.update(dataset=dataset, n_nodes=int(n), n_samples=int(n_samples),
                nonzero=int(np.count_nonzero(graph)), cache_version=1,
                note='Paper-faithful reimplementation; publication does not specify RF hyperparameters or non-speed intervention values.')
    np.savez_compressed(path, graph=graph, meta=np.array(json.dumps(meta)))
    if verbose:
        print(f'[CausalGRIT graph] saved -> {path}', flush=True)
    return graph, path, meta


def _quantile_discretize_pair(prev: np.ndarray, curr: np.ndarray, bins: int):
    S, N = prev.shape
    dp = np.empty((S, N), dtype=np.int16)
    dc = np.empty((S, N), dtype=np.int16)
    qs = np.linspace(0.0, 1.0, int(bins) + 1)[1:-1]
    for j in range(N):
        vals = np.concatenate([prev[:, j], curr[:, j]])
        edges = np.unique(np.quantile(vals, qs))
        dp[:, j] = np.digitize(prev[:, j], edges, right=False)
        dc[:, j] = np.digitize(curr[:, j], edges, right=False)
    return dp, dc


def _discrete_te(source_prev: np.ndarray, target_prev: np.ndarray, target_curr: np.ndarray, bins: int) -> float:
    """I(source_{t-1}; target_t | target_{t-1}) discrete plug-in estimator."""
    b = int(bins)
    s = np.asarray(source_prev, dtype=np.int64)
    p = np.asarray(target_prev, dtype=np.int64)
    y = np.asarray(target_curr, dtype=np.int64)
    n = len(y)
    if n == 0:
        return 0.0
    idx3 = (y * b + p) * b + s
    c3 = np.bincount(idx3, minlength=b*b*b).reshape(b, b, b).astype(np.float64)
    cyp = c3.sum(axis=2)          # y,p
    cps = c3.sum(axis=0)          # p,s
    cp = cyp.sum(axis=0)          # p
    nz = c3 > 0
    yy, pp, ss = np.nonzero(nz)
    cnt = c3[yy, pp, ss]
    # p(y|p,s) / p(y|p) = [c(y,p,s)/c(p,s)] / [c(y,p)/c(p)]
    ratio = (cnt * cp[pp] + 1e-12) / (cps[pp, ss] * cyp[yy, pp] + 1e-12)
    return float(np.sum((cnt / n) * np.log(np.clip(ratio, 1e-12, None))))


def build_tfmvgnn_te_graph(
    dataset: str,
    train_x: np.ndarray,
    adj: np.ndarray,
    *,
    te_bins: int = 8,
    te_threshold: float = 0.6,
    te_max_samples: int = 8192,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, Path, dict]:
    """Training-only TF-MVGNN transfer-entropy graph.

    The paper gives the TE equation and threshold th=0.6 but not the discrete
    estimator/binning nor history-order used in the estimator.  We therefore use
    the standard one-lag discrete TE I(X_j,t-1; X_i,t | X_i,t-1), quantile bins,
    and row-max normalization before applying th=0.6.  TE is computed only on
    physical edges because A_C = A_R ⊙ A_TE makes all other entries irrelevant.
    """
    adj = np.asarray(adj, dtype=np.float32)
    params = dict(te_bins=int(te_bins), te_threshold=float(te_threshold),
                  te_max_samples=int(te_max_samples), te_history=1,
                  estimator='discrete_conditional_mutual_information_quantile_bins',
                  normalization='row_max_before_threshold')
    path = _cache_file('tfmvgnn', dataset, params, adj)
    if path.exists() and not force_rebuild:
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z['meta'].item()))
        return z['graph'].astype(np.float32), path, meta

    prev, curr = _transitions(train_x, te_max_samples)
    dp, dc = _quantile_discretize_pair(prev, curr, int(te_bins))
    n = prev.shape[1]
    support = np.abs(adj) > 1e-12
    np.fill_diagonal(support, False)
    raw = np.zeros((n, n), dtype=np.float32)  # target,row <- source,col
    edges = np.argwhere(support)
    if verbose:
        print(f'[TF-MVGNN TE] building graph for {dataset}: N={n}, physical_edges={len(edges)}, samples={len(prev)}', flush=True)
    report_every = max(1, len(edges) // 10)
    for ei, (target, source) in enumerate(edges.tolist()):
        raw[target, source] = max(0.0, _discrete_te(dp[:, source], dp[:, target], dc[:, target], int(te_bins)))
        if verbose and ((ei + 1) % report_every == 0 or ei + 1 == len(edges)):
            print(f'[TF-MVGNN TE] {ei+1}/{len(edges)} edges', flush=True)
    score = _row_max_normalize(raw)
    ate = np.where(score >= float(te_threshold), score, 0.0).astype(np.float32)
    meta = dict(params)
    meta.update(dataset=dataset, n_nodes=int(n), n_samples=int(len(prev)),
                physical_edges=int(len(edges)), kept_edges=int(np.count_nonzero(ate)), cache_version=1,
                note='Paper-faithful reimplementation; publication specifies th=0.6 but not TE estimator/binning/history order.')
    np.savez_compressed(path, graph=ate, raw_te=raw, meta=np.array(json.dumps(meta)))
    if verbose:
        print(f'[TF-MVGNN TE] saved -> {path}', flush=True)
    return ate, path, meta
