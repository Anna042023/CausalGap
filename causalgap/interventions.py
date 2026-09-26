from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import networkx as nx


# Paper-aligned structural intervention protocol (Appendix A.1.1).
EPS_A = 1e-6
DELTA_A = 0.01
PAPER_EDGE_REMOVAL_LEVELS = (0.05, 0.10, 0.20)
PAPER_WEIGHT_GAMMAS = (0.8, 0.5, 0.2)  # weak -> strong; strength = 1-gamma
PAPER_NODE_ISOLATION_LEVELS = (0.10, 0.20, 0.40, 0.60)
PAPER_INTERVENTION_SEEDS = tuple(range(20))

# node_masking is retained only as a backwards-compatible CLI/file alias.  In the
# formal protocol it is implemented as *structural node isolation*: A_k=M_k A_0 M_k,
# X_k=X_0, and errors are evaluated on unisolated nodes.
GRAPH_INTERVENTIONS = {'edge_removal', 'weight_modification', 'node_isolation', 'node_masking'}
INTERVENTION_TYPES = ['edge_removal', 'weight_modification', 'node_isolation', 'time_shift']
INTERVENTION_FAMILY = {
    'edge_removal': 'structural',
    'weight_modification': 'structural',
    'node_isolation': 'structural',
    'node_masking': 'structural',
    'time_shift': 'temporal',
}


@dataclass
class Intervention:
    kind: str
    family: str
    x: np.ndarray
    y: np.ndarray
    y_mask: np.ndarray | None
    adj: np.ndarray
    severity: float
    meta: dict
    environment_distance: float
    distance_kind: str
    environment_source: str = 'csg_input'
    causal_valid: bool = True
    leakage_detected: bool = False
    # Indices in the ORIGINAL baseline test set whose targets are being evaluated.
    baseline_indices: np.ndarray | None = None
    # For temporal substitution, source input-window indices in the original test set.
    source_indices: np.ndarray | None = None


def intervention_family(kind: str, environment_source: str = 'csg_input') -> str:
    if environment_source == 'external_physical':
        return 'physical'
    return INTERVENTION_FAMILY.get(kind, 'other')


def is_symmetric_graph(adj, atol=1e-6):
    return bool(np.allclose(adj, adj.T, atol=atol, rtol=0.0))


def _adjacency_digest(adj: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(adj, dtype=np.float32))
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


_BETWEENNESS_CACHE = {}


def _graph_for_betweenness(adj):
    """Build the A0 graph used once for edge-betweenness ranking.

    Traffic adjacencies in this benchmark are affinities, so NetworkX's shortest
    path length is c_ij = 1/(A_ij + 1e-6), exactly matching the manuscript rule.
    """
    a = np.asarray(adj, dtype=np.float64)
    symmetric = is_symmetric_graph(a)
    G = nx.Graph() if symmetric else nx.DiGraph()
    G.add_nodes_from(range(a.shape[0]))
    if symmetric:
        edges = np.argwhere(np.triu(a != 0, k=1))
    else:
        edges = np.argwhere(a != 0)
    for i, j in edges:
        w = float(a[int(i), int(j)])
        # The benchmark road graphs use non-negative affinities.  abs keeps the
        # path metric well-defined for any legacy signed matrix without changing
        # the intervention weights themselves.
        affinity = max(abs(w), EPS_A)
        G.add_edge(int(i), int(j), distance=1.0 / (affinity + EPS_A))
    return G, symmetric


def _rank_edges_by_betweenness(adj):
    """Rank A0 edges once and reuse the ranking for all intensity/seed draws."""
    key = _adjacency_digest(adj)
    cached = _BETWEENNESS_CACHE.get(key)
    if cached is not None:
        return cached
    G, symmetric = _graph_for_betweenness(adj)
    scores = nx.edge_betweenness_centrality(G, normalized=True, weight='distance')
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    _BETWEENNESS_CACHE[key] = (ranked, symmetric)
    return ranked, symmetric


def _apply_edge_removal(adj, rng, ratio=0.10, mode='paper_top40_uniform', pool_ratio=0.40):
    ranked, symmetric = _rank_edges_by_betweenness(adj)
    k = max(1, int(round(len(ranked) * float(ratio)))) if ranked else 0
    out = np.asarray(adj).copy()
    removed = []

    if mode == 'paper_top40_uniform':
        # Appendix A.1.1: sample without replacement from the top 40% candidate
        # pool.  Sampling is uniform inside that pool; betweenness only defines
        # membership/rank of the candidate pool.
        pool_ratio = 0.40
        pool_k = max(k, int(np.ceil(len(ranked) * pool_ratio)))
        pool = ranked[:min(pool_k, len(ranked))]
        k = min(k, len(pool))
        ids = rng.choice(len(pool), size=k, replace=False) if k else []
        chosen = [pool[int(i)] for i in sorted(ids)]
    elif mode == 'strict_topk':
        # Legacy/debug mode retained for compatibility; not used by paper runs.
        chosen = ranked[:k]
    elif mode in ('stochastic_pool', 'centrality_weighted_pool'):
        # Legacy modes retained so old command lines still parse.  Formal paper
        # execution uses paper_top40_uniform.
        pool_k = max(k, int(round(len(ranked) * float(pool_ratio))))
        pool = ranked[:min(pool_k, len(ranked))]
        k = min(k, len(pool))
        if k:
            if mode == 'centrality_weighted_pool':
                scores = np.asarray([max(float(item[1]), 0.0) for item in pool], dtype=np.float64)
                scores = scores + max(1e-12, float(scores.max()) * 1e-6 if len(scores) else 1e-12)
                ids = rng.choice(len(pool), size=k, replace=False, p=scores / scores.sum())
            else:
                ids = rng.choice(len(pool), size=k, replace=False)
        else:
            ids = []
        chosen = [pool[int(i)] for i in sorted(ids)]
    else:
        raise ValueError(f'Unknown edge_removal mode: {mode}')

    for (i, j), score in chosen:
        out[i, j] = 0.0
        if symmetric:
            out[j, i] = 0.0
        removed.append((int(i), int(j), float(score)))
    return out, removed, symmetric


def _candidate_edges(adj, symmetric):
    if symmetric:
        return np.argwhere(np.triu(np.asarray(adj) != 0, k=1))
    return np.argwhere(np.asarray(adj) != 0)


def _apply_weight_modification(adj, rng, ratio=0.15, gamma=0.8):
    """Multiply one uniformly sampled 15% edge set by a fixed gamma level."""
    symmetric = is_symmetric_graph(adj)
    edges = _candidate_edges(adj, symmetric)
    k = max(1, int(round(len(edges) * float(ratio)))) if len(edges) else 0
    ids = rng.choice(len(edges), size=k, replace=False) if k else []
    out = np.asarray(adj).copy()
    modified = []
    gamma = float(gamma)
    for idx in ids:
        i, j = map(int, edges[int(idx)])
        out[i, j] *= gamma
        if symmetric:
            out[j, i] *= gamma
        modified.append((i, j, gamma))
    return out, modified, symmetric


def _apply_node_isolation(adj, rng, ratio):
    """A_k=M_k A_0 M_k: zero every in/out edge of the sampled nodes."""
    n = int(np.asarray(adj).shape[0])
    k = max(1, int(round(n * float(ratio)))) if n else 0
    nodes = np.asarray(rng.choice(n, size=k, replace=False), dtype=np.int64) if k else np.array([], dtype=np.int64)
    out = np.asarray(adj).copy()
    if len(nodes):
        out[nodes, :] = 0.0
        out[:, nodes] = 0.0
    return out, nodes


def structural_distance(a0, ak, kind, isolated_nodes=None, eps_a=EPS_A):
    """Paper d_T,k for a structural intervention."""
    canonical = 'node_isolation' if kind == 'node_masking' else kind
    if canonical == 'node_isolation':
        if isolated_nodes is None:
            # Infer structurally isolated nodes when possible (all incident edges
            # present in A0 are removed in Ak).
            a0a = np.asarray(a0)
            aka = np.asarray(ak)
            incident0 = np.any(a0a != 0, axis=0) | np.any(a0a != 0, axis=1)
            incidentk = np.any(aka != 0, axis=0) | np.any(aka != 0, axis=1)
            isolated_nodes = np.flatnonzero(incident0 & ~incidentk)
        return float(len(np.asarray(isolated_nodes).reshape(-1)) / max(int(np.asarray(a0).shape[0]), 1))
    a0a = np.asarray(a0, dtype=np.float64)
    aka = np.asarray(ak, dtype=np.float64)
    return float(np.linalg.norm(aka - a0a) / (np.linalg.norm(a0a) + float(eps_a)))


def validate_no_future_leakage(source_indices, target_origin_indices):
    src = np.asarray(source_indices, dtype=np.int64)
    origin = np.asarray(target_origin_indices, dtype=np.int64)
    if src.shape != origin.shape:
        raise ValueError(f'source/origin shape mismatch: {src.shape} vs {origin.shape}')
    delta = src - origin
    leak_mask = delta > 0
    leak_count = int(np.count_nonzero(leak_mask))
    result = {
        'passed': leak_count == 0,
        'future_source_count': leak_count,
        'max_source_minus_origin': int(delta.max()) if len(delta) else 0,
        'min_source_minus_origin': int(delta.min()) if len(delta) else 0,
    }
    if leak_count:
        raise ValueError(
            'Temporal intervention failed causal-order check: '
            f'{leak_count} source samples are later than target origins; '
            f'max delta={result["max_source_minus_origin"]}.'
        )
    return result


def _aligned_past_time_shift(x, y, lag_steps, max_steps):
    """Leakage-safe observational time-shift control (not a structural intervention)."""
    lag_steps = int(lag_steps)
    max_steps = int(max_steps)
    if lag_steps <= 0 or lag_steps > max_steps:
        raise ValueError(f'lag_steps must be in [1,{max_steps}], got {lag_steps}')
    n = len(x)
    if n <= max_steps:
        raise ValueError(f'Not enough test samples ({n}) for max temporal lag {max_steps}.')
    target_idx = np.arange(max_steps, n, dtype=np.int64)
    src_idx = target_idx - lag_steps
    check = validate_no_future_leakage(src_idx, target_idx)
    return x[src_idx].copy(), y[target_idx].copy(), src_idx, target_idx, check


def make_intervention(kind, x, y, adj, seed=0, *, y_mask=None, scaler=None, interval_minutes=5,
                      severity=None, weight_gamma=None,
                      edge_removal_mode='paper_top40_uniform', edge_pool_ratio=0.40,
                      node_mask_fill='mean', time_shift_mode='native_steps',
                      time_shift_max_steps=12, time_shift_max_minutes=60,
                      noise_ratio=0.05):
    """Create one intervention using an intervention seed independent of model seed.

    `node_mask_fill` is accepted only for backwards-compatible command lines; the
    paper-aligned node intervention never changes X and therefore ignores it.
    """
    rng = np.random.default_rng(int(seed))
    xk, yk, ak = np.asarray(x).copy(), np.asarray(y).copy(), np.asarray(adj).copy()
    if y_mask is None:
        ymk = np.ones_like(y, dtype=bool)
    else:
        ymk = np.asarray(y_mask, dtype=bool).copy()
        if ymk.shape != np.asarray(y).shape:
            raise ValueError(f'y_mask shape {ymk.shape} != y shape {np.asarray(y).shape}')
    meta = {'seed': int(seed), 'intervention_sampling_seed': int(seed)}
    source = 'csg_input'
    canonical_kind = 'node_isolation' if kind == 'node_masking' else kind
    family = intervention_family(canonical_kind, source)
    causal_valid = True
    leakage_detected = False
    baseline_indices = np.arange(len(x), dtype=np.int64)
    source_indices = baseline_indices.copy()

    if canonical_kind == 'edge_removal':
        ratio = 0.10 if severity is None else float(severity)
        if not (0.0 < ratio < 1.0):
            raise ValueError(f'edge-removal severity must be in (0,1), got {ratio}')
        ak, removed, symmetric = _apply_edge_removal(
            ak, rng, ratio=ratio, mode=edge_removal_mode, pool_ratio=edge_pool_ratio
        )
        sev = ratio
        distance = structural_distance(adj, ak, canonical_kind)
        distance_kind = 'normalized_graph_frobenius'
        meta.update({
            'removed_edges': removed,
            'graph_symmetric': symmetric,
            'edge_removal_fraction': float(ratio),
            'edge_removal_mode': edge_removal_mode,
            'edge_pool_ratio': 0.40 if edge_removal_mode == 'paper_top40_uniform' else float(edge_pool_ratio),
            'candidate_pool_definition': 'top_40_percent_edge_betweenness' if edge_removal_mode == 'paper_top40_uniform' else 'legacy',
            'betweenness_distance': '1/(A_ij+1e-6)',
            'intervention_signature': '|'.join(f'{i}-{j}' for i, j, _ in removed),
        })

    elif canonical_kind == 'weight_modification':
        gamma = 0.5 if weight_gamma is None else float(weight_gamma)
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f'weight gamma must be in (0,1], got {gamma}')
        ak, factors, symmetric = _apply_weight_modification(ak, rng, ratio=0.15, gamma=gamma)
        sev = 1.0 - gamma
        distance = structural_distance(adj, ak, canonical_kind)
        distance_kind = 'normalized_graph_frobenius'
        meta.update({
            'modified_edges': factors,
            'graph_symmetric': symmetric,
            'modified_edge_fraction': 0.15,
            'gamma': float(gamma),
            'intervention_strength': float(1.0 - gamma),
            'intervention_signature': '|'.join(f'{i}-{j}' for i, j, _ in factors),
        })

    elif canonical_kind == 'node_isolation':
        ratio = 0.20 if severity is None else float(severity)
        if not (0.0 < ratio < 1.0):
            raise ValueError(f'node-isolation severity must be in (0,1), got {ratio}')
        ak, nodes = _apply_node_isolation(ak, rng, ratio)
        # X_k = X_0.  Y is unchanged; only evaluation support excludes isolated nodes.
        if len(nodes):
            ymk[:, :, nodes, :] = False
        actual_ratio = float(len(nodes) / max(xk.shape[2], 1))
        sev = actual_ratio
        distance = structural_distance(adj, ak, canonical_kind, isolated_nodes=nodes)
        distance_kind = 'isolated_node_fraction'
        meta.update({
            'isolated_nodes': [int(v) for v in nodes],
            'requested_isolated_node_fraction': float(ratio),
            'isolated_node_fraction': actual_ratio,
            'node_observations_changed': False,
            'node_mask_fill_legacy_arg_ignored': str(node_mask_fill),
            'evaluation_support': 'unisolated_nodes_only_for_both_IPE0_and_IPEk',
            'intervention_signature': '|'.join(str(int(v)) for v in sorted(nodes.tolist())),
        })

    elif canonical_kind == 'time_shift':
        # Observational-shift control retained as in the current code.  It is not
        # included in the structural intervention family.
        if time_shift_mode == 'native_steps':
            max_steps = int(max(1, time_shift_max_steps))
        elif time_shift_mode == 'physical_minutes':
            max_steps = int(max(1, round(float(time_shift_max_minutes) / float(interval_minutes))))
        else:
            raise ValueError(f'Unknown time_shift_mode: {time_shift_mode}')

        lag_steps = int(rng.integers(1, max_steps + 1))
        xk, yk, source_indices, baseline_indices, causal_check = _aligned_past_time_shift(
            x, y, lag_steps, max_steps
        )
        ymk = ymk[baseline_indices].copy()
        sigma = float(np.std(xk)) * float(noise_ratio)
        if sigma > 0:
            xk = xk + rng.normal(0.0, sigma, size=xk.shape).astype(np.float32)

        shift = -lag_steps
        sev = lag_steps / float(max_steps)
        distance = sev
        distance_kind = 'normalized_past_temporal_lag'
        meta.update({
            'shift_steps': int(shift),
            'lag_steps': int(lag_steps),
            'shift_minutes': int(shift * interval_minutes),
            'lag_minutes': int(lag_steps * interval_minutes),
            'max_shift_steps': int(max_steps),
            'time_shift_mode': time_shift_mode,
            'time_shift_direction': 'past_only',
            'noise_ratio': float(noise_ratio),
            'noise_sigma_normalized': float(sigma),
            'boundary_policy': 'drop_common_invalid_prefix',
            'alignment_policy': 'common_valid_aligned_target_subset',
            'common_target_start_index': int(max_steps),
            'common_target_stop_index_exclusive': int(len(x)),
            'dropped_prefix_samples': int(max_steps),
            'aligned_evaluation_samples': int(len(baseline_indices)),
            'causal_order_check_passed': bool(causal_check['passed']),
            'future_source_count': int(causal_check['future_source_count']),
            'max_source_minus_origin': int(causal_check['max_source_minus_origin']),
            'min_source_minus_origin': int(causal_check['min_source_minus_origin']),
            'source_index_first': int(source_indices[0]),
            'source_index_last': int(source_indices[-1]),
            'target_index_first': int(baseline_indices[0]),
            'target_index_last': int(baseline_indices[-1]),
            'protocol_note': 'observational-shift control; not a structural intervention',
        })
        causal_valid = bool(causal_check['passed'])
        leakage_detected = not causal_valid

    else:
        raise ValueError(f'Unknown intervention: {kind}')

    if family == 'structural' and float(distance) < DELTA_A:
        raise ValueError(
            f'Generated structural intervention has d_T={distance:.6g} < delta_A={DELTA_A}; '
            'paper protocol evaluates only nontrivial interventions.'
        )

    return Intervention(
        kind=canonical_kind,
        family=family,
        x=np.asarray(xk, dtype=np.float32),
        y=np.asarray(yk, dtype=np.float32),
        y_mask=np.asarray(ymk, dtype=bool),
        adj=np.asarray(ak, dtype=np.float32),
        severity=float(sev),
        meta=meta,
        environment_distance=float(distance),
        distance_kind=distance_kind,
        environment_source=source,
        causal_valid=bool(causal_valid),
        leakage_detected=bool(leakage_detected),
        baseline_indices=np.asarray(baseline_indices, dtype=np.int64),
        source_indices=np.asarray(source_indices, dtype=np.int64),
    )


def load_external_intervention(path, bundle, kind=None):
    """Load physically generated aligned Xk/Yk/Ak (SUMO, synthetic, I-405, etc.)."""
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    for key in ('x', 'y', 'adj'):
        if key not in z.files:
            raise ValueError(f'{path} missing required key {key}; keys={z.files}')
    x = np.asarray(z['x'], dtype=np.float32)
    y = np.asarray(z['y'], dtype=np.float32)
    adj = np.asarray(z['adj'], dtype=np.float32)
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(f'External x/y must be [B,T,N,F], got {x.shape}, {y.shape}')
    if adj.shape != bundle.adj.shape:
        raise ValueError(f'External adjacency shape {adj.shape} != baseline {bundle.adj.shape}')

    sidecar = path.with_suffix('.json')
    meta = {'external_file': str(path), 'external_units': 'original_units_then_normalized_by_bundle_scaler'}
    if sidecar.exists():
        try:
            meta.update(json.loads(sidecar.read_text(encoding='utf-8')))
        except Exception:
            pass

    b_start = int(meta.get('baseline_start', 0))
    b_stop = int(meta.get('baseline_stop', bundle.test_x.shape[0]))
    baseline_indices = np.arange(b_start, b_stop, dtype=np.int64)
    if len(x) != len(baseline_indices):
        raise ValueError(
            f'External environment has {len(x)} samples, but aligned baseline range '
            f'[{b_start},{b_stop}) has {len(baseline_indices)}. Fix sidecar alignment.'
        )

    external_y_mask = bundle.infer_external_target_mask(y)
    x = bundle.scaler.transform(x).astype(np.float32)
    y = bundle.scaler.transform(y).astype(np.float32)
    inferred_kind = kind or (str(z['kind'].item()) if 'kind' in z.files else path.stem.split('_000')[0])
    inferred_kind = 'node_isolation' if inferred_kind == 'node_masking' else inferred_kind
    generator = str(meta.get('generator', 'UNSPECIFIED'))
    if generator.upper().startswith('FORMAT_TEMPLATE'):
        raise ValueError(f'{path} is only a format template and cannot be used for paper evaluation.')
    causal_valid = bool(meta.get('causal_valid', True))
    leakage_detected = bool(meta.get('leakage_detected', False))
    if leakage_detected:
        causal_valid = False

    if inferred_kind == 'node_isolation':
        isolated = meta.get('isolated_nodes', [])
        sd = structural_distance(bundle.adj, adj, inferred_kind, isolated_nodes=isolated)
        distance_kind = 'isolated_node_fraction'
        if isolated:
            external_y_mask[:, :, np.asarray(isolated, dtype=np.int64), :] = False
    else:
        sd = structural_distance(bundle.adj, adj, inferred_kind)
        distance_kind = 'normalized_graph_frobenius'

    return Intervention(
        kind=inferred_kind,
        family=intervention_family(inferred_kind, 'external_physical'),
        x=x,
        y=y,
        y_mask=np.asarray(external_y_mask, dtype=bool),
        adj=adj.astype(np.float32),
        severity=float(meta.get('severity', sd)),
        meta=meta,
        environment_distance=float(sd if sd > 0 else meta.get('environment_distance', 1.0)),
        distance_kind=str(meta.get('distance_kind', distance_kind)),
        environment_source='external_physical',
        causal_valid=causal_valid,
        leakage_detected=leakage_detected,
        baseline_indices=baseline_indices,
        source_indices=None,
    )
