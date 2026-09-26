from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx


def _candidate_coordinate_files(dataset_dir: Path):
    pats = [
        '*location*.csv', '*locations*.csv', '*sensor*.csv', '*station*.csv',
        'coords.csv', 'coordinates.csv', '*coord*.csv'
    ]
    seen = set(); out = []
    for pat in pats:
        for p in dataset_dir.rglob(pat):
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp); out.append(p)
    return out


def load_coordinates(dataset_dir: Path, num_nodes: int):
    """Best-effort coordinate discovery without inventing missing locations."""
    dataset_dir = Path(dataset_dir)
    lat_names = {'lat','latitude','y'}
    lon_names = {'lon','lng','longitude','x'}
    id_names = {'sensor_id','sensorid','node_id','nodeid','id','index','station_id'}
    for p in _candidate_coordinate_files(dataset_dir):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        lower = {str(c).strip().lower(): c for c in df.columns}
        lat_col = next((lower[k] for k in lat_names if k in lower), None)
        lon_col = next((lower[k] for k in lon_names if k in lower), None)
        if lat_col is None or lon_col is None:
            continue
        vals = df[[lat_col, lon_col]].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
        if len(vals) == num_nodes and np.isfinite(vals).any():
            mask = np.isfinite(vals).all(axis=1)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            return vals, str(p), mask
        id_col = next((lower[k] for k in id_names if k in lower), None)
        if id_col is not None:
            coords = np.zeros((num_nodes,2), dtype=np.float32)
            mask = np.zeros(num_nodes, dtype=bool)
            ids = pd.to_numeric(df[id_col], errors='coerce')
            for row_i, node_id in enumerate(ids):
                if pd.isna(node_id):
                    continue
                ni = int(node_id)
                if 0 <= ni < num_nodes and np.isfinite(vals[row_i]).all():
                    coords[ni] = vals[row_i]; mask[ni] = True
            if mask.any():
                return coords, str(p), mask
    return np.zeros((num_nodes,2), dtype=np.float32), 'missing_zero_filled', np.zeros(num_nodes, dtype=bool)


def _weighted_graph(adj, eps=1e-8):
    """Create a graph with both affinity and path-cost attributes.

    Traffic adjacency matrices are treated as non-negative affinities.  Path
    distance is therefore inverse affinity.  This makes DSAI descriptors react
    to both edge deletion *and* edge-weight interventions.
    """
    a = np.asarray(adj, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f'adj must be square, got {a.shape}')
    symmetric = np.allclose(a, a.T, atol=1e-6, rtol=0.0)
    G = nx.Graph() if symmetric else nx.DiGraph()
    n = a.shape[0]; G.add_nodes_from(range(n))
    rows, cols = np.nonzero(a)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i == j:
            continue
        affinity = float(abs(a[i,j]))
        if affinity <= eps:
            continue
        # Avoid double insertion for undirected matrices.
        if symmetric and j < i:
            continue
        G.add_edge(i, j, affinity=affinity, distance=1.0/max(affinity, eps))
    return G, symmetric


def topology_features(adj):
    """Weight-aware DSAI descriptors per node.

    Returns [log(1+d_avg^w), log(1+strength), log(1+betweenness^w)].
    Unlike v2.3, these descriptors change when existing edge weights change.
    """
    G, symmetric = _weighted_graph(adj)
    n = G.number_of_nodes()
    avgd = np.zeros(n, dtype=np.float64)
    strength = np.zeros(n, dtype=np.float64)

    # Weighted node strength. For directed graphs use total in+out strength so
    # the descriptor remains comparable with undirected datasets.
    if symmetric:
        for i in range(n):
            strength[i] = float(G.degree(i, weight='affinity'))
    else:
        for i in range(n):
            strength[i] = float(G.in_degree(i, weight='affinity') + G.out_degree(i, weight='affinity'))

    btw = nx.betweenness_centrality(G, normalized=True, weight='distance')
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(G, i, weight='distance')
        vals = [float(d) for j, d in lengths.items() if j != i and np.isfinite(d)]
        avgd[i] = float(np.mean(vals)) if vals else 0.0

    feat = np.stack([
        np.log1p(avgd),
        np.log1p(strength),
        np.log1p(np.array([btw.get(i,0.0) for i in range(n)], dtype=np.float64)),
    ], axis=1)
    return feat.astype(np.float32)


def dsai_raw_encoding(adj, coords=None):
    n = np.asarray(adj).shape[0]
    if coords is None:
        coords = np.zeros((n,2), dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape != (n,2):
        raise ValueError(f'coords must be [N,2], got {coords.shape} for N={n}')
    return np.concatenate([coords, topology_features(adj)], axis=1).astype(np.float32)


def fit_fixed_bn(raw):
    raw = np.asarray(raw, dtype=np.float32)
    mean = raw.mean(axis=0); std = raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_fixed_bn(raw, mean, std):
    return ((np.asarray(raw, dtype=np.float32) - mean) / std).astype(np.float32)


def normalized_adjacency(adj, self_loop=True, eps=1e-8):
    """Row-normalized non-negative affinity matrix for dynamic DSAI messages."""
    a = np.abs(np.asarray(adj, dtype=np.float32)).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f'adj must be square, got {a.shape}')
    if self_loop:
        a = a + np.eye(a.shape[0], dtype=np.float32)
    rowsum = a.sum(axis=1, keepdims=True)
    rowsum = np.where(rowsum < eps, 1.0, rowsum)
    return (a / rowsum).astype(np.float32)
