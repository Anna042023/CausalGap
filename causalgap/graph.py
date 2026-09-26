from pathlib import Path
import pickle
import numpy as np
import pandas as pd


def _square_from_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    preferred = ['adj_mx', 'adj', 'A', 'data']
    for k in preferred + list(z.files):
        if k in z.files:
            a = z[k]
            if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[0] == a.shape[1]:
                return a.astype(np.float32)
    raise ValueError(f'No square adjacency matrix found in {path}. keys={z.files}')


def _adj_from_pickle(path: Path):
    with path.open('rb') as f:
        obj = pickle.load(f, encoding='latin1')
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32)
    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            if isinstance(item, np.ndarray) and item.ndim == 2 and item.shape[0] == item.shape[1]:
                return item.astype(np.float32)
    if isinstance(obj, dict):
        for key in ['adj_mx', 'adj', 'A']:
            if key in obj:
                a = np.asarray(obj[key])
                if a.ndim == 2 and a.shape[0] == a.shape[1]:
                    return a.astype(np.float32)
    raise ValueError(f'Unsupported graph pickle structure: {path}')


def distance_csv_to_adj(path: Path, num_nodes: int, sigma=None, threshold=0.1):
    df = pd.read_csv(path)
    cols = [str(c).lower() for c in df.columns]
    src_col = None
    dst_col = None
    dist_col = None
    for i, c in enumerate(cols):
        if c in ('from', 'src', 'source', 'from_node', 'fromnode'):
            src_col = df.columns[i]
        elif c in ('to', 'dst', 'target', 'to_node', 'tonode'):
            dst_col = df.columns[i]
        elif 'dist' in c or 'cost' in c:
            dist_col = df.columns[i]
    if src_col is None or dst_col is None:
        # ASTGCN distance.csv usually columns: from,to,cost
        src_col, dst_col = df.columns[:2]
        dist_col = df.columns[2] if len(df.columns) >= 3 else None

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    dists = []
    rows = []
    for _, r in df.iterrows():
        i, j = int(r[src_col]), int(r[dst_col])
        if not (0 <= i < num_nodes and 0 <= j < num_nodes):
            continue
        d = float(r[dist_col]) if dist_col is not None else 1.0
        dists.append(d)
        rows.append((i, j, d))
    if not rows:
        raise ValueError(f'No valid graph edges parsed from {path}')
    if sigma is None:
        sigma = float(np.std(dists))
        if sigma < 1e-8:
            sigma = 1.0
    for i, j, d in rows:
        w = float(np.exp(-(d * d) / (sigma * sigma + 1e-12)))
        if w >= threshold:
            A[i, j] = w
            A[j, i] = max(A[j, i], w)
    np.fill_diagonal(A, 0.0)
    return A


def load_graph(dataset_dir: Path, num_nodes: int):
    dataset_dir = Path(dataset_dir)

    # ST-SSL
    p = dataset_dir / 'adj_mx.npz'
    if p.exists():
        return _square_from_npz(p)

    # direct npy/npz/pkl candidates
    candidates = []
    for pattern in ['*.pkl', '*.pickle', '*adj*.npy', '*adj*.npz', '*graph*.npy', '*graph*.npz']:
        candidates += list(dataset_dir.rglob(pattern))
    for p in candidates:
        try:
            if p.suffix in ('.pkl', '.pickle'):
                A = _adj_from_pickle(p)
            elif p.suffix == '.npy':
                A = np.load(p).astype(np.float32)
            else:
                A = _square_from_npz(p)
            if A.shape == (num_nodes, num_nodes):
                return A
        except Exception:
            pass

    # PEMS04/08
    distance_files = list(dataset_dir.rglob('distance.csv')) + list(dataset_dir.rglob('*distance*.csv'))
    for p in distance_files:
        try:
            return distance_csv_to_adj(p, num_nodes)
        except Exception:
            pass

    raise FileNotFoundError(
        f'Cannot locate usable graph for {dataset_dir}. Expected adj_mx.npz, sensor_graph/*.pkl, or distance.csv.'
    )
