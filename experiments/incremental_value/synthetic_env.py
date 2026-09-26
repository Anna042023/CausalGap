#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from causalgap.data import load_dataset  # noqa: E402
from causalgap.interventions import make_intervention  # noqa: E402

from protocol import (  # noqa: E402
    DEFAULT_EXPERIMENT_ROOT,
    H1_LEVELS,
    STRUCTURAL_TYPES,
    json_dump,
    normalized_weak_strength,
    safe_name,
)


def row_normalize(adj: np.ndarray) -> np.ndarray:
    a = np.asarray(adj, dtype=np.float32)
    denom = np.sum(np.abs(a), axis=1, keepdims=True)
    out = np.zeros_like(a)
    good = denom[:, 0] > 1e-12
    out[good] = a[good] / denom[good]
    return out


def simulate(A: np.ndarray, x_init: np.ndarray, u: np.ndarray, eps: np.ndarray,
             beta1: float = 0.7, beta2: float = 0.2) -> np.ndarray:
    """Paper nonlinear graph dynamics with fixed exogenous realization."""
    Abar = row_normalize(A)
    x = np.asarray(x_init, dtype=np.float32).copy()
    T, N = u.shape
    out = np.empty((T, N), dtype=np.float32)
    for t in range(T):
        graph_term = Abar @ x
        x = np.maximum(beta1 * x + beta2 * graph_term + u[t], 0.0) + eps[t]
        out[t] = x
    return out


def windows(series: np.ndarray, history: int = 12, pred: int = 12):
    total = history + pred
    n = len(series) - total + 1
    if n <= 0:
        raise ValueError("Synthetic series is too short")
    x = np.stack([series[i:i + history] for i in range(n)], axis=0)[..., None]
    y = np.stack([series[i + history:i + total] for i in range(n)], axis=0)[..., None]
    return x.astype(np.float32), y.astype(np.float32)


def split_7_1_2(x: np.ndarray, y: np.ndarray):
    n = len(x)
    b1, b2 = int(0.7 * n), int(0.8 * n)
    if not (0 < b1 < b2 < n):
        raise ValueError(f"Bad 7:1:2 split for n={n}")
    return x[:b1], y[:b1], x[b1:b2], y[b1:b2], x[b2:], y[b2:]


def selected_indices(n: int, max_windows: int) -> np.ndarray:
    if max_windows <= 0 or n <= max_windows:
        return np.arange(n, dtype=np.int64)
    k = int(max_windows)
    idx = np.floor(np.arange(k, dtype=np.float64) * (n - 1) / max(k - 1, 1)).astype(np.int64)
    idx[-1] = n - 1
    return idx


def topology_only(adj: np.ndarray, kind: str, spec: dict, seed: int):
    N = adj.shape[0]
    dummy_x = np.zeros((1, 1, N, 1), dtype=np.float32)
    dummy_y = np.zeros_like(dummy_x)
    inter = make_intervention(
        kind,
        dummy_x,
        dummy_y,
        adj,
        seed=int(seed),
        y_mask=np.ones_like(dummy_y, dtype=bool),
        severity=spec.get("severity"),
        weight_gamma=spec.get("gamma"),
        edge_removal_mode="paper_top40_uniform",
        edge_pool_ratio=0.40,
    )
    return inter


def save_external(path: Path, x: np.ndarray, y: np.ndarray, adj: np.ndarray,
                  kind: str, spec: dict, seed: int, inter_meta: dict, n_baseline: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x.astype(np.float32), y=y.astype(np.float32), adj=adj.astype(np.float32), kind=np.array(kind))
    meta = dict(inter_meta)
    meta.update({
        "generator": "CAUSALGAP_SYNTHETIC_NONLINEAR_GRAPH_DYNAMICS_V1",
        "causal_valid": True,
        "leakage_detected": False,
        "baseline_start": 0,
        "baseline_stop": int(n_baseline),
        "seed": int(seed),
        "intervention_sampling_seed": int(seed),
        "paper_intensity_level": spec["level"],
        "h1_phase": spec["phase"],
        "declared_strength": float(spec["strength"]),
        "severity": float(spec["strength"]),
    })
    json_dump(path.with_suffix(".json"), meta)


def main():
    ap = argparse.ArgumentParser(description="Generate the intervention-consistent synthetic H1 setting.")
    ap.add_argument("--project-root", type=Path, default=Path("/data/wanganna/CausalGap"))
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    ap.add_argument("--graph-dataset", default="METR-LA",
                    help="Base graph used for the controlled synthetic dynamics. The revised draft does not specify a synthetic graph generator; default reuses the benchmark METR-LA graph explicitly.")
    ap.add_argument("--dynamics-seed", type=int, default=2027)
    ap.add_argument("--burn-in", type=int, default=200)
    ap.add_argument("--snapshots", type=int, default=5000)
    ap.add_argument("--history", type=int, default=12)
    ap.add_argument("--pred", type=int, default=12)
    ap.add_argument("--max-test-windows", type=int, default=32,
                    help="<=0 uses the full 20%% test split. 32 is the compute-reduced H1 pilot default.")
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--calibration-seeds", nargs="+", type=int, default=list(range(1000, 1020)))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.experiment_root / "environments" / "synthetic"
    bundle_path = root / "synthetic_bundle.npz"
    manifest_path = root / "manifest.csv"
    cal_path = root / "calibration_distance_reference.csv"
    if bundle_path.exists() and manifest_path.exists() and cal_path.exists() and not args.force:
        print(f"[REUSE] synthetic environment already prepared: {root}")
        return

    graph_bundle = load_dataset(args.graph_dataset, history_len=12, pred_len=12)
    A0 = np.asarray(graph_bundle.adj, dtype=np.float32)
    N = A0.shape[0]
    total = int(args.burn_in + args.snapshots)
    rng = np.random.default_rng(int(args.dynamics_seed))
    x_init = rng.normal(0.0, 0.2, size=N).astype(np.float32)
    u = rng.normal(0.0, 0.1, size=(total, N)).astype(np.float32)
    eps = rng.normal(0.0, 0.05, size=(total, N)).astype(np.float32)

    base_all = simulate(A0, x_init, u, eps)
    base_series = base_all[int(args.burn_in):]
    bx, by = windows(base_series, args.history, args.pred)
    train_x, train_y, val_x, val_y, full_test_x, full_test_y = split_7_1_2(bx, by)
    test_idx = selected_indices(len(full_test_x), int(args.max_test_windows))
    test_x, test_y = full_test_x[test_idx], full_test_y[test_idx]

    # Fit the synthetic scaler from training inputs only, exactly like the benchmark convention.
    scaler_mean = float(np.mean(train_x))
    scaler_std = float(np.std(train_x))
    if not np.isfinite(scaler_std) or scaler_std < 1e-8:
        raise RuntimeError(f"Degenerate synthetic scaler std={scaler_std}")

    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bundle_path,
        train_x=train_x, train_y=train_y,
        val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y,
        adj=A0,
        coords=np.asarray(graph_bundle.coords) if graph_bundle.coords is not None else np.zeros((N, 2), dtype=np.float32),
        scaler_mean=np.array(scaler_mean, dtype=np.float64),
        scaler_std=np.array(scaler_std, dtype=np.float64),
        test_indices_in_full_split=test_idx,
        full_test_window_count=np.array(len(full_test_x), dtype=np.int64),
    )

    rows = []
    for kind in STRUCTURAL_TYPES:
        for spec in H1_LEVELS[kind]:
            for seed in args.eval_seeds:
                inter = topology_only(A0, kind, spec, seed)
                traj_all = simulate(inter.adj, x_init, u, eps)
                series = traj_all[int(args.burn_in):]
                ix, iy = windows(series, args.history, args.pred)
                _, _, _, _, full_ix, full_iy = split_7_1_2(ix, iy)
                ix, iy = full_ix[test_idx], full_iy[test_idx]
                env_path = root / "interventions" / kind / safe_name(spec["level"]) / f"seed_{int(seed):04d}.npz"
                save_external(env_path, ix, iy, inter.adj, kind, spec, seed, inter.meta, len(test_x))
                rows.append({
                    "setting": "synthetic",
                    "intervention_type": kind,
                    "level": spec["level"],
                    "phase": spec["phase"],
                    "strength": float(spec["strength"]),
                    "normalized_weak_intensity": normalized_weak_strength(kind, spec["strength"]) if spec["phase"] == "weak" else np.nan,
                    "seed": int(seed),
                    "environment_distance": float(inter.environment_distance),
                    "path": str(env_path),
                })
                print(f"[ENV] {kind:22s} {spec['level']:10s} seed={seed:3d} dT={inter.environment_distance:.6f}", flush=True)

    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    cal_rows = []
    for kind in STRUCTURAL_TYPES:
        for spec in H1_LEVELS[kind]:
            if spec["phase"] != "weak":
                continue
            for seed in args.calibration_seeds:
                inter = topology_only(A0, kind, spec, seed)
                cal_rows.append({
                    "setting": "synthetic",
                    "intervention_type": kind,
                    "level": spec["level"],
                    "seed": int(seed),
                    "environment_distance": float(inter.environment_distance),
                })
    pd.DataFrame(cal_rows).to_csv(cal_path, index=False)

    json_dump(root / "synthetic_metadata.json", {
        "graph_source": args.graph_dataset,
        "num_nodes": int(N),
        "dynamics": "x[t+1] = ReLU(0.7*x[t] + 0.2*Arow*x[t] + u[t]) + eps[t]",
        "u_std": 0.1,
        "eps_std": 0.05,
        "burn_in": int(args.burn_in),
        "snapshots_after_burn_in": int(args.snapshots),
        "history": int(args.history),
        "prediction_horizon": int(args.pred),
        "split": "windows_then_chronological_7_1_2",
        "full_test_window_count": int(len(full_test_x)),
        "selected_test_window_count": int(len(test_x)),
        "selected_test_indices": test_idx.tolist(),
        "dynamics_seed": int(args.dynamics_seed),
        "eval_intervention_seeds": list(map(int, args.eval_seeds)),
        "calibration_topology_seeds": list(map(int, args.calibration_seeds)),
        "common_random_numbers": True,
        "note": "The revised draft specifies the dynamics but not the synthetic base-graph generator. This implementation explicitly reuses the METR-LA benchmark adjacency; keep this choice in the manuscript or change --graph-dataset before running.",
    })
    print(f"[DONE] synthetic bundle: {bundle_path}")
    print(f"[DONE] synthetic manifest: {manifest_path}")
    print(f"[DONE] calibration distances: {cal_path}")


if __name__ == "__main__":
    main()
