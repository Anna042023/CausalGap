#!/usr/bin/env python3
"""Build the fixed cross-family intervention-magnitude calibration reference C_{d,t}.

This script performs NO model training and NO model inference.  It only loads each
baseline adjacency A0 and regenerates the weak structural perturbations required by
the manuscript's Dcal x Iweak magnitude normalizer.  Because d_T depends only on
A0/Ak (or the isolated-node fraction), the temporal Dcal samples and labels are not
needed to build C_{d,t}.

Paper protocol implemented here:
  edge removal weak:       s in {0.05, 0.10}
  weight modification weak gamma in {0.8, 0.5}
  node isolation weak:     s in {0.10, 0.20, 0.40}
  intervention seeds:      0..19 by default

The exact same graph perturbation helpers as the benchmark are imported, including
paper_top40_uniform edge removal and normalized Frobenius d_T.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from causalgap.data import load_dataset  # noqa: E402
from causalgap.interventions import (  # noqa: E402
    _apply_edge_removal,
    _apply_weight_modification,
    _apply_node_isolation,
    structural_distance,
)

DEFAULT_DATASETS = [
    "NYCTaxi", "NYCBike1", "NYCBike2", "BJTaxi",
    "METR-LA", "PEMS-BAY", "PEMS04", "PEMS08",
]

WEAK = {
    "edge_removal": [
        {"level": "edge_s0.05", "severity": 0.05},
        {"level": "edge_s0.10", "severity": 0.10},
    ],
    "weight_modification": [
        {"level": "weight_gamma0.8", "gamma": 0.8},
        {"level": "weight_gamma0.5", "gamma": 0.5},
    ],
    "node_isolation": [
        {"level": "node_s0.10", "severity": 0.10},
        {"level": "node_s0.20", "severity": 0.20},
        {"level": "node_s0.40", "severity": 0.40},
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_range(text: str) -> list[int]:
    out: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(token))
    if not out:
        raise ValueError("No seeds parsed")
    return sorted(set(out))


def adj_digest(adj: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(adj, dtype=np.float32))
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def one_dataset(dataset: str, data_root: Path, seeds: list[int]) -> tuple[list[dict], dict]:
    print(f"[CALIBRATION] loading {dataset}", flush=True)
    bundle = load_dataset(dataset, data_root=data_root)
    a0 = np.asarray(bundle.adj, dtype=np.float32)
    rows: list[dict] = []

    for seed in seeds:
        for spec in WEAK["edge_removal"]:
            rng = np.random.default_rng(seed)
            ak, removed, symmetric = _apply_edge_removal(
                a0, rng, ratio=float(spec["severity"]),
                mode="paper_top40_uniform", pool_ratio=0.40,
            )
            dt = structural_distance(a0, ak, "edge_removal")
            rows.append({
                "dataset": dataset,
                "intervention_type": "edge_removal",
                "calibration_level": spec["level"],
                "intervention_seed": seed,
                "requested_strength": float(spec["severity"]),
                "gamma": np.nan,
                "environment_distance": float(dt),
                "distance_kind": "normalized_graph_frobenius",
                "n_changed_units": int(len(removed)),
                "graph_symmetric": bool(symmetric),
            })

        for spec in WEAK["weight_modification"]:
            rng = np.random.default_rng(seed)
            gamma = float(spec["gamma"])
            ak, modified, symmetric = _apply_weight_modification(a0, rng, ratio=0.15, gamma=gamma)
            dt = structural_distance(a0, ak, "weight_modification")
            rows.append({
                "dataset": dataset,
                "intervention_type": "weight_modification",
                "calibration_level": spec["level"],
                "intervention_seed": seed,
                "requested_strength": float(1.0 - gamma),
                "gamma": gamma,
                "environment_distance": float(dt),
                "distance_kind": "normalized_graph_frobenius",
                "n_changed_units": int(len(modified)),
                "graph_symmetric": bool(symmetric),
            })

        for spec in WEAK["node_isolation"]:
            rng = np.random.default_rng(seed)
            ak, nodes = _apply_node_isolation(a0, rng, float(spec["severity"]))
            dt = structural_distance(a0, ak, "node_isolation", isolated_nodes=nodes)
            rows.append({
                "dataset": dataset,
                "intervention_type": "node_isolation",
                "calibration_level": spec["level"],
                "intervention_seed": seed,
                "requested_strength": float(spec["severity"]),
                "gamma": np.nan,
                "environment_distance": float(dt),
                "distance_kind": "isolated_node_fraction",
                "n_changed_units": int(len(nodes)),
                "graph_symmetric": bool(np.allclose(a0, a0.T, atol=1e-6, rtol=0.0)),
            })

    meta = {
        "dataset": dataset,
        "num_nodes": int(a0.shape[0]),
        "adjacency_digest": adj_digest(a0),
        "reference_rows": len(rows),
        "seeds": seeds,
    }
    del bundle, a0
    gc.collect()
    return rows, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CausalGap DMIS cross-family calibration reference")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/data/wanganna/CausalGap/datasets"))
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref_csv = out / "dmis_magnitude_calibration_reference.csv"
    manifest_path = out / "calibration_manifest.json"
    if ref_csv.exists() and not args.overwrite:
        print(f"[SKIP] calibration already exists: {ref_csv}")
        print("       pass --overwrite to regenerate it")
        return

    seeds = parse_int_range(args.seeds)
    all_rows: list[dict] = []
    dataset_meta = []
    for dataset in args.datasets:
        rows, meta = one_dataset(dataset, args.data_root.expanduser().resolve(), seeds)
        all_rows.extend(rows)
        dataset_meta.append(meta)
        pd.DataFrame(all_rows).to_csv(ref_csv, index=False)
        print(f"[CALIBRATION] {dataset}: {len(rows)} reference magnitudes | checkpoint saved", flush=True)

    ref = pd.DataFrame(all_rows)
    if ref.empty:
        raise SystemExit("Calibration reference is empty")
    if not np.isfinite(ref["environment_distance"].to_numpy(float)).all():
        raise SystemExit("Calibration reference contains non-finite d_T")
    if (ref["environment_distance"].to_numpy(float) <= 0).any():
        raise SystemExit("Calibration reference contains non-positive d_T")

    summary = (
        ref.groupby(["dataset", "intervention_type"], dropna=False)["environment_distance"]
        .agg(["count", "min", "median", "max"])
        .reset_index()
    )
    summary.to_csv(out / "calibration_reference_summary.csv", index=False)

    manifest = {
        "created_at_utc": utc_now(),
        "purpose": "Fixed C_{d,t} magnitude reference for cross-family DMIS calibration",
        "data_root": str(args.data_root.expanduser().resolve()),
        "datasets": list(args.datasets),
        "intervention_seeds": seeds,
        "weak_sets": WEAK,
        "reference_formula_note": (
            "For each main d_T, d_tilde=(1 + #ref<d_T + 0.5*#ref==d_T)/(N_ref+1), capped at 1. "
            "Dcal labels are unnecessary here because d_T is topology-only."
        ),
        "dataset_metadata": dataset_meta,
        "n_reference_rows": int(len(ref)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] reference: {ref_csv}")
    print(f"[DONE] summary  : {out/'calibration_reference_summary.csv'}")


if __name__ == "__main__":
    main()
