#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from causalgap.data import load_dataset  # noqa: E402
from causalgap.interventions import make_intervention  # noqa: E402

from protocol import DEFAULT_EXPERIMENT_ROOT, H1_LEVELS, STRUCTURAL_TYPES, json_dump, normalized_weak_strength, safe_name  # noqa: E402


def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    missing = [k for k in ("x", "y") if k not in z.files]
    if missing:
        raise ValueError(f"{path} missing {missing}; required x/y in ORIGINAL speed units")
    x = np.asarray(z["x"], dtype=np.float32)
    y = np.asarray(z["y"], dtype=np.float32)
    adj = np.asarray(z["adj"], dtype=np.float32) if "adj" in z.files else None
    return x, y, adj


def sidecar(path: Path) -> dict:
    p = path.with_suffix(".json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def source_candidate(root: Path, kind: str, level: str, seed: int) -> Path:
    return root / "interventions" / kind / safe_name(level) / f"seed_{int(seed):04d}.npz"


def infer_isolated_nodes(a0: np.ndarray, ak: np.ndarray) -> list[int]:
    changed = np.abs(ak - a0) > 1e-8
    deg0 = np.sum(np.abs(a0), axis=0) + np.sum(np.abs(a0), axis=1)
    degk = np.sum(np.abs(ak), axis=0) + np.sum(np.abs(ak), axis=1)
    return np.flatnonzero((deg0 > 1e-8) & (degk <= 1e-8) & (np.any(changed, axis=0) | np.any(changed, axis=1))).astype(int).tolist()


def copy_or_link(src: Path, dst: Path, hardlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def topology_only(adj: np.ndarray, kind: str, spec: dict, seed: int):
    N = adj.shape[0]
    x = np.zeros((1, 1, N, 1), dtype=np.float32)
    y = np.zeros_like(x)
    return make_intervention(
        kind, x, y, adj, seed=int(seed), y_mask=np.ones_like(y, dtype=bool),
        severity=spec.get("severity"), weight_gamma=spec.get("gamma"),
        edge_removal_mode="paper_top40_uniform", edge_pool_ratio=0.40,
    )


def write_contract_template(root: Path, seeds: list[int]):
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        "SUMO source contract for CausalGap Section 5.2 H1",
        "",
        "baseline.npz must contain raw/original-unit arrays:",
        "  x: [B,12,207,1] pre-intervention METR-LA-aligned speed context",
        "  y: [B,12,207,1] baseline simulated future",
        "  adj: [207,207] baseline model adjacency (optional; checked if present)",
        "",
        "Each intervention file must contain x,y,adj in original units and be placed at:",
    ]
    for kind in STRUCTURAL_TYPES:
        for spec in H1_LEVELS[kind]:
            for seed in seeds:
                lines.append(f"  interventions/{kind}/{safe_name(spec['level'])}/seed_{seed:04d}.npz")
    lines += [
        "",
        "SUMO requirements enforced by the packer:",
        "  * x is identical to baseline x (the 12-step input is entirely pre-intervention);",
        "  * y is intervention-consistent simulator output under Ak;",
        "  * all arrays share the same B and node order;",
        "  * node_isolation sidecar may provide isolated_nodes; otherwise zero-degree isolated nodes are inferred;",
        "  * baseline/intervention runs must use identical OD demand, departures, vehicle types, and simulator seed.",
    ]
    (root / "SUMO_SOURCE_CONTRACT.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Validate and stage calibrated SUMO-METR-LA H1 environments.")
    ap.add_argument("--source-root", type=Path, required=True,
                    help="Directory containing calibrated SUMO baseline.npz and intervention files in the documented contract.")
    ap.add_argument("--project-root", type=Path, default=Path("/data/wanganna/CausalGap"))
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--calibration-seeds", nargs="+", type=int, default=list(range(1000, 1020)))
    ap.add_argument("--hardlink", action="store_true")
    ap.add_argument("--x-atol", type=float, default=1e-5)
    args = ap.parse_args()

    write_contract_template(args.source_root, args.eval_seeds)
    baseline_src = args.source_root / "baseline.npz"
    if not baseline_src.exists():
        raise FileNotFoundError(
            f"Missing calibrated SUMO baseline: {baseline_src}\n"
            f"A contract template was written to {args.source_root / 'SUMO_SOURCE_CONTRACT.txt'}. "
            "This experiment intentionally does not fabricate SUMO/OD calibration data."
        )

    metr = load_dataset("METR-LA", history_len=12, pred_len=12)
    a0 = np.asarray(metr.adj, dtype=np.float32)
    bx, by, badj = load_npz(baseline_src)
    if bx.ndim != 4 or by.ndim != 4 or bx.shape[1] != 12 or by.shape[1] != 12:
        raise ValueError(f"SUMO baseline must have [B,12,N,F], got x={bx.shape}, y={by.shape}")
    if bx.shape[2] != metr.num_nodes or by.shape[2] != metr.num_nodes:
        raise ValueError(f"SUMO node count must match METR-LA N={metr.num_nodes}, got x={bx.shape}, y={by.shape}")
    if badj is not None and not np.allclose(badj, a0, atol=1e-6, rtol=0):
        raise ValueError("baseline.npz adj does not match the current METR-LA model adjacency")

    dst_root = args.experiment_root / "environments" / "sumo_metrla"
    dst_root.mkdir(parents=True, exist_ok=True)
    baseline_dst = dst_root / "baseline.npz"
    copy_or_link(baseline_src, baseline_dst, args.hardlink)

    rows = []
    for kind in STRUCTURAL_TYPES:
        for spec in H1_LEVELS[kind]:
            for seed in args.eval_seeds:
                src = source_candidate(args.source_root, kind, spec["level"], seed)
                if not src.exists():
                    raise FileNotFoundError(f"Missing SUMO intervention: {src}")
                x, y, ak = load_npz(src)
                if ak is None:
                    raise ValueError(f"{src} must contain adj=Ak")
                if x.shape != bx.shape or y.shape != by.shape:
                    raise ValueError(f"SUMO shape mismatch at {src}: x={x.shape}, y={y.shape}; baseline={bx.shape}/{by.shape}")
                if ak.shape != a0.shape:
                    raise ValueError(f"SUMO adjacency shape mismatch at {src}: {ak.shape} != {a0.shape}")
                if not np.allclose(x, bx, atol=float(args.x_atol), rtol=0):
                    maxdiff = float(np.max(np.abs(x - bx)))
                    raise ValueError(
                        f"SUMO input window is not identical to baseline pre-intervention x at {src}; max |dx|={maxdiff:.6g}. "
                        "The revised H1 timeline requires the entire 12-step input to precede the intervention."
                    )

                meta = sidecar(src)
                if kind == "node_isolation" and not meta.get("isolated_nodes"):
                    meta["isolated_nodes"] = infer_isolated_nodes(a0, ak)
                meta.update({
                    "generator": meta.get("generator", "SUMO_CALIBRATED_METR_LA"),
                    "causal_valid": bool(meta.get("causal_valid", True)),
                    "leakage_detected": bool(meta.get("leakage_detected", False)),
                    "baseline_start": 0,
                    "baseline_stop": int(len(bx)),
                    "seed": int(seed),
                    "intervention_sampling_seed": int(seed),
                    "paper_intensity_level": spec["level"],
                    "h1_phase": spec["phase"],
                    "declared_strength": float(spec["strength"]),
                    "severity": float(spec["strength"]),
                    "sumo_common_od_and_randomness_required": True,
                    "sumo_route_recomputation_required": True,
                })
                dst = dst_root / "interventions" / kind / safe_name(spec["level"]) / f"seed_{seed:04d}.npz"
                copy_or_link(src, dst, args.hardlink)
                json_dump(dst.with_suffix(".json"), meta)

                # dT is recomputed by the evaluator from A0/Ak, but save it here for audit.
                diff = float(np.linalg.norm(ak - a0) / max(np.linalg.norm(a0), 1e-6))
                if kind == "node_isolation":
                    diff = float(len(meta.get("isolated_nodes", [])) / metr.num_nodes)
                rows.append({
                    "setting": "sumo_metrla",
                    "intervention_type": kind,
                    "level": spec["level"],
                    "phase": spec["phase"],
                    "strength": float(spec["strength"]),
                    "normalized_weak_intensity": normalized_weak_strength(kind, spec["strength"]) if spec["phase"] == "weak" else np.nan,
                    "seed": int(seed),
                    "environment_distance": diff,
                    "path": str(dst),
                })

    pd.DataFrame(rows).to_csv(dst_root / "manifest.csv", index=False)

    # DMIScal magnitude calibration uses disjoint weak topology draws and no test labels.
    cal_rows = []
    for kind in STRUCTURAL_TYPES:
        for spec in H1_LEVELS[kind]:
            if spec["phase"] != "weak":
                continue
            for seed in args.calibration_seeds:
                inter = topology_only(a0, kind, spec, seed)
                cal_rows.append({
                    "setting": "sumo_metrla",
                    "intervention_type": kind,
                    "level": spec["level"],
                    "seed": int(seed),
                    "environment_distance": float(inter.environment_distance),
                })
    pd.DataFrame(cal_rows).to_csv(dst_root / "calibration_distance_reference.csv", index=False)

    json_dump(dst_root / "sumo_metadata.json", {
        "source_root": str(args.source_root),
        "baseline_source": str(baseline_src),
        "n_windows": int(len(bx)),
        "num_nodes": int(metr.num_nodes),
        "history": 12,
        "prediction_horizon": 12,
        "input_timeline": "entire 12-step x window precedes structural intervention",
        "eval_intervention_seeds": list(map(int, args.eval_seeds)),
        "calibration_topology_seeds": list(map(int, args.calibration_seeds)),
        "scientific_contract": "OD demand, departures, vehicle types and simulator seeds identical across paired baseline/intervention runs; routes recomputed after intervention.",
    })
    print(f"[DONE] staged SUMO H1 environments -> {dst_root}")


if __name__ == "__main__":
    main()
