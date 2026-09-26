#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from causalgap.data import TrafficDataBundle, StandardScaler, load_dataset, make_loaders  # noqa: E402
from causalgap.evaluator import evaluate_original, evaluate_intervention, load_checkpoint, aggregate_run_results  # noqa: E402
from causalgap.interventions import load_external_intervention  # noqa: E402
from causalgap.models.factory import build_model  # noqa: E402
from causalgap.models.registry import get_spec  # noqa: E402
from causalgap.profiles import resolve as resolve_profile  # noqa: E402
from causalgap.trainer import train_model  # noqa: E402
from causalgap.utils import seed_everything, device_from_arg  # noqa: E402

from protocol import (  # noqa: E402
    ALL_MODELS,
    DEFAULT_EXPERIMENT_ROOT,
    FOUNDATION_MODELS,
    H1_LEVELS,
    family_of,
    json_dump,
    model_control_record,
)


def add_model_args(p):
    p.add_argument("--model", choices=ALL_MODELS, required=True)
    p.add_argument("--setting", choices=["synthetic", "sumo_metrla"], required=True)
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--profile", choices=["paper", "smoke"], default="paper")
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    p.add_argument("--project-root", type=Path, default=Path("/data/wanganna/CausalGap"))
    p.add_argument("--checkpoint-roots", nargs="*", type=Path, default=None)
    p.add_argument("--train-missing", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--artifact-level", choices=["minimal", "full"], default="full")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-rep-samples", type=int, default=64)
    p.add_argument("--foundation-num-samples", type=int, default=100)
    p.add_argument("--foundation-cache", action="store_true", default=True)
    p.add_argument("--foundation-cache-dir", type=Path, default=None)
    p.add_argument("--foundation-cache-readonly", action="store_true")
    p.add_argument("--foundation-worker-startup-timeout", type=int, default=180)
    p.add_argument("--foundation-worker-infer-timeout", type=int, default=600)
    p.add_argument("--foundation-max-split-depth", type=int, default=8)
    p.add_argument("--foundation-single-window-retries", type=int, default=1)
    p.add_argument("--no-foundation-adaptive-split", action="store_false", dest="foundation_adaptive_split")
    p.set_defaults(foundation_adaptive_split=True)
    p.add_argument("--dsai-dynamic-mix", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--optimizer", default=None)
    p.add_argument("--loss-policy", default=None)
    p.add_argument("--lr-schedule", default=None)
    p.add_argument("--ks", type=int, default=None)
    p.add_argument("--kt", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--d-model", dest="d_model", type=int, default=None)
    p.add_argument("--d-ff", dest="d_ff", type=int, default=None)
    p.add_argument("--e-layers", dest="e_layers", type=int, default=None)
    p.add_argument("--d-layers", dest="d_layers", type=int, default=None)
    p.add_argument("--n-heads", dest="n_heads", type=int, default=None)
    p.add_argument("--moving-avg", dest="moving_avg", type=int, default=None)
    p.add_argument("--factor", type=int, default=None)
    p.add_argument("--patch-len", dest="patch_len", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--rebuild-paper-graph-cache", action="store_true")
    p.add_argument("--paper-graph-seed", type=int, default=2026)
    p.add_argument("--causalgrit-rf-trees", type=int, default=32)
    p.add_argument("--causalgrit-rf-max-samples", type=int, default=4096)
    p.add_argument("--causalgrit-rf-eval-samples", type=int, default=256)
    p.add_argument("--causalgrit-k-hop", type=int, default=2)
    p.add_argument("--causalgrit-max-adjustment-nodes", type=int, default=32)
    p.add_argument("--causalgrit-flow-steps", type=int, default=4)
    p.add_argument("--causalgrit-refine-rank", type=int, default=16)
    p.add_argument("--causalgrit-egc-eps", type=float, default=0.10)
    p.add_argument("--causalgrit-kl-beta-max", type=float, default=1e-3)
    p.add_argument("--causalgrit-kl-steepness", type=float, default=10.0)
    p.add_argument("--causalgrit-lambda-main", type=float, default=1.0)
    p.add_argument("--causalgrit-lambda-consistency", type=float, default=0.1)
    p.add_argument("--causalgrit-graph-reg-weight", type=float, default=1e-4)
    p.add_argument("--tfmvgnn-te-bins", type=int, default=8)
    p.add_argument("--tfmvgnn-te-threshold", type=float, default=0.6)
    p.add_argument("--tfmvgnn-te-max-samples", type=int, default=8192)
    p.add_argument("--tfmvgnn-k", type=int, default=3)


def synthetic_bundle(experiment_root: Path) -> TrafficDataBundle:
    path = experiment_root / "environments" / "synthetic" / "synthetic_bundle.npz"
    if not path.exists():
        raise FileNotFoundError(f"Synthetic environment not prepared: {path}. Run synthetic_env.py first.")
    z = np.load(path, allow_pickle=False)
    mean, std = float(z["scaler_mean"]), float(z["scaler_std"])
    scaler = StandardScaler(mean, std)
    def norm(key):
        return scaler.transform(np.asarray(z[key], dtype=np.float32)).astype(np.float32)
    train_x, train_y = norm("train_x"), norm("train_y")
    val_x, val_y = norm("val_x"), norm("val_y")
    test_x, test_y = norm("test_x"), norm("test_y")
    one = lambda a: np.ones_like(a, dtype=bool)
    coords = np.asarray(z["coords"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] != z["adj"].shape[0]:
        coords = np.zeros((z["adj"].shape[0], 2), dtype=np.float32)
    return TrafficDataBundle(
        name="SYNTHETIC-H1",
        train_x=train_x, train_y=train_y,
        val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y,
        train_y_mask=one(train_y), val_y_mask=one(val_y), test_y_mask=one(test_y),
        adj=np.asarray(z["adj"], dtype=np.float32),
        scaler=scaler,
        num_nodes=int(z["adj"].shape[0]), history_len=int(train_x.shape[1]), pred_len=int(train_y.shape[1]),
        interval_minutes=5,
        temporal_protocol="previous 12 synthetic native steps -> next 12 steps",
        data_source="controlled nonlinear graph dynamics; METR-LA base graph",
        report_primary_mape=False,
        split_protocol="windows_then_chronological_7_1_2",
        scaler_fit="train_x_only",
        coords=coords,
        coordinate_source="METR-LA graph coordinates reused for synthetic Track-B interface",
        coordinate_coverage=float(np.mean(np.any(np.isfinite(coords), axis=1))) if len(coords) else 0.0,
        missing_value_policy="finite_only",
        speed_missing_policy="raw",
        speed_missing_input_fill="keep_zero",
        mape_min_abs=1.0,
    )


def sumo_bundle(experiment_root: Path) -> TrafficDataBundle:
    base = load_dataset("METR-LA", history_len=12, pred_len=12)
    path = experiment_root / "environments" / "sumo_metrla" / "baseline.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"SUMO baseline not staged: {path}. Run prepare_sumo_envs.py with the calibrated SUMO source directory first."
        )
    z = np.load(path, allow_pickle=True)
    if "x" not in z.files or "y" not in z.files:
        raise ValueError(f"{path} must contain x and y")
    x_raw = np.asarray(z["x"], dtype=np.float32)
    y_raw = np.asarray(z["y"], dtype=np.float32)
    if x_raw.ndim != 4 or y_raw.ndim != 4 or x_raw.shape[2] != base.num_nodes:
        raise ValueError(f"Invalid SUMO baseline shape x={x_raw.shape}, y={y_raw.shape}")
    test_mask = base.infer_external_target_mask(y_raw)
    return replace(
        base,
        test_x=base.scaler.transform(x_raw).astype(np.float32),
        test_y=base.scaler.transform(y_raw).astype(np.float32),
        test_y_mask=test_mask,
        temporal_protocol="SUMO: 12 pre-intervention steps -> 12 post-t0 baseline-simulation steps",
        data_source="SUMO-calibrated METR-LA baseline for H1",
        test_fully_invalid_samples=int((~test_mask.reshape(len(test_mask), -1).any(axis=1)).sum()),
    )


def environment_manifest(experiment_root: Path, setting: str) -> pd.DataFrame:
    folder = "synthetic" if setting == "synthetic" else "sumo_metrla"
    path = experiment_root / "environments" / folder / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing H1 environment manifest: {path}")
    df = pd.read_csv(path)
    required = {"intervention_type", "level", "phase", "seed", "path"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing columns {sorted(required - set(df.columns))}")
    return df


def completed_keys(csv_path: Path) -> set[tuple[str, str, int]]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()
    req = {"intervention_type", "paper_intensity_level", "intervention_seed"}
    if not req.issubset(df.columns):
        return set()
    return {
        (str(r.intervention_type), str(r.paper_intensity_level), int(r.intervention_seed))
        for r in df.itertuples(index=False)
    }


def load_baseline_artifacts(run_dir: Path, bundle: TrafficDataBundle):
    mp = run_dir / "metrics_original.json"
    pp = run_dir / "predictions_original.npz"
    rp = run_dir / "representations_original.npz"
    if not (mp.exists() and pp.exists() and rp.exists()):
        return None
    try:
        metrics = json.loads(mp.read_text(encoding="utf-8"))
        with np.load(pp, allow_pickle=False) as z:
            pred = np.asarray(z["pred"]); y = np.asarray(z["y"]); mask = np.asarray(z["y_mask"], dtype=bool)
        with np.load(rp, allow_pickle=False) as z:
            rep = np.asarray(z["z"]); rep_idx = np.asarray(z["rep_indices"], dtype=np.int64)
        if len(pred) != len(bundle.test_x) or len(y) != len(bundle.test_y):
            return None
        if len(rep) != len(rep_idx):
            return None
        return metrics, pred, y, mask, rep, rep_idx
    except Exception:
        return None


def locate_checkpoint(roots: list[Path], dataset: str, model: str, seed: int) -> Path | None:
    candidates = []
    for root in roots:
        root = Path(root)
        candidates += [
            root / dataset / model / "trackB" / f"seed{seed}" / "checkpoint.pt",
            root / "work_runs" / dataset / model / "trackB" / f"seed{seed}" / "checkpoint.pt",
            root / "runs" / dataset / model / "trackB" / f"seed{seed}" / "checkpoint.pt",
        ]
    for p in candidates:
        if p.exists():
            return p
    return None


def foundation_weight_path(model: str, project_root: Path) -> Path | None:
    if model == "Chronos": return project_root / "model_weights" / "Chronos"
    if model == "TimesFM": return project_root / "model_weights" / "TimesFM" / "torch_model.ckpt"
    if model == "Moirai": return project_root / "model_weights" / "Moirai"
    if model == "Lag-Llama": return project_root / "model_weights" / "Lag-Llama" / "lag-llama.ckpt"
    return None


def save_controls(run_dir: Path, model_name: str, model, project_root: Path):
    rec = model_control_record(
        model_name,
        constructed_model=model,
        native_uses_graph=bool(get_spec(model_name).native_uses_external_graph),
        foundation_weight_path=foundation_weight_path(model_name, project_root),
    )
    pd.DataFrame([rec]).to_csv(run_dir / "model_controls.csv", index=False)
    json_dump(run_dir / "model_controls.json", rec)


def main():
    ap = argparse.ArgumentParser(description="Run one CausalGap Section 5.2 H1 model/setting job.")
    add_model_args(ap)
    args = ap.parse_args()

    os.environ.setdefault("CAUSALGAP_ROOT", str(args.project_root))
    seed_everything(int(args.model_seed))
    device = device_from_arg(args.device)
    args.seed = int(args.model_seed)
    args.track = "B"
    args.dataset = "SYNTHETIC-H1" if args.setting == "synthetic" else "METR-LA"
    if args.foundation_cache_dir is None:
        args.foundation_cache_dir = args.project_root / "cache" / "foundation_inference"

    bundle = synthetic_bundle(args.experiment_root) if args.setting == "synthetic" else sumo_bundle(args.experiment_root)
    args = resolve_profile(args, args.model, history_len=bundle.history_len, num_nodes=bundle.num_nodes)
    model = build_model(args.model, bundle, args)

    run_dir = args.experiment_root / "runs" / args.setting / args.model / "trackB" / f"seed{args.model_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    json_dump(run_dir / "h1_job_config.json", {
        "setting": args.setting,
        "model": args.model,
        "model_family": family_of(args.model),
        "model_seed": int(args.model_seed),
        "evaluation_track": "B",
        "profile": args.profile,
        "eval_intervention_seeds": list(map(int, args.eval_seeds)),
        "artifact_level": args.artifact_level,
        "alpha": float(args.alpha),
        "max_rep_samples": int(args.max_rep_samples),
        "bundle_name_for_checkpoint": bundle.name,
        "test_windows": int(len(bundle.test_x)),
    })
    save_controls(run_dir, args.model, model, args.project_root)

    zero_shot = args.model in FOUNDATION_MODELS or bool(getattr(model, "is_zero_shot", False))
    if not zero_shot:
        local_ckpt = run_dir / "checkpoint.pt"
        if args.rerun and local_ckpt.is_symlink():
            local_ckpt.unlink()
        if not local_ckpt.exists():
            if args.setting == "sumo_metrla":
                roots = args.checkpoint_roots or [
                    args.project_root / "experiment_results" / "observational_diagnostics",
                    args.project_root,
                ]
                src = locate_checkpoint(roots, "METR-LA", args.model, args.model_seed)
                if src is not None:
                    local_ckpt.symlink_to(src.resolve())
                    json_dump(run_dir / "checkpoint_source.json", {"mode": "reused_METR-LA_TrackB", "source": str(src.resolve())})
                elif not args.train_missing:
                    raise FileNotFoundError(
                        f"No compatible METR-LA Track-B checkpoint found for {args.model}. "
                        "Use --train-missing or pass --checkpoint-roots."
                    )
            # Synthetic has a different training distribution and must train locally.
            if not local_ckpt.exists():
                if args.setting != "synthetic" and not args.train_missing:
                    raise FileNotFoundError(local_ckpt)
                loaders = make_loaders(bundle, args.batch_size)
                print(f"[TRAIN] {args.setting} / {args.model} / seed{args.model_seed}", flush=True)
                model = train_model(
                    model, bundle, loaders, device, run_dir,
                    args.epochs, args.lr, args.patience, args.weight_decay,
                    args.optimizer, args.loss_policy, args.lr_schedule,
                )
        if local_ckpt.exists():
            try:
                model = load_checkpoint(model.to(device), local_ckpt, device, bundle=bundle, allow_legacy=False)
            except Exception:
                # If a reused SUMO/METR-LA symlink is stale, --train-missing is allowed to
                # replace only the H1-local link/copy, never the original checkpoint.
                if args.setting == "sumo_metrla" and args.train_missing and local_ckpt.is_symlink():
                    local_ckpt.unlink()
                    loaders = make_loaders(bundle, args.batch_size)
                    model = train_model(
                        model, bundle, loaders, device, run_dir,
                        args.epochs, args.lr, args.patience, args.weight_decay,
                        args.optimizer, args.loss_policy, args.lr_schedule,
                    )
                    model = load_checkpoint(model.to(device), local_ckpt, device, bundle=bundle, allow_legacy=False)
                else:
                    raise
    else:
        model = model.to(device)

    # Parameter count is saved again after any training/rebuild so the exact model
    # used for H1 is auditable.
    save_controls(run_dir, args.model, model, args.project_root)

    baseline = None if args.rerun else load_baseline_artifacts(run_dir, bundle)
    if baseline is None:
        baseline = evaluate_original(model, bundle, device, run_dir, args.batch_size, args.max_rep_samples)
    else:
        print(f"[RESUME] baseline artifacts reused from {run_dir}", flush=True)
    base_metrics, base_pred, base_y, base_y_mask, z0, rep_indices = baseline

    result_csv = run_dir / "causalgap_results.csv"
    if args.rerun and result_csv.exists():
        backup = run_dir / "partial_backups" / "causalgap_results_before_rerun.csv"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_csv, backup)
        result_csv.unlink()
    done = completed_keys(result_csv)

    manifest = environment_manifest(args.experiment_root, args.setting)
    manifest = manifest[manifest["seed"].astype(int).isin(set(map(int, args.eval_seeds)))].copy()
    expected = len(H1_LEVELS["edge_removal"]) * len(args.eval_seeds) + len(H1_LEVELS["weight_modification"]) * len(args.eval_seeds) + len(H1_LEVELS["node_isolation"]) * len(args.eval_seeds)
    if len(manifest) != expected:
        raise RuntimeError(f"Manifest rows {len(manifest)} != expected {expected} for eval seeds {args.eval_seeds}")

    for idx, r in enumerate(manifest.itertuples(index=False)):
        key = (str(r.intervention_type), str(r.level), int(r.seed))
        if key in done:
            print(f"[RESUME] skip {key}", flush=True)
            continue
        p = Path(r.path)
        if not p.exists():
            raise FileNotFoundError(p)
        inter = load_external_intervention(p, bundle, kind=str(r.intervention_type))
        inter.meta["paper_intensity_level"] = str(r.level)
        inter.meta["h1_phase"] = str(r.phase)
        inter.meta["model_training_seed"] = int(args.model_seed)
        inter.meta["intervention_sampling_seed"] = int(r.seed)
        inter.meta["seed"] = int(r.seed)
        row = evaluate_intervention(
            model, bundle, inter, device, run_dir,
            base_metrics, base_pred, base_y, base_y_mask, z0, rep_indices,
            args.batch_size, args.max_rep_samples, args.alpha, idx,
            force_include_adjacency_only=False,
            artifact_level=args.artifact_level,
            fail_on_unobservable_trackb=True,
        )
        done.add(key)
        print(
            f"[H1] {args.setting}/{args.model} {key} RD={row['RD']:.6f} "
            f"DZ={row['representation_response_raw']:.6f} dT={row['environment_distance']:.6f}",
            flush=True,
        )

    aggregate_run_results(run_dir)
    if hasattr(model, "close"):
        try:
            model.close()
        except Exception:
            pass
    print(f"[PASS] H1 job complete -> {run_dir}")


if __name__ == "__main__":
    main()
