#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one Controlled-Synthetic Track-A job for the CausalGap DSAI audit.

This is an isolated Section 5.8 runner. It reuses the already-prepared Section 5.2
synthetic environments but writes ALL Track-A artifacts to a new Section 5.8 output
root. It never edits the frozen Section 5.2 Track-B results.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
H1_DIR = CODE_ROOT / "experiments" / "incremental_value"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(H1_DIR) not in sys.path:
    sys.path.insert(0, str(H1_DIR))


def load_h1_module():
    path = H1_DIR / "evaluate_h1_job.py"
    if not path.exists():
        raise FileNotFoundError(f"Required frozen H1 evaluator not found: {path}")
    spec = importlib.util.spec_from_file_location("causalgap_h1_base_eval", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


H1 = load_h1_module()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    H1.add_model_args(ap)
    ap.add_argument(
        "--output-root", type=Path,
        default=Path("/data/wanganna/CausalGap/experiment_results/access_sensitivity/synthetic_track_a"),
        help="Isolated Section 5.8 Track-A output root."
    )
    args = ap.parse_args()

    if args.setting != "synthetic":
        raise ValueError("This evaluator is intentionally restricted to controlled synthetic Track A.")

    source_root = Path(args.experiment_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CAUSALGAP_ROOT", str(args.project_root))
    H1.seed_everything(int(args.model_seed))
    device = H1.device_from_arg(args.device)

    # The only scientific change relative to frozen H1 is the declared access track.
    args.seed = int(args.model_seed)
    args.track = "A"
    args.dataset = "SYNTHETIC-H1"
    if args.foundation_cache_dir is None:
        args.foundation_cache_dir = args.project_root / "cache" / "foundation_inference"

    bundle = H1.synthetic_bundle(source_root)
    args = H1.resolve_profile(args, args.model, history_len=bundle.history_len, num_nodes=bundle.num_nodes)
    model = H1.build_model(args.model, bundle, args)

    run_dir = output_root / "runs" / "synthetic" / args.model / "trackA" / f"seed{args.model_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    H1.json_dump(run_dir / "h1_job_config.json", {
        "section": "5.8 DSAI Track-A audit",
        "source_environment_root": str(source_root),
        "setting": "synthetic",
        "model": args.model,
        "model_family": H1.family_of(args.model),
        "model_seed": int(args.model_seed),
        "evaluation_track": "A",
        "profile": args.profile,
        "eval_intervention_seeds": list(map(int, args.eval_seeds)),
        "artifact_level": args.artifact_level,
        "alpha": float(args.alpha),
        "max_rep_samples": int(args.max_rep_samples),
        "bundle_name_for_checkpoint": bundle.name,
        "test_windows": int(len(bundle.test_x)),
        "scope_note": "Track A is native structural access; results are kept separate from Track B.",
    })
    H1.save_controls(run_dir, args.model, model, args.project_root)

    zero_shot = args.model in H1.FOUNDATION_MODELS or bool(getattr(model, "is_zero_shot", False))
    if not zero_shot:
        local_ckpt = run_dir / "checkpoint.pt"
        if args.rerun and (local_ckpt.exists() or local_ckpt.is_symlink()):
            local_ckpt.unlink()
        if not local_ckpt.exists():
            if not args.train_missing:
                raise FileNotFoundError(
                    f"Missing Track-A synthetic checkpoint: {local_ckpt}. "
                    "Re-run with --train-missing. Track-B checkpoints are not reused because the access interface differs."
                )
            loaders = H1.make_loaders(bundle, args.batch_size)
            print(f"[TRAIN-A] synthetic / {args.model} / seed{args.model_seed}", flush=True)
            model = H1.train_model(
                model, bundle, loaders, device, run_dir,
                args.epochs, args.lr, args.patience, args.weight_decay,
                args.optimizer, args.loss_policy, args.lr_schedule,
            )
        model = H1.load_checkpoint(model.to(device), local_ckpt, device, bundle=bundle, allow_legacy=False)
    else:
        model = model.to(device)

    H1.save_controls(run_dir, args.model, model, args.project_root)

    baseline = None if args.rerun else H1.load_baseline_artifacts(run_dir, bundle)
    if baseline is None:
        baseline = H1.evaluate_original(model, bundle, device, run_dir, args.batch_size, args.max_rep_samples)
    else:
        print(f"[RESUME] Track-A baseline reused from {run_dir}", flush=True)
    base_metrics, base_pred, base_y, base_y_mask, z0, rep_indices = baseline

    result_csv = run_dir / "causalgap_results.csv"
    if args.rerun and result_csv.exists():
        backup = run_dir / "partial_backups" / "causalgap_results_before_rerun.csv"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_csv, backup)
        result_csv.unlink()
    done = H1.completed_keys(result_csv)

    manifest = H1.environment_manifest(source_root, "synthetic")
    manifest = manifest[manifest["seed"].astype(int).isin(set(map(int, args.eval_seeds)))].copy()
    expected = sum(len(v) for v in H1.H1_LEVELS.values()) * len(args.eval_seeds)
    if len(manifest) != expected:
        raise RuntimeError(f"Synthetic manifest rows={len(manifest)} expected={expected}")

    for idx, r in enumerate(manifest.itertuples(index=False)):
        key = (str(r.intervention_type), str(r.level), int(r.seed))
        if key in done:
            print(f"[RESUME] skip {key}", flush=True)
            continue
        p = Path(r.path)
        if not p.exists():
            raise FileNotFoundError(p)
        inter = H1.load_external_intervention(p, bundle, kind=str(r.intervention_type))
        inter.meta["paper_intensity_level"] = str(r.level)
        inter.meta["h1_phase"] = str(r.phase)
        inter.meta["model_training_seed"] = int(args.model_seed)
        inter.meta["intervention_sampling_seed"] = int(r.seed)
        inter.meta["seed"] = int(r.seed)

        row = H1.evaluate_intervention(
            model, bundle, inter, device, run_dir,
            base_metrics, base_pred, base_y, base_y_mask, z0, rep_indices,
            args.batch_size, args.max_rep_samples, args.alpha, idx,
            force_include_adjacency_only=False,
            artifact_level=args.artifact_level,
            fail_on_unobservable_trackb=False,
        )
        done.add(key)
        print(
            f"[TRACK-A] {args.model} {key} RD={row['RD']:.6f} "
            f"DMIS={row['DMIS']:.6f} DZ={row['representation_response_raw']:.6f}",
            flush=True,
        )

    H1.aggregate_run_results(run_dir)
    if hasattr(model, "close"):
        try:
            model.close()
        except Exception:
            pass
    print(f"[PASS] Synthetic Track-A job complete -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
