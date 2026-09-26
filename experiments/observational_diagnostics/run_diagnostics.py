#!/usr/bin/env python3
"""Section 5.1 runner: DMIS--RD association.

This script deliberately does NOT modify the benchmark core.  It creates an
experiment-isolated run tree, reuses compatible checkpoints when available,
optionally trains missing supervised Track-B checkpoints, and then launches the
paper-aligned structural intervention protocol through the existing run.py.

Default protocol (``primary_fast``):
  * Track B only
  * one model-training seed: 0
  * one fixed intervention sampling seed: 0
  * all 10 paper structural family/intensity configurations
  * deterministic whole-period diagnostic test subset, selected by a node-series budget
  * true intervention-level resume

This protocol is intentionally cheaper than the draft manuscript's 10 training seeds x
20 intervention seeds full-test protocol.  It is designed for the main association
diagnostic, with multi-seed/full-test checks left as targeted robustness experiments.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Make the repository root importable when this file lives at
# experiments/observational_diagnostics/run_diagnostics.py
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from causalgap.config import DATASET_SPECS, PROJECT_ROOT, canonical_dataset, canonical_model  # noqa: E402
from causalgap.models.registry import SPECS  # noqa: E402

FOUNDATION = {m for m, s in SPECS.items() if s.training_mode == "zero_shot"}
STRUCTURAL_TYPES = ["edge_removal", "weight_modification", "node_isolation"]
LEVELS_PER_SEED = 3 + 3 + 4


@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    track: str
    model_seed: int

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.model}__track{self.track}__seed{self.model_seed}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_models(xs: Iterable[str]) -> list[str]:
    out: list[str] = []
    for x in xs:
        m = canonical_model(x)
        if m not in SPECS:
            raise ValueError(f"Unknown model: {x}")
        if m not in out:
            out.append(m)
    return out


def normalize_datasets(xs: Iterable[str]) -> list[str]:
    out: list[str] = []
    for x in xs:
        d = canonical_dataset(x)
        if d not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset: {x}")
        if d not in out:
            out.append(d)
    return out


def resolve_protocol(args) -> tuple[list[int], list[int]]:
    if args.protocol in ("quick", "primary_fast"):
        default_model_seeds = [0]
        default_intervention_seeds = [0]
    elif args.protocol == "multi_intervention_robustness":
        default_model_seeds = [0]
        default_intervention_seeds = [0, 1, 2, 3, 4]
    elif args.protocol == "paper_full":
        default_model_seeds = list(range(10))
        default_intervention_seeds = list(range(20))
    else:
        raise ValueError(args.protocol)

    model_seeds = default_model_seeds if args.model_seeds is None else list(dict.fromkeys(args.model_seeds))
    intervention_seeds = (
        default_intervention_seeds
        if args.intervention_seeds is None
        else list(dict.fromkeys(args.intervention_seeds))
    )
    if not model_seeds:
        raise ValueError("At least one model seed is required.")
    if not intervention_seeds:
        raise ValueError("At least one intervention seed is required.")
    if any(s < 0 for s in model_seeds + intervention_seeds):
        raise ValueError("Seeds must be non-negative integers.")
    return model_seeds, intervention_seeds


def run_dir(root: Path, job: Job) -> Path:
    return root / job.dataset / job.model / f"track{job.track}" / f"seed{job.model_seed}"


def expected_counts(intervention_seeds: list[int]) -> dict[str, int]:
    n = len(intervention_seeds)
    return {
        "edge_removal": 3 * n,
        "weight_modification": 3 * n,
        "node_isolation": 4 * n,
    }


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def intervention_result_complete(path: Path, job: Job, intervention_seeds: list[int]) -> tuple[bool, str]:
    csv_path = path / "causalgap_results.csv"
    if not csv_path.exists():
        return False, "missing causalgap_results.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"unreadable csv: {e}"

    counts = expected_counts(intervention_seeds)
    expected_total = sum(counts.values())
    if len(df) != expected_total:
        return False, f"row count {len(df)} != expected {expected_total}"

    required = {
        "dataset", "model", "evaluation_track", "intervention_type", "intervention_seed",
        "DMIS", "RD", "environment_distance", "representation_response_raw",
        "paper_intensity_level", "diagnostic_subset_active", "diagnostic_selected_test_windows",
        "include_in_information_equivalent_structural", "leakage_detected",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        return False, f"missing columns: {missing}"

    if set(df["dataset"].astype(str)) != {job.dataset}:
        return False, "dataset mismatch"
    if set(df["model"].astype(str)) != {job.model}:
        return False, "model mismatch"
    if set(df["evaluation_track"].astype(str).str.upper()) != {job.track}:
        return False, "track mismatch"
    if set(df["intervention_seed"].astype(int).tolist()) != set(intervention_seeds):
        return False, "intervention seed coverage mismatch"

    for kind, n in counts.items():
        got = int((df["intervention_type"].astype(str) == kind).sum())
        if got != n:
            return False, f"{kind} rows {got} != expected {n}"

    for col in ["DMIS", "RD", "environment_distance", "representation_response_raw"]:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            return False, f"non-finite values in {col}"

    if _bool_series(df["leakage_detected"]).any():
        return False, "leakage_detected=True present"

    # Track-B structural rows must be observable/information-equivalent to be
    # eligible for the primary association analysis.
    info = _bool_series(df["include_in_information_equivalent_structural"])
    if not info.all():
        return False, f"{int((~info).sum())} Track-B structural rows failed information-equivalent validity"
    if not _bool_series(df["diagnostic_subset_active"]).all():
        return False, "diagnostic_subset_active=False present"
    if int(pd.to_numeric(df["diagnostic_selected_test_windows"], errors="coerce").nunique()) != 1:
        return False, "inconsistent diagnostic window count within job"

    return True, "complete"


def compatible_checkpoint_source(root: Path, job: Job) -> Path | None:
    rd = run_dir(root, job)
    ckpt = rd / "checkpoint.pt"
    cfg = rd / "run_config.json"
    if not ckpt.exists():
        return None
    c = read_json(cfg) if cfg.exists() else None
    if c:
        args = c.get("command_args", {}) or {}
        if str(c.get("evaluation_track", args.get("track", ""))).upper() != job.track:
            return None
        if int(args.get("seed", job.model_seed)) != job.model_seed:
            return None
        profile = str(args.get("profile", "paper")).lower()
        if profile and profile != "paper":
            return None
    return ckpt


def locate_checkpoint(job: Job, checkpoint_roots: list[Path]) -> Path | None:
    for root in checkpoint_roots:
        p = compatible_checkpoint_source(root, job)
        if p is not None:
            return p
    return None


def ensure_checkpoint_link(dst_run_dir: Path, src_ckpt: Path) -> None:
    dst_run_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_run_dir / "checkpoint.pt"
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src_ckpt.resolve():
                return
        except Exception:
            pass
        dst.unlink()
    # Symlink avoids duplicating many checkpoints and leaves the original bank untouched.
    dst.symlink_to(src_ckpt.resolve())
    dump_json(dst_run_dir / "checkpoint_source.json", {
        "mode": "reused_symlink",
        "source_checkpoint": str(src_ckpt.resolve()),
        "linked_at_utc": utc_now(),
    })


def tail(path: Path, n: int = 50) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def common_cmd_args(job: Job, args) -> list[str]:
    out = [
        "--model", job.model,
        "--dataset", job.dataset,
        "--track", job.track,
        "--seed", str(job.model_seed),
        "--profile", "paper",
        "--device", "cuda",
        "--max-rep-samples", str(args.max_rep_samples),
        "--foundation-num-samples", str(args.foundation_num_samples),
        "--foundation-worker-startup-timeout", str(args.foundation_worker_startup_timeout),
        "--foundation-worker-infer-timeout", str(args.foundation_worker_infer_timeout),
        "--foundation-max-split-depth", str(args.foundation_max_split_depth),
        "--foundation-single-window-retries", str(args.foundation_single_window_retries),
    ]
    if args.speed_missing_policy is not None:
        out += ["--speed-missing-policy", args.speed_missing_policy]
    if args.speed_missing_input_fill is not None:
        out += ["--speed-missing-input-fill", args.speed_missing_input_fill]
    if job.model in FOUNDATION:
        out += ["--foundation-cache", "--foundation-cache-dir", str(args.foundation_cache_dir)]
        if not args.foundation_adaptive_split:
            out += ["--no-foundation-adaptive-split"]
    return out


def build_train_cmd(job: Job, args) -> list[str]:
    return [sys.executable, "-u", str(CODE_ROOT / "run.py"), "train"] + common_cmd_args(job, args)


def build_intervene_cmd(job: Job, args, intervention_seeds: list[int]) -> list[str]:
    return [
        sys.executable, "-u", str(CODE_ROOT / "run.py"), "intervene",
    ] + common_cmd_args(job, args) + [
        "--types", *STRUCTURAL_TYPES,
        "--intervention-seeds", *[str(s) for s in intervention_seeds],
        "--edge-removal-mode", "paper_top40_uniform",
        "--artifact-level", "minimal",
        "--alpha", "0.5",
        "--resume-interventions",
        "--diagnostic-node-series-budget", str(args.diagnostic_node_series_budget),
        "--diagnostic-max-windows", str(args.diagnostic_max_windows),
        "--diagnostic-min-windows", str(args.diagnostic_min_windows),
    ]


def run_process(cmd: list[str], env: dict[str, str], log_path: Path,
                timeout_hours: float, heartbeat_seconds: float,
                job_key: str, gpu: str, lock: threading.Lock) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write("\n# ===== NEW ATTEMPT =====\n# COMMAND: " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=CODE_ROOT,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_beat = 0.0
        timeout_s = float(timeout_hours) * 3600.0 if timeout_hours > 0 else 0.0
        while proc.poll() is None:
            time.sleep(5)
            elapsed = time.monotonic() - t0
            if elapsed - last_beat >= heartbeat_seconds:
                with lock:
                    print(f"[RUNNING][GPU {gpu}] {job_key} elapsed={elapsed/60:.1f} min", flush=True)
                last_beat = elapsed
            if timeout_s > 0 and elapsed > timeout_s:
                with lock:
                    print(f"[TIMEOUT][GPU {gpu}] {job_key} > {timeout_hours:g} h", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                return 124
    return int(proc.returncode)


def backup_partial_result(rd: Path, backup_root: Path, job: Job) -> None:
    csv_path = rd / "causalgap_results.csv"
    if not csv_path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = backup_root / f"{job.key}__{stamp}.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, dst)


def cost(job: Job) -> tuple[int, int, int]:
    """Order jobs for useful early output: supervised first, foundation last.

    Foundation jobs are additionally guarded by a semaphore, so two Lag-Llama-like
    workloads do not occupy both GPUs at the same time.
    """
    is_foundation = 1 if job.model in FOUNDATION else 0
    model_cost = {
        "STGCN": 10, "DLinear": 11, "AGCRN": 15, "GWNet": 16, "DCRNN": 18,
        "PatchTST": 20, "iTransformer": 21, "TimesNet": 22, "FEDformer": 23,
        "Autoformer": 24, "ASTGNN": 25, "STEVE": 26, "DCGCN": 27,
        "ST-Norm": 28, "CCHMM": 29, "CausalGRIT": 30, "TF-MVGNN": 31,
        "Chronos": 80, "TimesFM": 85, "Moirai": 90, "Lag-Llama": 100,
    }.get(job.model, 40)
    dataset_cost = {
        "NYCBike1": 10, "NYCTaxi": 12, "NYCBike2": 14, "PEMS08": 16,
        "METR-LA": 18, "PEMS04": 20, "PEMS-BAY": 22, "BJTaxi": 24,
    }.get(job.dataset, 30)
    return (is_foundation, model_cost, dataset_cost)


def worker(gpu: str, q: queue.Queue, args, intervention_seeds: list[int],
           checkpoint_roots: list[Path], failures: list[dict], lock: threading.Lock,
           foundation_sem: threading.Semaphore) -> None:
    while True:
        try:
            job: Job = q.get_nowait()
        except queue.Empty:
            return

        foundation_token = False
        if job.model in FOUNDATION:
            with lock:
                print(f"[WAIT-FOUNDATION][GPU {gpu}] {job.key}", flush=True)
            foundation_sem.acquire()
            foundation_token = True

        rd = run_dir(args.work_run_root, job)
        status_path = args.status_root / f"{job.key}.json"
        ok, reason = intervention_result_complete(rd, job, intervention_seeds)
        if ok and not args.rerun:
            with lock:
                print(f"[SKIP][GPU {gpu}] {job.key} already complete", flush=True)
            dump_json(status_path, {
                "job": job.__dict__, "status": "PASS", "reason": "already complete",
                "checked_at_utc": utc_now(), "run_dir": str(rd),
            })
            if foundation_token:
                foundation_sem.release()
            q.task_done()
            continue

        if rd.exists() and not ok:
            backup_partial_result(rd, args.partial_backup_root, job)

        if args.dry_run:
            with lock:
                print(f"[DRY][GPU {gpu}] {job.key} ({reason})", flush=True)
            if foundation_token:
                foundation_sem.release()
            q.task_done()
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        env["CAUSALGAP_ROOT"] = str(args.project_root)
        env["CAUSALGAP_RUN_ROOT"] = str(args.work_run_root)
        env["CAUSALGAP_CHRONOS_CHUNK"] = str(args.chronos_chunk)
        env["CAUSALGAP_TIMESFM_CHUNK"] = str(args.timesfm_chunk)
        env["CAUSALGAP_TIMESFM_PER_CORE_BATCH"] = str(args.timesfm_per_core_batch)
        env["CAUSALGAP_MOIRAI_PRED_BATCH"] = str(args.moirai_batch)
        env["CAUSALGAP_MOIRAI_REQUEST_CHUNK"] = str(args.moirai_request_chunk)
        env["CAUSALGAP_LAGLLAMA_PRED_BATCH"] = str(args.lagllama_batch)
        env["CAUSALGAP_LAGLLAMA_REQUEST_CHUNK"] = str(args.lagllama_request_chunk)

        started = utc_now()
        t0 = time.monotonic()
        train_used = False
        checkpoint_source = None

        try:
            if job.model not in FOUNDATION:
                existing_work_ckpt = rd / "checkpoint.pt"
                if not existing_work_ckpt.exists():
                    src = locate_checkpoint(job, checkpoint_roots)
                    if src is not None:
                        ensure_checkpoint_link(rd, src)
                        checkpoint_source = str(src.resolve())
                    elif args.train_missing:
                        train_used = True
                        train_cmd = build_train_cmd(job, args)
                        train_log = args.log_root / "train" / f"{job.key}.log"
                        with lock:
                            print(f"[TRAIN][GPU {gpu}] {job.key}", flush=True)
                        rc = run_process(
                            train_cmd, env, train_log, args.timeout_hours,
                            args.heartbeat_seconds, job.key + " [train]", gpu, lock,
                        )
                        if rc != 0:
                            raise RuntimeError(f"training failed rc={rc}; see {train_log}")
                        if not (rd / "checkpoint.pt").exists():
                            raise RuntimeError("training returned success but checkpoint.pt is missing")
                        dump_json(rd / "checkpoint_source.json", {
                            "mode": "trained_in_observational_diagnostics",
                            "trained_at_utc": utc_now(),
                            "training_log": str(train_log),
                        })
                    else:
                        raise FileNotFoundError(
                            f"No compatible Track-{job.track} checkpoint for {job.key}. "
                            "Re-run with --train-missing, or provide --checkpoint-roots."
                        )

            cmd = build_intervene_cmd(job, args, intervention_seeds)
            log_path = args.log_root / "intervene" / f"{job.key}.log"
            with lock:
                print(f"[START][GPU {gpu}] {job.key} | log={log_path}", flush=True)
            timeout = args.foundation_timeout_hours if job.model in FOUNDATION else args.timeout_hours
            rc = run_process(cmd, env, log_path, timeout, args.heartbeat_seconds, job.key, gpu, lock)

            # A run_config can look compatible while an older checkpoint still
            # carries a stale internal protocol signature.  When checkpoint reuse
            # fails specifically for that reason, --train-missing is allowed to
            # replace only the experiment-local symlink and train one fresh copy.
            # Other failures are never masked by automatic retraining.
            if rc != 0 and checkpoint_source is not None and args.train_missing and job.model not in FOUNDATION:
                log_tail = tail(log_path, 120)
                signature_failure = (
                    "Checkpoint/data protocol mismatch" in log_tail
                    or "predates the v3.0 paper protocol signature" in log_tail
                    or "Checkpoint scaler statistics differ" in log_tail
                )
                if signature_failure:
                    local_ckpt = rd / "checkpoint.pt"
                    if local_ckpt.exists() or local_ckpt.is_symlink():
                        local_ckpt.unlink()
                    train_used = True
                    train_cmd = build_train_cmd(job, args)
                    train_log = args.log_root / "train" / f"{job.key}__fallback.log"
                    with lock:
                        print(f"[RETRAIN-STALE][GPU {gpu}] {job.key}", flush=True)
                    trc = run_process(
                        train_cmd, env, train_log, args.timeout_hours,
                        args.heartbeat_seconds, job.key + " [fallback train]", gpu, lock,
                    )
                    if trc != 0 or not local_ckpt.exists():
                        raise RuntimeError(f"fallback training failed rc={trc}; see {train_log}")
                    checkpoint_source = None
                    dump_json(rd / "checkpoint_source.json", {
                        "mode": "trained_in_observational_diagnostics_after_stale_checkpoint",
                        "trained_at_utc": utc_now(),
                        "training_log": str(train_log),
                    })
                    retry_log = args.log_root / "intervene" / f"{job.key}__after_retrain.log"
                    rc = run_process(
                        cmd, env, retry_log, timeout, args.heartbeat_seconds,
                        job.key + " [after retrain]", gpu, lock,
                    )
                    log_path = retry_log

            ok, reason = intervention_result_complete(rd, job, intervention_seeds)
            status = "PASS" if rc == 0 and ok else "FAIL"
            rec = {
                "job": job.__dict__,
                "status": status,
                "returncode": rc,
                "reason": reason,
                "started_at_utc": started,
                "finished_at_utc": utc_now(),
                "wall_seconds": float(time.monotonic() - t0),
                "gpu_physical_id": str(gpu),
                "run_dir": str(rd),
                "checkpoint_source": checkpoint_source,
                "trained_missing_checkpoint": train_used,
                "intervention_seeds": intervention_seeds,
                "command": cmd,
                "log": str(log_path),
            }
            dump_json(status_path, rec)
            with lock:
                print(f"[{status}][GPU {gpu}] {job.key} | {reason}", flush=True)
                if status != "PASS":
                    print(tail(log_path, 35), flush=True)
            if status != "PASS":
                failures.append(rec)
        except Exception as e:
            rec = {
                "job": job.__dict__, "status": "FAIL", "reason": repr(e),
                "started_at_utc": started, "finished_at_utc": utc_now(),
                "wall_seconds": float(time.monotonic() - t0), "gpu_physical_id": str(gpu),
                "run_dir": str(rd), "intervention_seeds": intervention_seeds,
            }
            dump_json(status_path, rec)
            with lock:
                print(f"[FAIL][GPU {gpu}] {job.key}: {e}", flush=True)
            failures.append(rec)
        finally:
            if foundation_token:
                foundation_sem.release()
            q.task_done()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CausalGap Section 5.1 DMIS--RD association data generation")
    ap.add_argument("--protocol", choices=["quick", "primary_fast", "multi_intervention_robustness", "paper_full"], default="primary_fast",
                    help="primary_fast=seed0 x intervention seed0 on diagnostic subset; "
                         "multi_intervention_robustness=seed0 x intervention seeds0..4; "
                         "paper_full=model seeds0..9 x intervention seeds0..19")
    ap.add_argument("--models", nargs="*", default=list(SPECS.keys()))
    ap.add_argument("--datasets", nargs="*", default=list(DATASET_SPECS.keys()))
    ap.add_argument("--track", choices=["A", "B", "a", "b"], default="B")
    ap.add_argument("--model-seeds", nargs="*", type=int, default=None,
                    help="Override protocol model-training seeds.")
    ap.add_argument("--intervention-seeds", nargs="*", type=int, default=None,
                    help="Override protocol intervention-sampling seeds.")
    ap.add_argument("--gpus", nargs="+", default=["0", "1"])
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--experiment-root", type=Path, default=None)
    ap.add_argument("--checkpoint-roots", nargs="*", type=Path, default=None,
                    help="Checkpoint banks searched in order. Defaults to <project>/runs_paper and <project>/runs.")
    ap.add_argument("--train-missing", action="store_true",
                    help="Train a missing supervised checkpoint inside this experiment folder. Never needed for zero-shot models.")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-rep-samples", type=int, default=256)
    ap.add_argument("--diagnostic-node-series-budget", type=int, default=8192,
                    help="Approximate per-dataset cap: selected_windows x num_nodes. Same subset for every model.")
    ap.add_argument("--diagnostic-max-windows", type=int, default=64,
                    help="Hard cap on uniformly sampled test windows per dataset.")
    ap.add_argument("--diagnostic-min-windows", type=int, default=16)
    ap.add_argument("--foundation-concurrency", type=int, default=1,
                    help="Maximum number of zero-shot foundation jobs running simultaneously across GPUs.")
    ap.add_argument("--speed-missing-policy", choices=["masked_zero", "raw", "interpolate"], default="masked_zero")
    ap.add_argument("--speed-missing-input-fill", choices=["keep_zero", "mean"], default="keep_zero")
    ap.add_argument("--foundation-cache-dir", type=Path, default=None)
    ap.add_argument("--foundation-num-samples", type=int, default=100)
    ap.add_argument("--foundation-worker-startup-timeout", type=int, default=180)
    ap.add_argument("--foundation-worker-infer-timeout", type=int, default=600)
    ap.add_argument("--foundation-max-split-depth", type=int, default=8)
    ap.add_argument("--foundation-single-window-retries", type=int, default=1)
    ap.add_argument("--no-foundation-adaptive-split", action="store_false", dest="foundation_adaptive_split")
    ap.set_defaults(foundation_adaptive_split=True)
    ap.add_argument("--timeout-hours", type=float, default=36.0)
    ap.add_argument("--foundation-timeout-hours", type=float, default=24.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=120.0)
    # Safe defaults used by the clean benchmark scheduler.
    ap.add_argument("--chronos-chunk", type=int, default=64)
    ap.add_argument("--timesfm-chunk", type=int, default=128)
    ap.add_argument("--timesfm-per-core-batch", type=int, default=32)
    ap.add_argument("--moirai-batch", type=int, default=64)
    ap.add_argument("--moirai-request-chunk", type=int, default=256)
    ap.add_argument("--lagllama-batch", type=int, default=32)
    ap.add_argument("--lagllama-request-chunk", type=int, default=64)
    args = ap.parse_args()

    args.project_root = args.project_root.expanduser().resolve()
    if args.experiment_root is None:
        args.experiment_root = args.project_root / "experiment_results" / "observational_diagnostics"
    args.experiment_root = args.experiment_root.expanduser().resolve()
    args.work_run_root = args.experiment_root / "work_runs"
    args.log_root = args.experiment_root / "logs"
    args.status_root = args.experiment_root / "status"
    args.partial_backup_root = args.experiment_root / "partial_backups"
    args.foundation_cache_dir = (
        (args.project_root / "cache" / "foundation_inference_paper")
        if args.foundation_cache_dir is None else args.foundation_cache_dir.expanduser().resolve()
    )
    if args.checkpoint_roots is None:
        checkpoint_roots = [args.project_root / "runs_paper", args.project_root / "runs"]
    else:
        checkpoint_roots = [p.expanduser().resolve() for p in args.checkpoint_roots]

    for p in [args.experiment_root, args.work_run_root, args.log_root, args.status_root,
              args.partial_backup_root, args.foundation_cache_dir]:
        p.mkdir(parents=True, exist_ok=True)

    models = normalize_models(args.models)
    datasets = normalize_datasets(args.datasets)
    track = args.track.upper()
    model_seeds, intervention_seeds = resolve_protocol(args)

    jobs = [Job(d, m, track, s) for d in datasets for m in models for s in model_seeds]
    jobs.sort(key=cost)

    manifest = {
        "section": "5.1 DMIS--RD Association",
        "created_at_utc": utc_now(),
        "protocol": args.protocol,
        "models": models,
        "datasets": datasets,
        "track": track,
        "model_seeds": model_seeds,
        "intervention_seeds": intervention_seeds,
        "structural_types": STRUCTURAL_TYPES,
        "levels_per_intervention_seed": LEVELS_PER_SEED,
        "expected_rows_per_job": LEVELS_PER_SEED * len(intervention_seeds),
        "expected_jobs": len(jobs),
        "expected_raw_rows": len(jobs) * LEVELS_PER_SEED * len(intervention_seeds),
        "experiment_root": str(args.experiment_root),
        "checkpoint_roots": [str(p) for p in checkpoint_roots],
        "train_missing": bool(args.train_missing),
        "diagnostic_node_series_budget": int(args.diagnostic_node_series_budget),
        "diagnostic_max_windows": int(args.diagnostic_max_windows),
        "diagnostic_min_windows": int(args.diagnostic_min_windows),
        "foundation_num_samples": int(args.foundation_num_samples),
        "foundation_concurrency": int(args.foundation_concurrency),
        "note": (
            "primary_fast intentionally uses model seed 0 and intervention seed 0, while preserving all 10 "
            "structural family/intensity configurations on a deterministic whole-period diagnostic subset. "
            "This is a compute-reduced main association protocol; multi-seed/full-test checks should be reported separately."
        ),
    }
    dump_json(args.experiment_root / "experiment_manifest.json", manifest)

    print("=" * 96)
    print("CausalGap Section 5.1 — DMIS–RD Association")
    print(f"Protocol           : {args.protocol}")
    print(f"Track              : {track}")
    print(f"Models             : {len(models)}")
    print(f"Datasets           : {len(datasets)}")
    print(f"Model seeds        : {model_seeds}")
    print(f"Intervention seeds : {intervention_seeds}")
    print(f"Expected jobs      : {len(jobs)}")
    print(f"Expected raw rows  : {manifest['expected_raw_rows']}")
    print(f"Diagnostic budget  : {args.diagnostic_node_series_budget} node-series, max {args.diagnostic_max_windows} windows")
    print(f"Foundation concur. : {args.foundation_concurrency}")
    print(f"Experiment root    : {args.experiment_root}")
    print("=" * 96, flush=True)

    q: queue.Queue = queue.Queue()
    for job in jobs:
        q.put(job)
    failures: list[dict] = []
    lock = threading.Lock()
    if args.foundation_concurrency < 1:
        raise ValueError("--foundation-concurrency must be >= 1")
    foundation_sem = threading.Semaphore(args.foundation_concurrency)
    threads = [
        threading.Thread(
            target=worker,
            args=(str(gpu), q, args, intervention_seeds, checkpoint_roots, failures, lock, foundation_sem),
            daemon=False,
        )
        for gpu in args.gpus
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = {
        "finished_at_utc": utc_now(),
        "jobs_total": len(jobs),
        "jobs_failed": len(failures),
        "failures": failures,
    }
    dump_json(args.experiment_root / "run_summary.json", summary)
    if args.dry_run:
        print(f"\n[SUMMARY] Dry run only; {len(jobs)} jobs were planned and no experiment was executed.")
        return
    if failures:
        print(f"\n[SUMMARY] {len(failures)}/{len(jobs)} jobs failed. See {args.experiment_root/'run_summary.json'}")
        raise SystemExit(2)
    print(f"\n[SUMMARY] All {len(jobs)} jobs passed.")
    print(f"Next: python -u {Path(__file__).with_name('analyze_diagnostics.py')} --experiment-root {args.experiment_root}")


if __name__ == "__main__":
    main()
