#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-GPU resumable scheduler for Controlled-Synthetic DSAI Track A."""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
H1_DIR = CODE_ROOT / "experiments" / "incremental_value"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(H1_DIR) not in sys.path:
    sys.path.insert(0, str(H1_DIR))

from protocol import ALL_MODELS, FOUNDATION_MODELS, H1_LEVELS  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def expected_rows(eval_seeds):
    return sum(len(v) for v in H1_LEVELS.values()) * len(eval_seeds)


def complete(run_dir: Path, eval_seeds: list[int]):
    p = run_dir / "causalgap_results.csv"
    if not p.exists():
        return False, "missing causalgap_results.csv"
    try:
        d = pd.read_csv(p)
    except Exception as e:
        return False, f"unreadable csv: {e}"
    n = expected_rows(eval_seeds)
    if len(d) != n:
        return False, f"rows={len(d)} expected={n}"
    req = {"evaluation_track", "intervention_type", "paper_intensity_level", "intervention_seed", "RD", "DMIS", "representation_response_raw", "environment_distance"}
    if not req.issubset(d.columns):
        return False, f"missing columns {sorted(req-set(d.columns))}"
    if set(d["evaluation_track"].astype(str).str.upper()) != {"A"}:
        return False, "not Track A"
    if set(pd.to_numeric(d["intervention_seed"], errors="coerce").dropna().astype(int)) != set(eval_seeds):
        return False, "intervention seed coverage mismatch"
    keys = d[["intervention_type", "paper_intensity_level", "intervention_seed"]].astype(str)
    if keys.duplicated().any():
        return False, "duplicate intervention keys"
    for c in ["RD", "DMIS", "representation_response_raw", "environment_distance"]:
        if not np.isfinite(pd.to_numeric(d[c], errors="coerce")).all():
            return False, f"non-finite {c}"
    return True, "complete"


@dataclass(frozen=True)
class Job:
    model: str
    seed: int

    @property
    def key(self):
        return f"synthetic__{self.model}__trackA__seed{self.seed}"


def cost(j: Job):
    foundation = 1 if j.model in FOUNDATION_MODELS else 0
    order = {
        "STGCN":10,"DLinear":11,"AGCRN":12,"GWNet":13,"DCRNN":14,
        "PatchTST":20,"iTransformer":21,"TimesNet":22,"FEDformer":23,"Autoformer":24,
        "ASTGNN":25,"STEVE":26,"DCGCN":27,"ST-Norm":28,"CCHMM":29,
        "CausalGRIT":30,"TF-MVGNN":31,"Chronos":80,"TimesFM":85,"Moirai":90,"Lag-Llama":100,
    }.get(j.model, 50)
    return foundation, order


def tail(path: Path, n=50):
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def worker(gpu, q, args, failures, lock, foundation_sem):
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        token = False
        try:
            if job.model in FOUNDATION_MODELS:
                foundation_sem.acquire(); token = True
            rd = args.output_root / "runs" / "synthetic" / job.model / "trackA" / f"seed{job.seed}"
            ok, why = complete(rd, args.eval_seeds)
            if ok and not args.rerun:
                with lock:
                    print(f"[SKIP][GPU {gpu}] {job.key} already complete", flush=True)
                q.task_done()
                if token: foundation_sem.release(); token = False
                continue

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            env["CAUSALGAP_ROOT"] = str(args.project_root)
            env.setdefault("CAUSALGAP_CHRONOS_CHUNK", str(args.chronos_chunk))
            env.setdefault("CAUSALGAP_TIMESFM_CHUNK", str(args.timesfm_chunk))
            env.setdefault("CAUSALGAP_TIMESFM_PER_CORE_BATCH", str(args.timesfm_per_core_batch))
            env.setdefault("CAUSALGAP_MOIRAI_PRED_BATCH", str(args.moirai_batch))
            env.setdefault("CAUSALGAP_MOIRAI_REQUEST_CHUNK", str(args.moirai_request_chunk))
            env.setdefault("CAUSALGAP_LAGLLAMA_PRED_BATCH", str(args.lagllama_batch))
            env.setdefault("CAUSALGAP_LAGLLAMA_REQUEST_CHUNK", str(args.lagllama_request_chunk))

            cmd = [
                sys.executable, "-u", str(EXP_DIR / "evaluate_synthetic_track_a.py"),
                "--setting", "synthetic", "--model", job.model,
                "--model-seed", str(job.seed), "--device", "cuda",
                "--profile", args.profile,
                "--project-root", str(args.project_root),
                "--experiment-root", str(args.source_experiment_root),
                "--output-root", str(args.output_root),
                "--eval-seeds", *map(str, args.eval_seeds),
                "--artifact-level", args.artifact_level,
                "--max-rep-samples", str(args.max_rep_samples),
                "--foundation-num-samples", str(args.foundation_num_samples),
                "--foundation-worker-startup-timeout", str(args.foundation_worker_startup_timeout),
                "--foundation-worker-infer-timeout", str(args.foundation_worker_infer_timeout),
                "--foundation-max-split-depth", str(args.foundation_max_split_depth),
                "--foundation-single-window-retries", str(args.foundation_single_window_retries),
                "--foundation-cache", "--foundation-cache-dir", str(args.foundation_cache_dir),
            ]
            if args.train_missing:
                cmd.append("--train-missing")
            if args.rerun:
                cmd.append("--rerun")
            if not args.foundation_adaptive_split:
                cmd.append("--no-foundation-adaptive-split")

            log_path = args.output_root / "logs" / f"{job.key}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            status_path = args.output_root / "status" / f"{job.key}.json"
            t0 = time.monotonic()
            with lock:
                print(f"[START][GPU {gpu}] {job.key} | log={log_path}", flush=True)
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write("\n# ===== NEW ATTEMPT =====\n# COMMAND: " + " ".join(cmd) + "\n"); lf.flush()
                proc = subprocess.Popen(cmd, cwd=CODE_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
                timeout_h = args.foundation_timeout_hours if job.model in FOUNDATION_MODELS else args.timeout_hours
                last = 0.0
                while proc.poll() is None:
                    time.sleep(5)
                    elapsed = time.monotonic() - t0
                    if elapsed-last >= args.heartbeat_seconds:
                        with lock:
                            print(f"[RUNNING][GPU {gpu}] {job.key} elapsed={elapsed/60:.1f} min", flush=True)
                        last = elapsed
                    if timeout_h > 0 and elapsed > timeout_h*3600:
                        proc.terminate()
                        try: proc.wait(timeout=30)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                        raise TimeoutError(f"job exceeded {timeout_h:g} h")
                rc = int(proc.returncode)
            ok, why = complete(rd, args.eval_seeds)
            if rc != 0 or not ok:
                raise RuntimeError(f"rc={rc}; {why}\n{tail(log_path)}")
            rec = {"status":"PASS","job":job.__dict__,"gpu":str(gpu),"finished":utc_now(),"run_dir":str(rd),"log":str(log_path)}
            safe_json(status_path, rec)
            with lock: print(f"[PASS][GPU {gpu}] {job.key}", flush=True)
        except Exception as e:
            rec = {"status":"FAIL","job":job.__dict__,"gpu":str(gpu),"finished":utc_now(),"reason":repr(e)}
            safe_json(args.output_root / "status" / f"{job.key}.json", rec)
            failures.append(rec)
            with lock: print(f"[FAIL][GPU {gpu}] {job.key}: {e}", flush=True)
        finally:
            if token:
                foundation_sem.release()
            q.task_done()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path("/data/wanganna/CausalGap"))
    ap.add_argument("--source-experiment-root", type=Path, default=Path("/data/wanganna/CausalGap/experiment_results/incremental_value"))
    ap.add_argument("--output-root", type=Path, default=Path("/data/wanganna/CausalGap/experiment_results/access_sensitivity/synthetic_track_a"))
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--model-seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=[0,1,2,3,4])
    ap.add_argument("--gpus", nargs="+", default=["0","1"])
    ap.add_argument("--profile", choices=["paper","smoke"], default="paper")
    ap.add_argument("--artifact-level", choices=["minimal","full"], default="minimal")
    ap.add_argument("--max-rep-samples", type=int, default=32)
    ap.add_argument("--train-missing", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--timeout-hours", type=float, default=36)
    ap.add_argument("--foundation-timeout-hours", type=float, default=96)
    ap.add_argument("--heartbeat-seconds", type=float, default=120)
    ap.add_argument("--foundation-concurrency", type=int, default=1)
    ap.add_argument("--foundation-num-samples", type=int, default=100)
    ap.add_argument("--foundation-cache-dir", type=Path, default=None)
    ap.add_argument("--foundation-worker-startup-timeout", type=int, default=180)
    ap.add_argument("--foundation-worker-infer-timeout", type=int, default=600)
    ap.add_argument("--foundation-max-split-depth", type=int, default=8)
    ap.add_argument("--foundation-single-window-retries", type=int, default=1)
    ap.add_argument("--no-foundation-adaptive-split", action="store_false", dest="foundation_adaptive_split")
    ap.set_defaults(foundation_adaptive_split=True)
    ap.add_argument("--chronos-chunk", type=int, default=64)
    ap.add_argument("--timesfm-chunk", type=int, default=128)
    ap.add_argument("--timesfm-per-core-batch", type=int, default=32)
    ap.add_argument("--moirai-batch", type=int, default=64)
    ap.add_argument("--moirai-request-chunk", type=int, default=256)
    ap.add_argument("--lagllama-batch", type=int, default=32)
    ap.add_argument("--lagllama-request-chunk", type=int, default=64)
    args = ap.parse_args()

    args.project_root = args.project_root.resolve()
    args.source_experiment_root = args.source_experiment_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.foundation_cache_dir = (args.project_root / "cache" / "foundation_inference") if args.foundation_cache_dir is None else args.foundation_cache_dir.resolve()

    unknown = sorted(set(args.models)-set(ALL_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    jobs = [Job(m,s) for m in args.models for s in args.model_seeds]
    jobs.sort(key=cost)
    q = queue.Queue()
    for j in jobs: q.put(j)
    failures=[]; lock=threading.Lock(); sem=threading.Semaphore(max(1,args.foundation_concurrency))
    threads=[]
    for g in args.gpus:
        t=threading.Thread(target=worker,args=(g,q,args,failures,lock,sem),daemon=True); t.start(); threads.append(t)
    for t in threads: t.join()

    summary={"section":"5.8 DSAI Synthetic Track A","jobs":len(jobs),"failures":failures,"finished":utc_now()}
    safe_json(args.output_root/"run_summary.json",summary)
    if failures:
        print(f"[DONE] failures={len(failures)}; see {args.output_root/'run_summary.json'}", flush=True)
        raise SystemExit(2)
    print(f"[DONE] all {len(jobs)} Track-A jobs complete -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
