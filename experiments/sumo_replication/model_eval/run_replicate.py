#!/usr/bin/env python3
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

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from protocol import ALL_MODELS, DEFAULT_EXPERIMENT_ROOT, FOUNDATION_MODELS, H1_LEVELS, family_of, json_dump  # noqa: E402


@dataclass(frozen=True)
class Job:
    setting: str
    model: str
    model_seed: int

    @property
    def key(self):
        return f"{self.setting}__{self.model}__trackB__seed{self.model_seed}"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def expected_rows(eval_seeds):
    return sum(len(v) for v in H1_LEVELS.values()) * len(eval_seeds)


def complete(run_dir: Path, eval_seeds: list[int]):
    p = run_dir / "causalgap_results.csv"
    if not p.exists(): return False, "missing causalgap_results.csv"
    try: d = pd.read_csv(p)
    except Exception as e: return False, f"unreadable csv: {e}"
    n = expected_rows(eval_seeds)
    if len(d) != n: return False, f"rows={len(d)} expected={n}"
    req = {"intervention_type", "paper_intensity_level", "intervention_seed", "RD", "representation_response_raw", "environment_distance"}
    if not req.issubset(d.columns): return False, f"missing columns {sorted(req-set(d.columns))}"
    if set(pd.to_numeric(d["intervention_seed"], errors="coerce").dropna().astype(int)) != set(eval_seeds):
        return False, "seed coverage mismatch"
    keys = d[["intervention_type", "paper_intensity_level", "intervention_seed"]].astype(str)
    if len(keys.drop_duplicates()) != n: return False, "duplicate intervention keys"
    expected = {(kind, item["level"], str(seed))
                for kind, items in H1_LEVELS.items()
                for item in items for seed in eval_seeds}
    observed = set(map(tuple, keys.to_numpy()))
    if observed != expected:
        return False, f"intervention identity mismatch; missing={len(expected-observed)} extra={len(observed-expected)}"
    numeric = d[["RD", "representation_response_raw"]].apply(pd.to_numeric, errors="coerce")
    if not numeric.notna().all().all(): return False, "missing or nonfinite primary metrics"
    import numpy as np
    if not np.isfinite(numeric.to_numpy()).all(): return False, "nonfinite primary metrics"
    return True, "complete"


def job_cost(j: Job):
    foundation = 1 if j.model in FOUNDATION_MODELS else 0
    order = {
        "STGCN": 10, "DLinear": 11, "AGCRN": 12, "GWNet": 13, "DCRNN": 14,
        "PatchTST": 20, "iTransformer": 21, "TimesNet": 22, "FEDformer": 23, "Autoformer": 24,
        "ASTGNN": 25, "STEVE": 26, "DCGCN": 27, "ST-Norm": 28, "CCHMM": 29,
        "CausalGRIT": 30, "TF-MVGNN": 31,
        "Chronos": 80, "TimesFM": 85, "Moirai": 90, "Lag-Llama": 100,
    }.get(j.model, 50)
    setting = 0 if j.setting == "sumo_metrla" else 1  # abstract SUMO jobs
    return foundation, order, setting


def tail(path: Path, n=40):
    if not path.exists(): return ""
    try: return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except Exception: return ""


def worker(gpu, q, args, failures, lock, foundation_sem):
    while True:
        try: job = q.get_nowait()
        except queue.Empty: return
        token = False
        if job.model in FOUNDATION_MODELS:
            with lock: print(f"[WAIT-FOUNDATION][GPU {gpu}] {job.key}", flush=True)
            foundation_sem.acquire(); token = True
        rd = args.experiment_root / "runs" / job.setting / job.model / "trackB" / f"seed{job.model_seed}"
        status = args.experiment_root / "status" / f"{job.key}.json"
        ok, why = complete(rd, args.eval_seeds)
        if ok and not args.rerun:
            with lock: print(f"[SKIP][GPU {gpu}] {job.key} already complete", flush=True)
            json_dump(status, {"status":"PASS","reason":"already complete","job":job.__dict__,"checked":utc_now()})
            if token: foundation_sem.release()
            q.task_done(); continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        env["CAUSALGAP_ROOT"] = str(args.project_root)
        # Worker-specific chunk controls inherited from the Section 5.1 speed fixes.
        env.setdefault("CAUSALGAP_CHRONOS_CHUNK", str(args.chronos_chunk))
        env.setdefault("CAUSALGAP_TIMESFM_CHUNK", str(args.timesfm_chunk))
        env.setdefault("CAUSALGAP_TIMESFM_PER_CORE_BATCH", str(args.timesfm_per_core_batch))
        env.setdefault("CAUSALGAP_MOIRAI_PRED_BATCH", str(args.moirai_batch))
        env.setdefault("CAUSALGAP_MOIRAI_REQUEST_CHUNK", str(args.moirai_request_chunk))
        env.setdefault("CAUSALGAP_LAGLLAMA_PRED_BATCH", str(args.lagllama_batch))
        env.setdefault("CAUSALGAP_LAGLLAMA_REQUEST_CHUNK", str(args.lagllama_request_chunk))

        cmd = [
            sys.executable, "-u", str(EXP_DIR / "evaluate_replication_model.py"),
            "--setting", job.setting,
            "--model", job.model,
            "--model-seed", str(job.model_seed),
            "--device", "cuda",
            "--profile", args.profile,
            "--project-root", str(args.project_root),
            "--experiment-root", str(args.experiment_root),
            "--eval-seeds", *map(str, args.eval_seeds),
            "--artifact-level", args.artifact_level,
            "--max-rep-samples", str(args.max_rep_samples),
            "--foundation-num-samples", str(args.foundation_num_samples),
            "--foundation-worker-startup-timeout", str(args.foundation_worker_startup_timeout),
            "--foundation-worker-infer-timeout", str(args.foundation_worker_infer_timeout),
            "--foundation-max-split-depth", str(args.foundation_max_split_depth),
            "--foundation-single-window-retries", str(args.foundation_single_window_retries),
            "--foundation-cache",
            "--foundation-cache-dir", str(args.foundation_cache_dir),
        ]
        if args.train_missing: cmd.append("--train-missing")
        if args.rerun: cmd.append("--rerun")
        if args.checkpoint_roots:
            cmd += ["--checkpoint-roots", *map(str, args.checkpoint_roots)]
        if not args.foundation_adaptive_split: cmd.append("--no-foundation-adaptive-split")

        log = args.experiment_root / "logs" / f"{job.key}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic(); started = utc_now()
        with lock: print(f"[START][GPU {gpu}] {job.key} | log={log}", flush=True)
        try:
            with log.open("a", encoding="utf-8") as lf:
                lf.write("\n# ===== NEW ATTEMPT =====\n# COMMAND: " + " ".join(cmd) + "\n"); lf.flush()
                proc = subprocess.Popen(cmd, cwd=CODE_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
                timeout_h = args.foundation_timeout_hours if job.model in FOUNDATION_MODELS else args.timeout_hours
                last = 0.0
                while proc.poll() is None:
                    time.sleep(5)
                    elapsed = time.monotonic() - t0
                    if elapsed - last >= args.heartbeat_seconds:
                        with lock: print(f"[RUNNING][GPU {gpu}] {job.key} elapsed={elapsed/60:.1f} min", flush=True)
                        last = elapsed
                    if timeout_h > 0 and elapsed > timeout_h * 3600:
                        proc.terminate()
                        try: proc.wait(timeout=30)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                        raise TimeoutError(f"job exceeded {timeout_h:g} h")
                rc = int(proc.returncode)
            ok, why = complete(rd, args.eval_seeds)
            if rc != 0 or not ok:
                raise RuntimeError(f"rc={rc}; {why}\n{tail(log)}")
            rec = {"status":"PASS","job":job.__dict__,"gpu":str(gpu),"started":started,"finished":utc_now(),"wall_seconds":time.monotonic()-t0,"run_dir":str(rd),"log":str(log)}
            json_dump(status, rec)
            with lock: print(f"[PASS][GPU {gpu}] {job.key}", flush=True)
        except Exception as e:
            rec = {"status":"FAIL","job":job.__dict__,"gpu":str(gpu),"started":started,"finished":utc_now(),"wall_seconds":time.monotonic()-t0,"reason":repr(e),"run_dir":str(rd),"log":str(log)}
            json_dump(status, rec); failures.append(rec)
            with lock:
                print(f"[FAIL][GPU {gpu}] {job.key}: {e}", flush=True)
        finally:
            if token: foundation_sem.release()
            q.task_done()


def main():
    ap = argparse.ArgumentParser(description="GPU scheduler: abstract SUMO replication only (not H1, not steering).")
    ap.add_argument("--project-root", type=Path, default=Path("/data/wanganna/CausalGap"))
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    ap.add_argument("--settings", nargs="+", choices=["synthetic", "sumo_metrla"], default=["synthetic", "sumo_metrla"])
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--model-seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=[0,1,2,3,4])
    ap.add_argument("--gpus", nargs="+", default=["0","1"])
    ap.add_argument("--profile", choices=["paper","smoke"], default="paper")
    ap.add_argument("--artifact-level", choices=["minimal","full"], default="full")
    ap.add_argument("--max-rep-samples", type=int, default=64)
    ap.add_argument("--train-missing", action="store_true")
    ap.add_argument("--checkpoint-roots", nargs="*", type=Path, default=None)
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
    ap.add_argument("--timesfm-chunk", type=int, default=64)
    ap.add_argument("--timesfm-per-core-batch", type=int, default=64)
    ap.add_argument("--moirai-batch", type=int, default=64)
    ap.add_argument("--moirai-request-chunk", type=int, default=32)
    ap.add_argument("--lagllama-batch", type=int, default=64)
    ap.add_argument("--lagllama-request-chunk", type=int, default=32)
    args = ap.parse_args()
    if args.settings != ['sumo_metrla']:
        raise SystemExit('ABSTRACT SUMO EXCLUDED FROM H1: only --settings sumo_metrla is supported here')

    if args.foundation_cache_dir is None:
        args.foundation_cache_dir = args.project_root / "cache" / "foundation_inference"
    args.experiment_root.mkdir(parents=True, exist_ok=True)

    # Fail before consuming GPUs if environments are missing.
    for s in args.settings:
        folder = "synthetic" if s == "synthetic" else "sumo_metrla"
        m = args.experiment_root / "environments" / folder / "manifest.csv"
        if not m.exists():
            raise FileNotFoundError(f"Environment manifest missing for {s}: {m}")

    jobs = [Job(s, m, seed) for s in args.settings for m in args.models for seed in args.model_seeds]
    jobs.sort(key=job_cost)
    q = queue.Queue()
    for j in jobs: q.put(j)
    failures = []; lock = threading.Lock()
    sem = threading.Semaphore(max(1, int(args.foundation_concurrency)))
    threads = [threading.Thread(target=worker, args=(g,q,args,failures,lock,sem), daemon=False) for g in args.gpus]

    print("="*90)
    print("CausalGap Section 5.2 abstract SUMO — regime replication ONLY")
    print("Settings          :", args.settings)
    print("Models            :", len(args.models))
    print("Model seeds       :", args.model_seeds)
    print("Intervention seeds:", args.eval_seeds)
    print("Expected jobs     :", len(jobs))
    print("Expected rows/job :", expected_rows(args.eval_seeds))
    print("GPUs              :", args.gpus)
    print("Experiment root   :", args.experiment_root)
    print("="*90, flush=True)

    for t in threads: t.start()
    for t in threads: t.join()

    statuses = []
    for p in sorted((args.experiment_root / "status").glob("*.json")) if (args.experiment_root / "status").exists() else []:
        try: statuses.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception: pass
    json_dump(args.experiment_root / "run_summary.json", {
        "created_at": utc_now(), "jobs_requested": len(jobs), "failures": failures,
        "settings": args.settings, "models": args.models, "model_seeds": args.model_seeds,
        "eval_seeds": args.eval_seeds, "statuses": statuses,
    })
    if failures:
        print(f"[SUMMARY] {len(failures)}/{len(jobs)} jobs failed. See {args.experiment_root/'run_summary.json'}")
        raise SystemExit(2)
    print(f"[SUMMARY] All {len(jobs)} replication jobs passed.")
    print("Next: validate all model CSVs, then run 05_regime_replication.py.")
    print("DO NOT invoke legacy H1 regression or steering on SUMO results.")


if __name__ == "__main__":
    main()
