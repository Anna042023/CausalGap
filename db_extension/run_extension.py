#!/usr/bin/env python3
"""Paired randomized database blocks and single-host resource sensitivity.

All live database files and mutable measurement state remain in a private /tmp
directory. The final records are copied out only after every connection and
worker has closed. No forecasting observations or H1/H2 run records are used.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "db_experiments"))
import duckdb
import benchmark as r2
from queries import naive_sql, checked_sql, cached_sql, pair_rows_sql

QUERIES = {"naive": naive_sql(), "checked": checked_sql(), "cached": cached_sql()}
CONFIGS = [
    {"config": "t1_m1024", "threads": 1, "memory_mib": 1024},
    {"config": "t2_m1024", "threads": 2, "memory_mib": 1024},
    {"config": "t4_m1024", "threads": 4, "memory_mib": 1024},
    {"config": "t1_m64", "threads": 1, "memory_mib": 64},
    {"config": "t1_m256", "threads": 1, "memory_mib": 256},
]
ORDER_SEED = 2026110703


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    columns = sorted(set().union(*(set(r) for r in rows)))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list))
                        else v for k, v in row.items()})


def read_proc_memory():
    result = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                result[line.split(":")[0]] = int(line.split()[1])
    except OSError:
        pass
    return result


def host_snapshot():
    result = {"utc": utcnow(), "python": sys.version,
              "python_executable": sys.executable, "sqlite": sqlite3.sqlite_version,
              "duckdb": duckdb.__version__, "platform": platform.platform(),
              "logical_cpus": os.cpu_count(),
              "affinity": sorted(os.sched_getaffinity(0)),
              "load_average": list(os.getloadavg()),
              "tmp_disk_usage": list(shutil.disk_usage("/tmp"))}
    for key, path in [
        ("cpuinfo", "/proc/cpuinfo"), ("meminfo", "/proc/meminfo"),
        ("cgroup_cpu_max", "/sys/fs/cgroup/cpu.max"),
        ("cgroup_memory_max", "/sys/fs/cgroup/memory.max"),
        ("cgroup_memory_current", "/sys/fs/cgroup/memory.current"),
        ("cgroup_cpu_stat", "/sys/fs/cgroup/cpu.stat"),
    ]:
        try:
            result[key] = Path(path).read_text()
        except OSError as e:
            result[key] = str(e)
    return result


def measure(connection, engine, sql, timeout, expected, reset_hwm=False):
    """Time execute+fetchall only; verify every successful result afterwards."""
    reset_status = "not_requested"
    if reset_hwm:
        try:
            # Linux value 5 resets only this process's resident high-water mark.
            with open("/proc/self/clear_refs", "w") as f:
                f.write("5\n")
            reset_status = "ok"
        except OSError as e:
            reset_status = f"{type(e).__name__}: {e}"
    before = read_proc_memory()
    timer = None
    deadline = time.monotonic() + timeout
    if engine == "SQLite":
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    else:
        timer = threading.Timer(timeout, connection.interrupt)
        timer.daemon = True
        timer.start()
    rows = []
    error = ""
    status = "ok"
    started = utcnow()
    c0 = time.process_time_ns()
    t0 = time.perf_counter_ns()
    try:
        rows = connection.execute(sql).fetchall()
    except Exception as exc:
        status = "query_failure"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        cpu_ms = (time.process_time_ns() - c0) / 1e6
        after = read_proc_memory()
        lifetime_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if engine == "SQLite":
            connection.set_progress_handler(None, 0)
        elif timer is not None:
            timer.cancel()
            timer.join()
    record = {
        "started_utc": started, "wall_ms": wall_ms, "cpu_ms": cpu_ms,
        "status": status, "error": error, "output_runs": None,
        "scored_pairs": None, "output_sha256": None,
        "oracle_status": "not_run", "rss_start_kib": before.get("VmRSS"),
        "peak_rss_kib": after.get("VmHWM", lifetime_peak),
        "process_lifetime_peak_rss_kib": lifetime_peak,
        "hwm_reset_status": reset_status,
        "peak_rss_scope": "query_window_including_warm_resident_state"
        if reset_status == "ok" else "process_lifetime_including_startup_and_warmup",
    }
    if status == "ok":
        try:
            digest, pairs = r2.verify(rows, expected)
            record.update(oracle_status="pass", output_runs=len(rows),
                          scored_pairs=pairs, output_sha256=digest)
        except Exception as exc:
            record.update(status="oracle_failure", oracle_status="fail",
                          error=f"{type(exc).__name__}: {exc}")
    return record


def worker(payload):
    os.sched_setaffinity(0, set(payload["affinity"]))
    config = payload["config"]
    connection = None
    result = {"worker_pid": os.getpid(), "records": []}
    try:
        connection = duckdb.connect(payload["database"], read_only=True)
        connection.execute(f"SET threads={int(config['threads'])}")
        connection.execute(f"SET memory_limit='{int(config['memory_mib'])}MiB'")
        connection.execute("SET enable_progress_bar=false")
        settings = dict(zip(["threads", "memory_limit", "temp_directory"],
            connection.execute("SELECT current_setting('threads'), "
                               "current_setting('memory_limit'), "
                               "current_setting('temp_directory')").fetchone()))
        expected = json.loads(Path(payload["expected"]).read_text())
        for phase in ["warmup", "measured"]:
            record = measure(connection, "DuckDB", QUERIES[payload["query"]],
                             payload["timeout"], expected, reset_hwm=True)
            record.update(phase=phase, config=config["config"],
                          threads=config["threads"], memory_mib=config["memory_mib"],
                          query=payload["query"], block=payload["block"],
                          order_in_block=payload["order_in_block"],
                          affinity=sorted(os.sched_getaffinity(0)),
                          worker_pid=os.getpid(), actual_settings=settings)
            result["records"].append(record)
    except Exception as exc:
        result["worker_error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if connection is not None:
            connection.close()
    return result


def save_plans(connections, out):
    for variant, (connection, engine) in connections.items():
        directory = out / "plans" / variant
        directory.mkdir(parents=True)
        prefix = "EXPLAIN QUERY PLAN " if engine == "SQLite" else "EXPLAIN "
        for name, sql in QUERIES.items():
            rows = connection.execute(prefix + sql).fetchall()
            (directory / f"{name}.sql").write_text(sql + ";\n")
            (directory / f"{name}.txt").write_text(
                "\n".join(" | ".join(map(str, r)) for r in rows) + "\n")


def main(args):
    destination = ROOT / args.output
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite: {destination}")
    runtime = Path(tempfile.mkdtemp(prefix="cg_db_r3_"))
    out = runtime / "results"
    out.mkdir()
    database_dir = runtime / "databases"
    database_dir.mkdir()
    original_affinity = sorted(os.sched_getaffinity(0))
    primary_cpu = original_affinity[-1]
    worker_affinity = original_affinity[-min(4, len(original_affinity)):]
    configuration = {
        "study": "New controlled synthetic relational experiment; no forecasting H1/H2 records",
        "blocks": args.blocks, "hardware_blocks": args.hardware_blocks,
        "primary_order_seed": ORDER_SEED, "hardware_order_seed": ORDER_SEED + 1,
        "timing": "perf_counter_ns execute plus fetchall; excludes verification/setup/startup",
        "cpu_timing": "process_time_ns across all worker threads; execute plus fetchall",
        "primary_design": "20 randomized complete blocks of all nine engine-query combinations",
        "primary_warmups": "one randomized complete warmup block, excluded",
        "hardware_design": "20 randomized complete blocks of all ten config-query combinations",
        "hardware_processes": "fresh sequential process per config/query/block; one same-query warmup",
        "rss_method": "VmHWM after reset through /proc/self/clear_refs=5 immediately before query; fallback explicitly marked",
        "hardware_memory_units": "MiB; 1024 MiB = 1 GiB, five deduplicated configurations",
        "original_duckdb_memory": "1GB (decimal), unchanged from R2 primary benchmark",
        "hardware_configs": CONFIGS, "primary_cpu_affinity": [primary_cpu],
        "hardware_worker_affinity": worker_affinity,
        "primary_variants": ["sqlite_auto", "sqlite_indexed", "duckdb_scan"],
        "primary_queries": list(QUERIES), "hardware_queries": ["checked", "cached"],
        "query_timeout_seconds": args.timeout, "cache_policy": "warm OS pages, no cache flush",
        "runtime_directory": str(runtime), "source_input": "inputs/forecast_atoms.csv.gz",
        "limitations": "One shared container/CPU model; configuration sensitivity, not cross-hardware generalization",
    }
    write_json(out / "configuration.json", configuration)
    write_json(out / "hardware_start.json", host_snapshot())
    source_files = [ROOT / "run_extension.py",
                    ROOT.parent / "db_experiments" / "benchmark.py",
                    ROOT.parent / "db_experiments" / "queries.py",
                    ROOT.parent / "db_experiments" / "workload.py",
                    *sorted((ROOT / "inputs").glob("*"))]
    write_json(out / "source_hashes.json", {str(p.relative_to(ROOT.parent)): sha256(p) for p in source_files})
    info = json.loads((ROOT / "inputs/input_manifest.json").read_text())
    expected = json.loads((ROOT / "inputs/expected_results.json").read_text())
    csv_path = runtime / "forecast_atoms.csv"
    with gzip.open(ROOT / "inputs/forecast_atoms.csv.gz", "rb") as src, csv_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    r2.verify_input(csv_path, info["csv_sha256"])
    write_json(out / "input_verification.json", dict(info, expected_output_sha256=r2.checksum(expected),
                                                    observed_csv_sha256=sha256(csv_path)))
    setup = []
    primary_records = []
    hardware_records = []
    connections = {}
    fatal_error = None
    began = time.monotonic()
    try:
        os.sched_setaffinity(0, {primary_cpu})
        sqlite_auto = database_dir / "sqlite_auto.db"
        sqlite_indexed = database_dir / "sqlite_indexed.db"
        duck_path = database_dir / "duckdb_scan.db"
        load_ms, analyze_ms, settings = r2.load_sqlite(sqlite_auto, csv_path, 120)
        write_json(out / "sqlite_settings.json", settings)
        shutil.copy2(sqlite_auto, sqlite_indexed)
        connections["sqlite_auto"] = (sqlite3.connect(str(sqlite_auto)), "SQLite")
        indexed = sqlite3.connect(str(sqlite_indexed))
        connections["sqlite_indexed"] = (indexed, "SQLite")
        _, index_ms = r2.timed_query(indexed, "SQLite", "CREATE UNIQUE INDEX "
            "forecast_pair_key ON forecast_atom (run_id,environment,origin,node_id,horizon)",
            120, True)
        _, index_analyze_ms = r2.timed_query(indexed, "SQLite", "ANALYZE", 120, True)
        duck, duck_load_ms, duck_analyze_ms = r2.load_duck(duck_path, csv_path, 120)
        connections["duckdb_scan"] = (duck, "DuckDB")
        paths = {"sqlite_auto": sqlite_auto, "sqlite_indexed": sqlite_indexed,
                 "duckdb_scan": duck_path}
        for variant, (connection, engine) in connections.items():
            atoms = int(connection.execute("SELECT COUNT(*) FROM forecast_atom").fetchone()[0])
            if atoms != info["forecast_atoms"]:
                raise AssertionError(f"Incomplete input in {variant}: {atoms}")
            before_bytes = paths[variant].stat().st_size
            _, build_ms = r2.timed_query(connection, engine,
                "CREATE TABLE validated_pair AS " + pair_rows_sql(), 120, True)
            pairs = int(connection.execute("SELECT COUNT(*) FROM validated_pair").fetchone()[0])
            if pairs != info["expected_pairs"]:
                raise AssertionError(f"Wrong materialized pairs in {variant}: {pairs}")
            setup.append({"variant": variant, "engine": engine,
                "load_ms": duck_load_ms if engine == "DuckDB" else load_ms,
                "analyze_ms": duck_analyze_ms if engine == "DuckDB" else
                    analyze_ms + (index_analyze_ms if variant == "sqlite_indexed" else 0),
                "index_ms": index_ms if variant == "sqlite_indexed" else 0,
                "materialize_ms": build_ms, "forecast_atoms": atoms,
                "materialized_pairs": pairs, "db_bytes_before_materialization": before_bytes,
                "db_bytes_after_materialization": paths[variant].stat().st_size})
        write_csv(out / "setup_costs.csv", setup)
        write_json(out / "duckdb_primary_settings.json", dict(zip(["threads", "memory_limit"],
            duck.execute("SELECT current_setting('threads'), current_setting('memory_limit')").fetchone())))
        save_plans(connections, out)
        rng = random.Random(ORDER_SEED)
        primary_schedule = []
        for block in range(args.blocks + 1):
            combos = [(v, q) for v in connections for q in QUERIES]
            rng.shuffle(combos)
            primary_schedule.append({"block": block, "phase": "warmup" if block == 0 else "measured",
                                     "order": combos})
            for order, (variant, query) in enumerate(combos):
                connection, engine = connections[variant]
                rec = measure(connection, engine, QUERIES[query], args.timeout, expected)
                rec.update(block=block, phase="warmup" if block == 0 else "measured",
                           order_in_block=order, variant=variant, engine=engine, query=query,
                           forecast_atoms=info["forecast_atoms"], affinity=[primary_cpu])
                primary_records.append(rec)
                append_jsonl(out / "primary_raw.jsonl", rec)
                if rec["status"] == "oracle_failure":
                    raise AssertionError(rec["error"])
            print(f"PRIMARY block {block}/{args.blocks}; {time.monotonic()-began:.1f}s", flush=True)
        write_json(out / "primary_schedule.json", primary_schedule)
        write_csv(out / "primary_raw.csv", primary_records)
        for connection, engine in connections.values():
            if engine == "DuckDB":
                connection.execute("CHECKPOINT")
            connection.close()
        connections.clear()
        # The controller is kept off the four CPUs made available to all workers.
        controller_cpu = original_affinity[0]
        os.sched_setaffinity(0, {controller_cpu})
        rng = random.Random(ORDER_SEED + 1)
        hardware_schedule = []
        for block in range(1, args.hardware_blocks + 1):
            combos = [(c, q) for c in CONFIGS for q in ["checked", "cached"]]
            rng.shuffle(combos)
            hardware_schedule.append({"block": block, "order": [(c["config"], q) for c, q in combos]})
            for order, (config, query) in enumerate(combos):
                payload = {"database": str(duck_path), "expected": str(ROOT / "inputs/expected_results.json"),
                           "config": config, "query": query, "block": block,
                           "order_in_block": order, "timeout": args.timeout,
                           "affinity": worker_affinity}
                t0 = time.monotonic()
                try:
                    proc = subprocess.run([sys.executable, str(ROOT / "run_extension.py"), "--worker"],
                        input=json.dumps(payload), text=True, capture_output=True,
                        timeout=2*args.timeout + 30, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
                    if proc.returncode != 0:
                        raise RuntimeError(f"worker exit {proc.returncode}: {proc.stderr[-5000:]}")
                    output = json.loads(proc.stdout)
                    if "worker_error" in output:
                        raise RuntimeError(output["worker_error"])
                    for rec in output["records"]:
                        hardware_records.append(rec)
                        append_jsonl(out / "hardware_raw.jsonl", rec)
                        if rec["status"] == "oracle_failure":
                            raise AssertionError(rec["error"])
                except Exception as exc:
                    if isinstance(exc, AssertionError):
                        raise
                    rec = dict(config, query=query, block=block, order_in_block=order,
                               phase="measured", status="worker_failure", oracle_status="not_run",
                               error=f"{type(exc).__name__}: {exc}", wall_ms=None, cpu_ms=None,
                               worker_elapsed_ms=1000*(time.monotonic()-t0))
                    hardware_records.append(rec)
                    append_jsonl(out / "hardware_raw.jsonl", rec)
            print(f"HARDWARE block {block}/{args.hardware_blocks}; {time.monotonic()-began:.1f}s", flush=True)
        write_json(out / "hardware_schedule.json", hardware_schedule)
        write_csv(out / "hardware_raw.csv", hardware_records)
    except Exception as exc:
        fatal_error = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        write_json(out / "fatal_error.json", fatal_error)
    finally:
        for connection, _ in connections.values():
            connection.close()
        os.sched_setaffinity(0, set(original_affinity))
        write_csv(out / "primary_raw.csv", primary_records)
        write_csv(out / "hardware_raw.csv", hardware_records)
        write_json(out / "hardware_end.json", host_snapshot())
        write_json(out / "completion.json", {"finished_utc": utcnow(),
            "elapsed_seconds": time.monotonic()-began, "fatal_error": fatal_error,
            "primary_measured": sum(r["phase"] == "measured" for r in primary_records),
            "hardware_measured": sum(r["phase"] == "measured" for r in hardware_records),
            "all_database_connections_closed": True,
            "database_snapshots_retained": False,
            "regeneration": "Persisted input + SQL + code recreate all databases"})
        # There are no live database connections or child processes at this point.
        shutil.copytree(out, destination)
        shutil.rmtree(runtime)
    if fatal_error:
        raise RuntimeError(fatal_error["error"])
    print(f"DONE: closed measurements copied to {destination}", flush=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        print(json.dumps(worker(json.load(sys.stdin)), sort_keys=True))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--blocks", type=int, default=20)
        parser.add_argument("--hardware-blocks", type=int, default=20)
        parser.add_argument("--timeout", type=float, default=30)
        parser.add_argument("--output", default="measurements")
        args = parser.parse_args()
        if args.blocks < 20 or args.hardware_blocks < 20:
            parser.error("The preregistered design requires at least 20 blocks in each experiment")
        main(args)
