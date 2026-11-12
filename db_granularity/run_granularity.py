#!/usr/bin/env python3
"""R4 aggregation-granularity experiment with sequential fresh query workers.

Each worker opens a frozen database read-only, runs one same-query warmup,
then one measured query. Timers cover execute() plus complete fetchall().
All live databases and mutable measurements stay in a private /tmp folder.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
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
import duckdb
from queries import GRAINS, GRAIN_MAP, PAIR_COLUMNS, SCHEMA, query_sql

ROOT = Path(__file__).resolve().parent
ORDER_SEED = 2026110705
CONFIGS = (
    {"config": "sqlite_default", "engine": "SQLite", "threads": 0, "memory_mib": None},
    {"config": "duckdb_1t", "engine": "DuckDB", "threads": 1, "memory_mib": 1024},
    {"config": "duckdb_4t", "engine": "DuckDB", "threads": 4, "memory_mib": 1024},
)


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_line(row):
    return json.dumps([int(v) for v in row], separators=(",", ":")).encode() + b"\n"


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path, records):
    fields = sorted(set().union(*(set(r) for r in records))) if records else []
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def read_memory():
    result = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                result[line.split(":")[0]] = int(line.split()[1])
    except OSError:
        pass
    return result


def host_snapshot():
    result = {
        "utc": utcnow(), "python": sys.version, "python_executable": sys.executable,
        "sqlite": sqlite3.sqlite_version, "duckdb": duckdb.__version__,
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)), "load_average": list(os.getloadavg()),
        "tmp_disk_usage": list(shutil.disk_usage("/tmp")),
    }
    for key, path in [
        ("cpuinfo", "/proc/cpuinfo"), ("meminfo", "/proc/meminfo"),
        ("cgroup_cpu_max", "/sys/fs/cgroup/cpu.max"),
        ("cgroup_memory_max", "/sys/fs/cgroup/memory.max"),
        ("cgroup_memory_current", "/sys/fs/cgroup/memory.current"),
        ("cgroup_cpu_stat", "/sys/fs/cgroup/cpu.stat"),
    ]:
        try:
            result[key] = Path(path).read_text()
        except OSError as exc:
            result[key] = str(exc)
    return result


def open_database(config, db_path, temp_directory):
    if config["engine"] == "SQLite":
        os.environ["SQLITE_TMPDIR"] = str(temp_directory)
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        settings = {k: connection.execute("PRAGMA " + k).fetchone()[0]
                    for k in ["threads", "cache_size", "page_size", "temp_store", "automatic_index",
                              "mmap_size", "query_only"]}
    else:
        connection = duckdb.connect(str(db_path), read_only=True)
        connection.execute(f"SET threads={int(config['threads'])}")
        connection.execute(f"SET memory_limit='{int(config['memory_mib'])}MiB'")
        connection.execute("SET enable_progress_bar=false")
        escaped_temp = str(temp_directory).replace("'", "''")
        connection.execute(f"SET temp_directory='{escaped_temp}'")
        names = ["threads", "memory_limit", "temp_directory", "preserve_insertion_order"]
        settings = dict(zip(names, connection.execute("SELECT " + ", ".join(
            "current_setting('" + k + "')" for k in names)).fetchone()))
    return connection, settings


def verify(rows, expected):
    h = hashlib.sha256()
    encoded_bytes = 0
    total_pairs = error0 = errork = 0
    previous = None
    for row in rows:
        values = tuple(int(v) for v in row)
        key = values[:-3]
        if previous is not None:
            assert previous < key, "Result keys are not strictly ordered and unique"
        previous = key
        line = canonical_line(values)
        h.update(line)
        encoded_bytes += len(line)
        total_pairs += values[-3]
        error0 += values[-2]
        errork += values[-1]
    actual = {
        "output_rows": len(rows), "scored_pairs": total_pairs,
        "error0_sum": error0, "errork_sum": errork,
        "canonical_output_bytes": encoded_bytes, "output_sha256": h.hexdigest(),
    }
    assert len(rows) == expected["groups"], "Group count mismatch"
    assert total_pairs == expected["pairs"], "Pair count mismatch"
    assert error0 == expected["error0"] and errork == expected["errork"], "Integer sum mismatch"
    assert encoded_bytes == expected["canonical_output_bytes"], "Canonical size mismatch"
    assert h.hexdigest() == expected["canonical_output_sha256"], "Full output checksum mismatch"
    return actual


def measure(connection, config, grain, expected, timeout):
    gc.collect()
    reset_status = "ok"
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5\n")
    except OSError as exc:
        reset_status = f"{type(exc).__name__}: {exc}"
    before = read_memory()
    deadline = time.monotonic() + timeout
    timer = None
    if config["engine"] == "SQLite":
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    else:
        timer = threading.Timer(timeout, connection.interrupt)
        timer.daemon = True
        timer.start()
    rows = []
    status, error = "ok", ""
    exec_wall_ms = exec_cpu_ms = fetch_wall_ms = fetch_cpu_ms = None
    started_utc = utcnow()
    c0 = time.process_time_ns()
    t0 = time.perf_counter_ns()
    try:
        cursor = connection.execute(query_sql(grain))
        t1 = time.perf_counter_ns()
        c1 = time.process_time_ns()
        exec_wall_ms, exec_cpu_ms = (t1 - t0) / 1e6, (c1 - c0) / 1e6
        rows = cursor.fetchall()
        t2 = time.perf_counter_ns()
        c2 = time.process_time_ns()
        fetch_wall_ms, fetch_cpu_ms = (t2 - t1) / 1e6, (c2 - c1) / 1e6
    except Exception as exc:
        status = "query_failure"
        error = f"{type(exc).__name__}: {exc}"
        t2, c2 = time.perf_counter_ns(), time.process_time_ns()
    finally:
        wall_ms, cpu_ms = (t2 - t0) / 1e6, (c2 - c0) / 1e6
        after = read_memory()
        lifetime_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if config["engine"] == "SQLite":
            connection.set_progress_handler(None, 0)
        elif timer is not None:
            timer.cancel()
            timer.join()
    result = {
        "started_utc": started_utc, "status": status, "error": error,
        "wall_ms": wall_ms, "cpu_ms": cpu_ms,
        "execute_wall_ms": exec_wall_ms, "execute_cpu_ms": exec_cpu_ms,
        "fetch_wall_ms": fetch_wall_ms, "fetch_cpu_ms": fetch_cpu_ms,
        "output_rows": None, "scored_pairs": None, "output_sha256": None,
        "oracle_status": "not_run", "rss_start_kib": before.get("VmRSS"),
        "peak_rss_kib": after.get("VmHWM", lifetime_peak),
        "process_lifetime_peak_rss_kib": lifetime_peak,
        "hwm_reset_status": reset_status,
        "rss_scope": "query_window_including_warm_resident_state_and_fetched_result"
        if reset_status == "ok" else "process_lifetime_including_startup_and_warmup",
    }
    if status == "ok":
        try:
            result.update(verify(rows, expected))
            result["oracle_status"] = "pass"
        except Exception as exc:
            result.update(status="oracle_failure", oracle_status="fail", error=f"{type(exc).__name__}: {exc}")
    del rows
    return result


def worker(payload):
    os.sched_setaffinity(0, set(payload["affinity"]))
    temporary = Path(payload["temp_directory"])
    temporary.mkdir(parents=True, exist_ok=True)
    config = payload["config"]
    connection = None
    result = {"worker_pid": os.getpid(), "records": [], "connection_closed": False}
    try:
        connection, settings = open_database(config, payload["database"], temporary)
        expected = json.loads(Path(payload["expected_manifest"]).read_text())[payload["grain"]]
        for phase in ["warmup", "measured"]:
            record = measure(connection, config, payload["grain"], expected, payload["timeout"])
            record.update(phase=phase, config=config["config"], engine=config["engine"],
                          configured_threads=config["threads"], memory_mib=config["memory_mib"],
                          grain=payload["grain"], key_count=len(GRAIN_MAP[payload["grain"]]["keys"]),
                          expected_output_rows=expected["groups"], block=payload["block"],
                          order_in_block=payload["order_in_block"], worker_pid=os.getpid(),
                          affinity=sorted(os.sched_getaffinity(0)), actual_settings=settings)
            result["records"].append(record)
    except Exception as exc:
        result["worker_error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if connection is not None:
            connection.close()
            result["connection_closed"] = True
        shutil.rmtree(temporary, ignore_errors=True)
    print(json.dumps(result, sort_keys=True), flush=True)


def setup_databases(work, inputs, measurements):
    info = json.loads((inputs / "pair_input_manifest.json").read_text())
    csv_path = work / "validated_pairs.csv"
    with gzip.open(inputs / "validated_pairs.csv.gz", "rb") as src, csv_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    assert sha(csv_path) == info["pairs_csv_sha256"]
    db_paths = {"SQLite": work / "pairs.sqlite", "DuckDB": work / "pairs.duckdb"}
    setup = []
    for engine in ["SQLite", "DuckDB"]:
        t0, c0 = time.perf_counter_ns(), time.process_time_ns()
        if engine == "SQLite":
            connection = sqlite3.connect(db_paths[engine])
            connection.execute(SCHEMA)
            with csv_path.open(newline="") as f:
                reader = csv.reader(f)
                assert tuple(next(reader)) == PAIR_COLUMNS
                batch = []
                for row in reader:
                    batch.append(tuple(int(v) for v in row))
                    if len(batch) == 20000:
                        connection.executemany("INSERT INTO validated_pair VALUES (?,?,?,?,?,?)", batch)
                        batch.clear()
                if batch:
                    connection.executemany("INSERT INTO validated_pair VALUES (?,?,?,?,?,?)", batch)
            connection.commit()
        else:
            connection = duckdb.connect(str(db_paths[engine]))
            connection.execute("SET threads=1")
            connection.execute("SET memory_limit='1024MiB'")
            connection.execute(SCHEMA)
            escaped_csv = str(csv_path).replace("'", "''")
            connection.execute(f"COPY validated_pair FROM '{escaped_csv}' (FORMAT CSV, HEADER true)")
            connection.execute("CHECKPOINT")
        count = connection.execute("SELECT COUNT(*) FROM validated_pair").fetchone()[0]
        assert count == info["validated_pairs"] == 299376
        connection.close()
        setup.append({"engine": engine, "rows": count, "wall_ms": (time.perf_counter_ns()-t0)/1e6,
                      "cpu_ms": (time.process_time_ns()-c0)/1e6, "database_bytes": db_paths[engine].stat().st_size,
                      "database_sha256": sha(db_paths[engine])})
    plans = measurements / "plans"
    for config in CONFIGS:
        config_dir = plans / config["config"]
        config_dir.mkdir(parents=True)
        temp_dir = work / ("plan_temp_" + config["config"])
        temp_dir.mkdir()
        conn, settings = open_database(config, db_paths[config["engine"]], temp_dir)
        write_json(config_dir / "settings.json", settings)
        for g in GRAINS:
            sql = query_sql(g["grain"])
            (config_dir / (g["grain"] + ".sql")).write_text(sql + ";\n")
            prefix = "EXPLAIN QUERY PLAN " if config["engine"] == "SQLite" else "EXPLAIN "
            result = conn.execute(prefix + sql).fetchall()
            write_json(config_dir / (g["grain"] + ".json"), result)
            (config_dir / (g["grain"] + ".txt")).write_text("\n".join(" | ".join(str(v) for v in row) for row in result) + "\n")
        conn.close()
        shutil.rmtree(temp_dir)
    write_json(measurements / "setup.json", setup)
    csv_path.unlink()
    gc.collect()
    return db_paths


def controller(args):
    destination = args.output.resolve()
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing measurements: {destination}")
    inputs = args.inputs.resolve()
    info = json.loads((inputs / "pair_input_manifest.json").read_text())
    assert sha(inputs / "validated_pairs.csv.gz") == info["pairs_gzip_sha256"]
    available = sorted(os.sched_getaffinity(0))
    assert len(available) >= 5, "Four worker CPUs plus one controller CPU are required"
    worker_affinity, controller_cpu = available[-4:], available[0]
    work = Path(tempfile.mkdtemp(prefix="causalgap_r4_grain_", dir="/tmp"))
    measurements = work / "measurements"
    measurements.mkdir()
    all_records = []
    all_workers = []
    error = None
    completed = 0
    timing_started = timing_ended = None
    try:
        write_json(measurements / "host_before.json", host_snapshot())
        os.sched_setaffinity(0, {controller_cpu})
        protocol = {
            "blocks": args.blocks, "order_seed": args.seed, "configurations": CONFIGS,
            "grains": GRAINS, "worker_affinity": worker_affinity, "controller_cpu": controller_cpu,
            "input_pairs": 299376, "timeout_seconds_per_query": args.timeout,
            "fresh_worker_per_configuration_grain_trial": True, "same_query_warmup_per_worker": 1,
            "timed_scope": "execute() plus full fetchall(); API segments are recorded, not assumed server/client cost partitions",
            "excluded_costs": ["input derivation", "database loading", "process startup", "connection setup", "warmup", "oracle comparison"],
            "rss_scope": "query-window VmHWM reset immediately before each query; includes warm resident state and fetched results",
            "query_changes": ["number of grouping keys", "group cardinality", "sort keys", "result cardinality and tuple width"],
            "scope": "One shared host, unchanged artificial pair input, warm OS cache; no natural-intervention or forecasting accuracy evidence",
        }
        write_json(measurements / "protocol.json", protocol)
        write_json(measurements / "source_hashes.json", {p: sha(ROOT / p) for p in [
            "run_granularity.py", "queries.py", "prepare_inputs.py", "../db_experiments/workload.py"]})
        write_json(measurements / "input_hashes.json", {str(p.relative_to(inputs)): sha(p) for p in sorted(inputs.iterdir()) if p.is_file()})
        db_paths = setup_databases(work, inputs, measurements)
        combinations = [(c, g["grain"]) for c in CONFIGS for g in GRAINS]
        rng = random.Random(args.seed)
        schedule = []
        for block in range(1, args.blocks + 1):
            order = combinations.copy()
            rng.shuffle(order)
            schedule.extend({"block": block, "order_in_block": i, "config": c["config"], "grain": g}
                            for i, (c, g) in enumerate(order))
        write_json(measurements / "randomization.json", schedule)
        configs = {c["config"]: c for c in CONFIGS}
        timing_started = utcnow()
        print("TIMING_START " + timing_started + " trials=" + str(len(schedule)), flush=True)
        for task in schedule:
            config = configs[task["config"]]
            payload = {**task, "config": config, "database": str(db_paths[config["engine"]]),
                       "expected_manifest": str(inputs / "expected_manifest.json"),
                       "temp_directory": str(work / ("worker_temp_" + str(completed))),
                       "affinity": worker_affinity, "timeout": args.timeout}
            try:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                    input=json.dumps(payload), text=True, capture_output=True,
                    timeout=args.timeout * 2 + 45, cwd=work)
                child = json.loads(process.stdout) if process.returncode == 0 else {
                    "worker_error": f"exit {process.returncode}", "stderr": process.stderr, "stdout": process.stdout,
                    "connection_closed": False, "records": []}
            except subprocess.TimeoutExpired as exc:
                child = {"worker_error": "worker_timeout", "stderr": str(exc.stderr),
                         "connection_closed": False, "records": []}
            except Exception as exc:
                child = {"worker_error": f"{type(exc).__name__}: {exc}", "connection_closed": False, "records": []}
            child_meta = {**task, "worker_pid": child.get("worker_pid"), "connection_closed": child.get("connection_closed"),
                          "worker_error": child.get("worker_error"), "stderr": child.get("stderr", ""),
                          "traceback": child.get("traceback", "")}
            all_workers.append(child_meta)
            append_jsonl(measurements / "workers.jsonl", child_meta)
            phases = {r["phase"] for r in child.get("records", [])}
            records = list(child.get("records", []))
            for phase in ["warmup", "measured"]:
                if phase not in phases:
                    records.append({**task, "engine": config["engine"], "configured_threads": config["threads"],
                                    "phase": phase, "status": "worker_failure", "oracle_status": "not_run",
                                    "error": child.get("worker_error", "worker did not return expected phase"),
                                    "expected_output_rows": GRAIN_MAP[task["grain"]]["expected_groups"]})
            for record in records:
                all_records.append(record)
                append_jsonl(measurements / "raw.jsonl", record)
            completed += 1
            if task["order_in_block"] == len(combinations) - 1:
                failures = sum(r["status"] != "ok" for r in all_records)
                print(f"BLOCK {task['block']:02d}/{args.blocks} completed; trials={completed}; failures={failures}", flush=True)
                snapshot = host_snapshot()
                append_jsonl(measurements / "host_blocks.jsonl", {"block": task["block"], **{
                    k: snapshot[k] for k in ["utc", "load_average", "cgroup_cpu_stat", "cgroup_memory_current"]}})
        timing_ended = utcnow()
        print("TIMING_END " + timing_ended, flush=True)
    except Exception as exc:
        error = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        print("FATAL " + json.dumps(error), flush=True)
    finally:
        if timing_started and not timing_ended:
            timing_ended = utcnow()
        write_csv(measurements / "raw.csv", all_records)
        write_json(measurements / "host_after.json", host_snapshot())
        write_json(measurements / "completion.json", {
            "timing_started_utc": timing_started, "timing_ended_utc": timing_ended,
            "completed_worker_trials": completed, "expected_worker_trials": args.blocks * 12,
            "measured_queries": sum(r["phase"] == "measured" for r in all_records),
            "warmup_queries": sum(r["phase"] == "warmup" for r in all_records),
            "oracle_passes": sum(r.get("oracle_status") == "pass" for r in all_records),
            "failures": sum(r["status"] != "ok" for r in all_records),
            "all_worker_connections_closed": all(w.get("connection_closed") for w in all_workers),
            "fatal_error": error,
        })
        os.sched_setaffinity(0, set(available))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(measurements, destination)
        shutil.rmtree(work)
        print("SAVED " + str(destination), flush=True)
    if error:
        raise RuntimeError(error["error"])


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker(json.loads(sys.stdin.read()))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--inputs", type=Path, default=ROOT / "inputs")
        parser.add_argument("--output", type=Path, default=ROOT / "measurements")
        parser.add_argument("--blocks", type=int, default=20)
        parser.add_argument("--seed", type=int, default=ORDER_SEED)
        parser.add_argument("--timeout", type=float, default=30.0)
        args = parser.parse_args()
        assert args.blocks >= 1 and args.timeout > 0
        controller(args)
