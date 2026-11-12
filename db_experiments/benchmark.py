#!/usr/bin/env python3
"""Run the NEW controlled relational benchmark, never forecasting experiments.

Requires duckdb==2.0.0. No original forecasting observations or run records are
used. Load/setup costs are single measurements. Query latency uses one warm-up
and at least five measured, deterministically interleaved repetitions.
"""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
import uuid

import duckdb

from queries import naive_sql, checked_sql, pair_rows_sql, cached_sql
from workload import SCHEMA, SEED, SCALES, metadata, insert_metadata, generate

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def append_csv(path, row):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def scalar(connection, sql):
    return connection.execute(sql).fetchone()[0]


def checksum(rows):
    canonical = [[int(x) for x in row[:4]] +
                 [format(float(x), ".12f") for x in row[4:]] for row in rows]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def verify_input(path, expected_sha256):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise AssertionError("Input CSV hash changed before loading")


def verify(rows, expected):
    if len(rows) != len(expected):
        raise AssertionError(f"Run count {len(rows)} != expected {len(expected)}")
    for got, want in zip(rows, expected):
        if tuple(map(int, got[:4])) != tuple(map(int, want[:4])):
            raise AssertionError(f"Count/error sums differ: {got[:4]} != {want[:4]}")
        for value, reference in zip(got[4:], want[4:]):
            if not math.isclose(float(value), float(reference), rel_tol=1e-12,
                                abs_tol=1e-12):
                raise AssertionError(f"Metric differs: {value} != {reference}")
    actual = checksum(rows)
    if actual != checksum(expected):
        raise AssertionError("Canonical output checksum differs")
    return actual, sum(int(row[1]) for row in rows)


def timed_query(connection, engine, sql, timeout, persist=False):
    deadline = time.monotonic() + timeout
    timer = None
    if engine == "SQLite":
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline),
                                        10000)
    else:
        timer = threading.Timer(timeout, connection.interrupt)
        timer.daemon = True
        timer.start()
    start = time.perf_counter_ns()
    try:
        cursor = connection.execute(sql)
        rows = cursor.fetchall() if cursor.description else []
        if persist:
            if engine == "SQLite":
                connection.commit()
            else:
                connection.execute("CHECKPOINT")
        elapsed = (time.perf_counter_ns() - start) / 1e6
    finally:
        if engine == "SQLite":
            connection.set_progress_handler(None, 0)
        elif timer is not None:
            timer.cancel()
            timer.join()
    return rows, elapsed


def connect_duck(path):
    connection = duckdb.connect(str(path))
    connection.execute("SET threads=1")
    connection.execute("SET memory_limit='1GB'")
    connection.execute("SET enable_progress_bar=false")
    return connection


def load_sqlite(path, csv_path, timeout):
    connection = sqlite3.connect(str(path))
    deadline = time.monotonic() + timeout
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    start = time.perf_counter_ns()
    connection.executescript(SCHEMA)
    insert_metadata(connection)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        while chunk := list(itertools.islice(reader, 10000)):
            connection.executemany("INSERT INTO forecast_atom VALUES (?,?,?,?,?,?,?,?,?,?)",
                                   [tuple(map(int, row)) for row in chunk])
    connection.commit()
    load_ms = (time.perf_counter_ns() - start) / 1e6
    connection.set_progress_handler(None, 0)
    _, analyze_ms = timed_query(connection, "SQLite", "ANALYZE", timeout, True)
    settings = {name: scalar(connection, f"PRAGMA {name}") for name in
                ("automatic_index", "cache_size", "temp_store", "journal_mode",
                 "synchronous", "page_size", "threads")}
    connection.close()
    return load_ms, analyze_ms, settings


def load_duck(path, csv_path, timeout):
    connection = connect_duck(path)
    start = time.perf_counter_ns()
    connection.execute(SCHEMA)
    insert_metadata(connection)
    escaped_path = str(csv_path).replace("'", "''")
    timed_query(connection, "DuckDB",
                f"COPY forecast_atom FROM '{escaped_path}' (HEADER TRUE, DELIMITER ',')",
                timeout, True)
    load_ms = (time.perf_counter_ns() - start) / 1e6
    _, analyze_ms = timed_query(connection, "DuckDB", "ANALYZE", timeout, True)
    return connection, load_ms, analyze_ms


def save_plans(connection, engine, scale, variant, queries):
    directory = ROOT / "plans" / scale / variant
    directory.mkdir(parents=True, exist_ok=True)
    for name, sql in queries.items():
        rows = connection.execute(("EXPLAIN QUERY PLAN " if engine == "SQLite"
                                   else "EXPLAIN ") + sql).fetchall()
        (directory / f"{name}.txt").write_text(
            "\n".join(" | ".join(map(str, row)) for row in rows) + "\n",
            encoding="utf-8")
        (directory / f"{name}.sql").write_text(sql + ";\n", encoding="utf-8")


def run_variant(connection, db_path, engine, variant, scale, info, expected,
                load_ms, analyze_ms, index_ms, repeats, timeout):
    actual_atoms = int(scalar(connection, "SELECT COUNT(*) FROM forecast_atom"))
    if actual_atoms != info["forecast_atoms"]:
        raise AssertionError(f"Incomplete load: {actual_atoms} atoms")
    base_bytes = db_path.stat().st_size
    _, materialize_ms = timed_query(
        connection, engine, "CREATE TABLE validated_pair AS " + pair_rows_sql(),
        timeout, True)
    cache_bytes = db_path.stat().st_size
    queries = {"naive": naive_sql(), "checked": checked_sql(), "cached": cached_sql()}
    save_plans(connection, engine, scale, variant, queries)
    setup = {"scale": scale, "forecast_atoms": info["forecast_atoms"],
             "engine": engine, "variant": variant, "load_ms": load_ms,
             "analyze_ms": analyze_ms, "index_ms": index_ms,
             "materialize_ms": materialize_ms,
             "db_bytes_before_materialization": base_bytes,
             "db_bytes_after_materialization": cache_bytes,
             "materialized_pairs": info["expected_pairs"]}
    append_csv(ROOT / "setup_costs.csv", setup)
    rng = random.Random(SEED + len(scale) + len(variant))
    for repeat in range(repeats + 1):
        names = list(queries)
        rng.shuffle(names)
        for name in names:
            record = {"scale": scale, "forecast_atoms": info["forecast_atoms"],
                      "engine": engine, "variant": variant, "query": name,
                      "repeat": repeat, "warmup": int(repeat == 0),
                      "latency_ms": "", "status": "", "output_runs": "",
                      "scored_pairs": "", "output_sha256": ""}
            try:
                rows, elapsed = timed_query(connection, engine, queries[name], timeout)
                digest, pairs = verify(rows, expected)
                record.update(latency_ms=elapsed, status="ok", output_runs=len(rows),
                              scored_pairs=pairs, output_sha256=digest)
            except Exception as error:
                record["status"] = type(error).__name__ + ": " + str(error)
                append_csv(ROOT / "latency_repeats.csv", record)
                raise
            append_csv(ROOT / "latency_repeats.csv", record)
    final_atoms = int(scalar(connection, "SELECT COUNT(*) FROM forecast_atom"))
    if final_atoms != actual_atoms:
        raise AssertionError("Input row count changed during timing")
    print(f"Completed {scale}: {variant}, {info['forecast_atoms']:,} atoms, "
          f"{info['expected_pairs']:,} checked pairs, checksum {checksum(expected)[:12]}",
          flush=True)


def summarize():
    with (ROOT / "latency_repeats.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        if row["warmup"] == "0" and row["status"] == "ok":
            key = (row["scale"], row["forecast_atoms"], row["engine"],
                   row["variant"], row["query"])
            grouped.setdefault(key, []).append(float(row["latency_ms"]))
    summaries = []
    for key, timings in grouped.items():
        q25, _, q75 = statistics.quantiles(timings, n=4, method="inclusive")
        row = dict(zip(("scale", "forecast_atoms", "engine", "variant", "query"), key))
        row.update(repeats=len(timings), median_ms=statistics.median(timings),
                   q25_ms=q25, q75_ms=q75, iqr_ms=q75-q25,
                   min_ms=min(timings), max_ms=max(timings))
        append_csv(ROOT / "latency_summary.csv", row)
        summaries.append(row)
    index = {(r["scale"], r["variant"], r["query"]): r for r in summaries}
    with (ROOT / "setup_costs.csv").open() as handle:
        setups = list(csv.DictReader(handle))
    for setup in setups:
        key = (setup["scale"], setup["variant"])
        checked = float(index[(*key, "checked")]["median_ms"])
        cached = float(index[(*key, "cached")]["median_ms"])
        saving = checked - cached
        build = float(setup["materialize_ms"])
        append_csv(ROOT / "materialization_break_even.csv",
                   {"scale": key[0], "variant": key[1], "build_ms": build,
                    "checked_median_ms": checked, "cached_median_ms": cached,
                    "saving_per_query_ms": saving,
                    "derived_q_star": build/saving if saving > 0 else "undefined",
                    "first_integer_queries_cheaper": math.floor(build/saving)+1
                    if saving > 0 else "undefined"})


def hardware():
    details = {"python": sys.version, "sqlite": sqlite3.sqlite_version,
               "duckdb": duckdb.__version__, "platform": platform.platform(),
               "logical_cpu_count": os.cpu_count(),
               "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "duckdb_threads": 1, "duckdb_memory_limit": "1GB",
               "sqlite_auxiliary_threads": 0}
    for name, source in (("cpuinfo", "/proc/cpuinfo"), ("meminfo", "/proc/meminfo"),
                         ("cgroup_cpu_max", "/sys/fs/cgroup/cpu.max"),
                         ("cgroup_memory_max", "/sys/fs/cgroup/memory.max")):
        try:
            content = Path(source).read_text()
            if name == "cpuinfo":
                content = "\n".join(line for line in content.splitlines()
                                    if line.startswith(("model name", "vendor_id",
                                                        "cpu MHz", "processor")))
            details[name] = content
        except OSError as error:
            details[name] = str(error)
    return details


def compress_input(path):
    target = path.with_suffix(".csv.gz")
    with path.open("rb") as source, target.open("wb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0,
                           compresslevel=6) as compressed:
            shutil.copyfileobj(source, compressed)
    path.unlink()


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--scales", nargs="+", choices=SCALES, default=list(SCALES))
    parser.add_argument("--output-dir", default="measurements",
                        help="Fresh output directory relative to this script")
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("At least five measured repeats are required")
    destination = SOURCE_ROOT / args.output_dir
    if destination.exists():
        parser.error(f"Refusing to overwrite existing output directory: {destination}")
    # Keep ALL mutable state outside the shared workspace while it is live.
    # Only finalized, closed files are copied into the deliverable directory.
    ROOT = Path(tempfile.mkdtemp(prefix="cg_db_"))
    for name in ("latency_repeats.csv", "setup_costs.csv", "latency_summary.csv",
                 "materialization_break_even.csv"):
        if (ROOT / name).exists():
            parser.error(f"Refusing to overwrite existing results: {name}")
    runtime = ROOT / ("runtime_" + uuid.uuid4().hex)
    runtime.mkdir()
    original_affinity = sorted(os.sched_getaffinity(0))
    try:
        os.sched_setaffinity(0, {original_affinity[-1]})
        affinity_status = "Pinned benchmark process to highest available CPU"
    except OSError as error:
        affinity_status = "CPU pinning unavailable: " + str(error)
    host = hardware()
    host.update(original_cpu_affinity=original_affinity,
                affinity_status=affinity_status,
                environment="Shared container, other services may be scheduled")
    write_json(ROOT / "hardware_versions.json", host)
    write_json(ROOT / "metadata.json", metadata())
    write_json(ROOT / "configuration.json",
               {"seed": SEED, "scales": {s: SCALES[s] for s in args.scales},
                "repeats_after_warmup": args.repeats, "warmups_per_query": 1,
                "query_timeout_seconds": args.timeout,
                "load_timeout_seconds": 120, "interface": "common", "tau": 1.0,
                "data": "New artificial relational records, not forecasting runs",
                "timing": "execute plus fetchall; warm OS cache, no cache flush",
                "repeat_order": "fixed-seed shuffle of queries in each repeat",
                "iqr": "inclusive empirical quartiles, not confidence intervals",
                "sqlite_load": "CSV parsed in Python, executemany batches of 10000",
                "duckdb_load": "native SQL COPY of the same CSV",
                "index": "UNIQUE(run_id,environment,origin,node_id,horizon)",
                "cache": "validated pairs for selected interface, static snapshot",
                "setup_costs": "one fresh measurement, including persistent commit",
                "sqlite_indexed_load": "same loaded base database copied before indexing",
                "runtime_storage": "private /tmp directory; immutable outputs copied after close"})
    for scale in args.scales:
        csv_path, expected, info = generate(ROOT / "inputs" / scale, SCALES[scale])
        base = runtime / f"{scale}_sqlite_auto.db"
        indexed = runtime / f"{scale}_sqlite_indexed.db"
        duck_path = runtime / f"{scale}_duckdb.db"
        verify_input(csv_path, info["csv_sha256"])
        load_ms, analyze_ms, sqlite_settings = load_sqlite(base, csv_path, 120)
        write_json(ROOT / "sqlite_settings.json", sqlite_settings)
        shutil.copy2(base, indexed)
        connection = sqlite3.connect(str(base))
        run_variant(connection, base, "SQLite", "sqlite_auto", scale, info, expected,
                    load_ms, analyze_ms, 0.0, args.repeats, args.timeout)
        connection.close()
        base.unlink()
        connection = sqlite3.connect(str(indexed))
        _, index_ms = timed_query(connection, "SQLite", "CREATE UNIQUE INDEX "
                                  "forecast_pair_key ON forecast_atom "
                                  "(run_id,environment,origin,node_id,horizon)",
                                  args.timeout, True)
        _, index_analyze_ms = timed_query(connection, "SQLite", "ANALYZE", args.timeout, True)
        run_variant(connection, indexed, "SQLite", "sqlite_indexed", scale, info,
                    expected, load_ms, analyze_ms + index_analyze_ms, index_ms,
                    args.repeats, args.timeout)
        connection.close()
        indexed.unlink()
        verify_input(csv_path, info["csv_sha256"])
        connection, duck_load_ms, duck_analyze_ms = load_duck(duck_path, csv_path, 120)
        run_variant(connection, duck_path, "DuckDB", "duckdb_scan", scale, info,
                    expected, duck_load_ms, duck_analyze_ms, 0.0,
                    args.repeats, args.timeout)
        connection.close()
        duck_path.unlink()
        compress_input(csv_path)
    summarize()
    runtime.rmdir()
    shutil.copytree(ROOT, destination)
    shutil.rmtree(ROOT)
    print("All query repetitions match independently generated expected sums, "
          "pair counts, MAE/RD, and canonical output hashes.", flush=True)


if __name__ == "__main__":
    main()

