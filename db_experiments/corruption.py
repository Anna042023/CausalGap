#!/usr/bin/env python3
"""Controlled corruption study over artificial records, not forecasting data.

All databases are private in-memory connections. This module never opens or
modifies the performance study's on-disk databases. Each individual case changes
one changed-environment atom. A fifth loaded run remains an unaffected control
in the combined case. The arithmetic oracle is derived from the clean fixture
and the known atom-level mutation, independently of SQL joins.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sqlite3
from collections import Counter
from pathlib import Path

import duckdb

from queries import (TAU, cached_sql, checked_sql, naive_sql, pair_rows_sql,
                     violation_sql)
from workload import ATOM_COLUMNS, HORIZONS, ORIGINS, SCHEMA, SEED, metadata, rows_for_run

HERE = Path(__file__).resolve().parent
FLAGS = ("missing_counterpart", "duplicate_key", "wrong_graph", "support_mismatch")
CASES = ("clean", "missing_counterpart", "duplicate_key", "wrong_graph",
         "support_mismatch", "combined_disjoint")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def serialized_rows(rows):
    return json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()


def make_fixture(nodes):
    runs = [r for r in metadata()["evaluation_run"] if r[2] == "common"][:5]
    rows = []
    next_id = 1
    for run in runs:
        group = list(rows_for_run(run, nodes, first_atom_id=next_id))
        rows.extend(group)
        next_id = max(r[0] for r in group) + 1
    return runs, rows


def clean_oracle(runs, rows):
    results = {}
    for run_id, *_ in runs:
        base = [r for r in rows if r[1] == run_id and r[2] == 0]
        changed = [r for r in rows if r[1] == run_id and r[2] == 1]
        assert len(base) == len(changed)
        assert all(r[9] == 1 for r in base + changed)
        results[run_id] = (
            len(base),
            sum(abs(r[7] - r[8]) for r in base),
            sum(abs(r[7] - r[8]) for r in changed),
        )
    return results


def mutation_target(rows, run_id):
    changed = [r for r in rows if r[1] == run_id and r[2] == 1]
    # Highest changed absolute error; ties select the earliest logical key.
    chosen = min(changed, key=lambda r: (-abs(r[7] - r[8]), r[4], r[5], r[6]))
    baseline = next(r for r in rows if r[1] == run_id and r[2] == 0
                    and r[4:7] == chosen[4:7])
    return chosen, baseline


def corrupt(case, runs, clean_rows):
    rows = list(clean_rows)
    if case == "clean":
        return rows, [], {}
    assignments = (
        list(zip((r[0] for r in runs[:4]), FLAGS))
        if case == "combined_disjoint" else [(runs[0][0], case)]
    )
    changes, expected_flags = [], {}
    next_id = max(r[0] for r in rows) + 1
    for run_id, kind in assignments:
        chosen, baseline = mutation_target(clean_rows, run_id)
        index = next(i for i, row in enumerate(rows) if row[0] == chosen[0])
        after = list(chosen)
        if kind == "missing_counterpart":
            rows.pop(index)
            after = None
        elif kind == "duplicate_key":
            after[0] = next_id
            next_id += 1
            rows.append(tuple(after))
        elif kind == "wrong_graph":
            # Existing, valid graph identifier; incorrect for the changed role.
            after[3] = baseline[3]
            rows[index] = tuple(after)
        elif kind == "support_mismatch":
            after[9] = 0
            rows[index] = tuple(after)
        else:
            raise ValueError(kind)
        expected_flags[run_id] = tuple(int(flag == kind) for flag in FLAGS)
        changes.append({
            "kind": kind, "run_id": run_id,
            "logical_pair_key": {
                "run_id": run_id, "origin": chosen[4], "node_id": chosen[5],
                "horizon": chosen[6]},
            "original_changed_atom": dict(zip(ATOM_COLUMNS, chosen)),
            "resulting_changed_atom": (
                dict(zip(ATOM_COLUMNS, after)) if after is not None else None),
            "paired_baseline_error": abs(baseline[7] - baseline[8]),
            "changed_error": abs(chosen[7] - chosen[8]),
            "physical_atom_delta": (-1 if kind == "missing_counterpart"
                                    else 1 if kind == "duplicate_key" else 0),
        })
    return rows, changes, expected_flags


def naive_oracle(clean, changes):
    expected = dict(clean)
    for change in changes:
        rid = change["run_id"]
        count, s0, sk = expected[rid]
        delta = {"missing_counterpart": -1, "support_mismatch": -1,
                 "duplicate_key": 1, "wrong_graph": 0}[change["kind"]]
        expected[rid] = (
            count + delta,
            s0 + delta * change["paired_baseline_error"],
            sk + delta * change["changed_error"],
        )
    return expected


def metric_values(integer_metrics):
    count, s0, sk = integer_metrics
    return (s0 / count, sk / count, (sk - s0) / (s0 + TAU * count))


def result_map(rows):
    return {int(r[0]): tuple(r[1:]) for r in rows}


def assert_results(actual, expected, label):
    assert set(actual) == set(expected), (label, sorted(actual), sorted(expected))
    for run_id, integers in expected.items():
        got = actual[run_id]
        assert tuple(int(v) for v in got[:3]) == integers, (label, run_id, got, integers)
        for observed, target in zip(got[3:], metric_values(integers)):
            assert math.isclose(float(observed), target, rel_tol=1e-12, abs_tol=1e-12), (
                label, run_id, observed, target)


def connect(engine):
    if engine == "sqlite":
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
    elif engine == "duckdb":
        conn = duckdb.connect(":memory:")
        conn.execute("SET threads=1")
        conn.execute(SCHEMA)
    else:
        raise ValueError(engine)
    return conn


def load_fixture(conn, runs, rows):
    meta = metadata()
    meta["evaluation_run"] = runs
    for table, values in meta.items():
        marks = ",".join("?" for _ in values[0])
        conn.executemany(f"INSERT INTO {table} VALUES ({marks})", values)
    conn.executemany("INSERT INTO forecast_atom VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def add_metrics(row, prefix, values):
    names = (
        "pairs", "baseline_error_numerator", "changed_error_numerator",
        "baseline_error_denominator", "changed_error_denominator",
        "baseline_mae", "changed_mae", "rd_numerator", "rd_denominator", "rd",
    )
    if values is None:
        row.update({f"{prefix}_{name}": None for name in names})
        return
    count, s0, sk = map(int, values[:3])
    mae0, maek, rd = metric_values((count, s0, sk))
    vals = (count, s0, sk, count, count, mae0, maek,
            sk - s0, s0 + TAU * count, rd)
    row.update({f"{prefix}_{name}": value for name, value in zip(names, vals)})


def run_study(nodes):
    runs, clean_rows = make_fixture(nodes)
    clean = clean_oracle(runs, clean_rows)
    versions = {"sqlite": sqlite3.sqlite_version, "duckdb": duckdb.__version__}
    output_rows, execution_cases = [], []
    cross_engine = {}
    for engine in ("sqlite", "duckdb"):
        for case in CASES:
            rows, changes, expected_flags = corrupt(case, runs, clean_rows)
            expected_naive = naive_oracle(clean, changes)
            expected_checked = {rid: m for rid, m in clean.items() if rid not in expected_flags}
            conn = connect(engine)
            try:
                load_fixture(conn, runs, rows)
                actual_flags = {int(row[0]): tuple(map(int, row[1:]))
                                for row in conn.execute(violation_sql()).fetchall()}
                naive = result_map(conn.execute(naive_sql()).fetchall())
                checked = result_map(conn.execute(checked_sql()).fetchall())
                conn.execute("CREATE TABLE validated_pair AS " + pair_rows_sql())
                cached = result_map(conn.execute(cached_sql()).fetchall())
                materialized_pairs = int(conn.execute(
                    "SELECT COUNT(*) FROM validated_pair").fetchone()[0])
                assert actual_flags == expected_flags, (engine, case, actual_flags, expected_flags)
                assert_results(naive, expected_naive, (engine, case, "naive"))
                assert_results(checked, expected_checked, (engine, case, "checked"))
                assert_results(cached, expected_checked, (engine, case, "cached"))
                assert materialized_pairs == sum(m[0] for m in expected_checked.values())
                # The combined case must preserve its sole clean control run.
                assert runs[-1][0] in checked
            finally:
                conn.close()
            for run_id, *_ in runs:
                row = {
                    "artificial_data": True, "engine": engine,
                    "engine_version": versions[engine], "case": case,
                    "nodes_per_run": nodes, "loaded_forecast_atoms": len(rows),
                    "run_id": run_id,
                    "role": "affected" if run_id in expected_flags else "unaffected",
                    "expected_excluded": int(run_id in expected_flags),
                    "checked_excluded": int(run_id not in checked),
                    "naive_admitted": int(run_id in naive),
                    "changed_atoms_in_this_run": sum(c["run_id"] == run_id for c in changes),
                }
                for flag, value in zip(FLAGS, actual_flags.get(run_id, (0, 0, 0, 0))):
                    row[f"detected_{flag}"] = value
                add_metrics(row, "clean", clean[run_id])
                add_metrics(row, "naive", naive[run_id])
                add_metrics(row, "checked", checked.get(run_id))
                row["naive_pair_count_delta"] = row["naive_pairs"] - row["clean_pairs"]
                row["naive_baseline_mae_bias"] = row["naive_baseline_mae"] - row["clean_baseline_mae"]
                row["naive_changed_mae_bias"] = row["naive_changed_mae"] - row["clean_changed_mae"]
                row["naive_rd_bias"] = row["naive_rd"] - row["clean_rd"]
                row["checked_matches_unaffected_clean_oracle"] = (
                    run_id not in expected_flags and run_id in checked)
                output_rows.append(row)
            case_record = {
                "engine": engine, "case": case,
                "loaded_forecast_atoms": len(rows),
                "input_sha256": sha(serialized_rows(rows)),
                "mutations": changes,
                "expected_excluded_run_ids": sorted(expected_flags),
                "observed_excluded_run_ids": sorted(set(clean) - set(checked)),
                "observed_violation_flags": actual_flags,
                "naive_run_ids": sorted(naive),
                "checked_run_ids": sorted(checked),
                "materialized_pairs": materialized_pairs,
                "all_assertions_passed": True,
            }
            execution_cases.append(case_record)
            integer_signature = {
                "flags": actual_flags,
                "naive": {rid: list(map(int, vals[:3])) for rid, vals in naive.items()},
                "checked": {rid: list(map(int, vals[:3])) for rid, vals in checked.items()},
                "materialized_pairs": materialized_pairs,
            }
            if case in cross_engine:
                assert integer_signature == cross_engine[case], ("engine disagreement", case)
            else:
                cross_engine[case] = integer_signature
            print(f"{engine:6s} {case:22s} atoms={len(rows):4d} "
                  f"bad={sorted(expected_flags)} retained={sorted(checked)} PASS", flush=True)

    summary = {
        "study": "Controlled relational corruption over artificial deterministic records",
        "scientific_boundary": (
            "No original forecast records are used or reconstructed. These are new database "
            "correctness measurements. Reported errors are artificial integer fixture values; "
            "they are not traffic forecasting MAE or source-paper RD results."),
        "database_isolation": "Fresh private in-memory connection per engine and case.",
        "versions": versions, "python": platform.python_version(),
        "platform": platform.platform(), "seed": SEED,
        "nodes_per_run": nodes, "origins_per_run": len(ORIGINS),
        "horizons": HORIZONS, "loaded_runs": runs,
        "clean_forecast_atoms": len(clean_rows),
        "clean_pairs": sum(m[0] for m in clean.values()),
        "clean_input_sha256": sha(serialized_rows(clean_rows)),
        "clean_oracle_by_run": {
            rid: {"pairs": m[0], "baseline_error_sum": m[1], "changed_error_sum": m[2],
                  "baseline_mae": metric_values(m)[0], "changed_mae": metric_values(m)[1],
                  "rd": metric_values(m)[2]}
            for rid, m in clean.items()},
        "metric_definition": {
            "tau": TAU, "mae0": "error0_sum / paired_count",
            "maek": "errork_sum / paired_count",
            "rd": "(errork_sum-error0_sum)/(error0_sum+tau*paired_count)",
            "bias_reference": "Same run in the complete clean fixture.",
            "exclusion_note": (
                "Checked queries drop a corrupt run in full; excluded metrics are null, "
                "not zeros. Surviving runs are compared only with their own clean oracle.")},
        "scope_limits": [
            "Faults affect one changed-environment atom per affected run.",
            "The combined case assigns four different defects to four distinct runs.",
            "High changed-error atoms are selected deterministically to expose selective loss/fanout.",
            "Wrong graph changes provenance only; zero numeric bias does not make that run admissible.",
            "This check does not detect a logical key absent from both environments without an expected-support manifest.",
            "A fully absent run also requires manifest-level completeness validation.",
            "No timing or scale conclusion is inferred from this small correctness fixture.",
        ],
        "query_sha256": {name: sha(fn().encode()) for name, fn in (
            ("naive_sql", naive_sql), ("checked_sql", checked_sql),
            ("pair_rows_sql", pair_rows_sql), ("cached_sql", cached_sql),
            ("violation_sql", violation_sql))},
        "source_sha256": {name: sha((HERE / name).read_bytes())
                          for name in ("corruption.py", "queries.py", "workload.py")},
        "case_engine_executions": len(execution_cases),
        "result_rows": len(output_rows), "cross_engine_integer_results_identical": True,
        "exact_expected_run_exclusions_verified": True,
        "unaffected_run_aggregates_verified": True,
        "cached_equals_checked_verified": True,
        "cases": execution_cases,
        "results": output_rows,
    }
    return output_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    if args.nodes < 1:
        parser.error("--nodes must be positive")
    rows, summary = run_study(args.nodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "corruption_results.csv"
    json_path = args.output_dir / "corruption_results.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
