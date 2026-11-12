#!/usr/bin/env python3
"""Derive validated pair records and a Python aggregate oracle from generated input.

No new model forecasts are generated here. Forecast atoms are read from the
unchanged earlier generated input; both environments are validated and paired.
The independent oracle aggregates the resulting integers in Python dictionaries.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "db_experiments"))
import workload as original_workload
sys.path.pop(0)
from queries import PAIR_COLUMNS, GRAINS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def deterministic_gzip(source, destination):
    with Path(source).open("rb") as src, Path(destination).open("wb") as dst:
        with gzip.GzipFile(filename="", mode="wb", fileobj=dst, mtime=0) as gz:
            shutil.copyfileobj(src, gz)


def canonical_line(row):
    return json.dumps([int(v) for v in row], separators=(",", ":")).encode() + b"\n"


def prepare(source_atoms, source_manifest, source_expected, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads(Path(source_manifest).read_text())
    expected_coarse = json.loads(Path(source_expected).read_text())
    uncompressed_hash = hashlib.sha256()
    with gzip.open(source_atoms, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            uncompressed_hash.update(block)
    assert uncompressed_hash.hexdigest() == info["csv_sha256"]
    metadata = original_workload.metadata()
    runs = {r[0]: r for r in metadata["evaluation_run"]}
    scenarios = {s[0]: s for s in metadata["scenario"]}
    selected = {k for k, r in runs.items() if r[2] == "common"}
    pending = {}
    completed = set()
    pairs = []
    source_count = selected_atoms = 0
    with gzip.open(source_atoms, "rt", newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == original_workload.ATOM_COLUMNS
        for raw in reader:
            source_count += 1
            rid = int(raw["run_id"])
            if rid not in selected:
                continue
            selected_atoms += 1
            key = tuple(int(raw[k]) for k in PAIR_COLUMNS[:4])
            assert key not in completed, f"Duplicate completed pair: {key}"
            role = int(raw["environment"])
            assert role in (0, 1)
            roles = pending.setdefault(key, {})
            assert role not in roles, f"Duplicate role: {key}, {role}"
            sid = runs[rid][3]
            scenario = scenarios[sid]
            assert scenario[6] == 1
            expected_graph = scenario[1] if role == 0 else scenario[2]
            assert int(raw["graph_version"]) == expected_graph
            roles[role] = (int(raw["valid"]), abs(int(raw["prediction"]) - int(raw["target"])))
            if len(roles) == 2:
                assert roles[0][0] == roles[1][0] == 1
                pairs.append(key + (roles[0][1], roles[1][1]))
                completed.add(key)
                del pending[key]
    assert not pending
    assert source_count == info["forecast_atoms"] == 1197504
    assert selected_atoms == 2 * len(pairs) == 598752
    assert len(pairs) == info["expected_pairs"] == 299376
    assert len({p[0] for p in pairs}) == info["selected_runs"] == 252
    pairs.sort()
    oracle = {}
    with tempfile.TemporaryDirectory(prefix="causalgap_r4_pairs_", dir="/tmp") as temp:
        temp = Path(temp)
        plain = temp / "validated_pairs.csv"
        with plain.open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(PAIR_COLUMNS)
            writer.writerows(pairs)
        pairs_csv_hash = sha(plain)
        pairs_csv_bytes = plain.stat().st_size
        deterministic_gzip(plain, output / "validated_pairs.csv.gz")
        for g in GRAINS:
            width = len(g["keys"])
            groups = {}
            for pair in pairs:
                value = groups.setdefault(pair[:width], [0, 0, 0])
                value[0] += 1
                value[1] += pair[4]
                value[2] += pair[5]
            rows = [key + tuple(groups[key]) for key in sorted(groups)]
            assert len(rows) == g["expected_groups"]
            digest = hashlib.sha256()
            canonical_bytes = 0
            plain_oracle = temp / (g["grain"] + ".csv")
            with plain_oracle.open("w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(tuple(g["keys"]) + ("pairs", "error0", "errork"))
                for row in rows:
                    writer.writerow(row)
                    line = canonical_line(row)
                    digest.update(line)
                    canonical_bytes += len(line)
            deterministic_gzip(plain_oracle, output / ("oracle_" + g["grain"] + ".csv.gz"))
            oracle[g["grain"]] = {
                "keys": list(g["keys"]), "groups": len(rows),
                "pairs": sum(r[-3] for r in rows),
                "error0": sum(r[-2] for r in rows),
                "errork": sum(r[-1] for r in rows),
                "canonical_output_sha256": digest.hexdigest(),
                "canonical_output_bytes": canonical_bytes,
                "oracle_csv_sha256": sha(plain_oracle),
            }
            if g["grain"] == "run":
                assert rows == [tuple(int(v) for v in row[:4]) for row in expected_coarse]
        assert len({(v["pairs"], v["error0"], v["errork"]) for v in oracle.values()}) == 1
    shutil.copy2(source_manifest, output / "r3_input_manifest.json")
    shutil.copy2(source_expected, output / "r3_expected_results.json")
    write_json(output / "expected_manifest.json", oracle)
    result = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_kind": "Unchanged, explicitly generated R3 relational workload; no new forecasts.",
        "source_atoms_file": str(Path(source_atoms).resolve()),
        "source_atoms_gzip_sha256": sha(source_atoms),
        "source_atoms_uncompressed_sha256": uncompressed_hash.hexdigest(),
        "source_atom_count": source_count, "selected_atoms": selected_atoms,
        "selected_interface": "common", "selected_runs": 252, "validated_pairs": len(pairs),
        "node_count": 33, "origins_per_run": 3, "horizons_per_origin_node": 12,
        "pairs_csv_sha256": pairs_csv_hash, "pairs_csv_bytes": pairs_csv_bytes,
        "pairs_gzip_sha256": sha(output / "validated_pairs.csv.gz"),
        "all_pair_roles_unique": True, "all_pair_versions_match": True,
        "all_validity_masks_equal": True, "all_selected_pairs_valid": True,
        "coarse_oracle_matches_r3_exact_integer_outputs": True,
        "derivation_source_sha256": sha(Path(__file__)),
        "original_workload_source_sha256": sha(ROOT.parent / "db_experiments/workload.py"),
        "grains": oracle,
    }
    write_json(output / "pair_input_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parent_input = ROOT.parent / "db_extension/inputs"
    parser.add_argument("--source-atoms", type=Path, default=parent_input / "forecast_atoms.csv.gz")
    parser.add_argument("--source-manifest", type=Path, default=parent_input / "input_manifest.json")
    parser.add_argument("--source-expected", type=Path, default=parent_input / "expected_results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "inputs")
    args = parser.parse_args()
    result = prepare(args.source_atoms, args.source_manifest, args.source_expected, args.output)
    print(json.dumps({k: result[k] for k in ["source_atom_count", "validated_pairs", "pairs_csv_sha256", "grains"]}, indent=2))
