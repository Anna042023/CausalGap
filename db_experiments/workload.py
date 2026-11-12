"""Deterministic artificial relational workload, not forecasting run records."""

import csv
import hashlib
import json
from pathlib import Path

SEED = 20261106
MODEL_COUNT = 21
INTERFACES = ("native", "common")
ORIGINS = (1700000000, 1700003600, 1700007200)
HORIZONS = 12
SCALES = {"small": 1, "medium": 8, "large": 33}
ATOM_COLUMNS = ("atom_id", "run_id", "environment", "graph_version", "origin",
                "node_id", "horizon", "prediction", "target", "valid")

SCHEMA = """
CREATE TABLE topology_version (
 graph_version INTEGER PRIMARY KEY,
 parent_version INTEGER
);
CREATE TABLE scenario (
 scenario_id INTEGER PRIMARY KEY,
 baseline_graph INTEGER NOT NULL,
 changed_graph INTEGER NOT NULL,
 intervention_type VARCHAR NOT NULL,
 intensity DOUBLE NOT NULL,
 exposure_role VARCHAR NOT NULL,
 outcome_consistent INTEGER NOT NULL
);
CREATE TABLE evaluation_run (
 run_id INTEGER PRIMARY KEY,
 model_id INTEGER NOT NULL,
 interface VARCHAR NOT NULL,
 scenario_id INTEGER NOT NULL
);
CREATE TABLE forecast_atom (
 atom_id INTEGER PRIMARY KEY,
 run_id INTEGER NOT NULL,
 environment INTEGER NOT NULL CHECK(environment IN (0,1)),
 graph_version INTEGER NOT NULL,
 origin INTEGER NOT NULL,
 node_id INTEGER NOT NULL,
 horizon INTEGER NOT NULL CHECK(horizon BETWEEN 1 AND 12),
 prediction BIGINT NOT NULL,
 target BIGINT NOT NULL,
 valid INTEGER NOT NULL CHECK(valid IN (0,1))
);
"""


def metadata():
    graphs = [(1000, None)] + [(1000 + k, 1000) for k in range(1, 13)]
    scenarios = []
    specs = (("removal", (0.05, 0.10, 0.15, 0.20)),
             ("weight", (0.8, 0.6, 0.4, 0.2)),
             ("isolation", (0.1, 0.2, 0.4, 0.6)))
    for name, levels in specs:
        for i, level in enumerate(levels):
            sid = len(scenarios) + 1
            scenarios.append((sid, 1000, 1000 + sid, name, level,
                              "weak" if i < 3 else "strong", 1))
    runs = []
    for model in range(1, MODEL_COUNT + 1):
        for interface in INTERFACES:
            for scenario in scenarios:
                runs.append((len(runs) + 1, model, interface, scenario[0]))
    return {"topology_version": graphs, "scenario": scenarios,
            "evaluation_run": runs}


def insert_metadata(connection):
    for name, rows in metadata().items():
        placeholders = ",".join("?" for _ in rows[0])
        connection.executemany(f"INSERT INTO {name} VALUES ({placeholders})", rows)


def rows_for_run(run, nodes, first_atom_id=1):
    run_id, model, interface, sid = run
    mode = INTERFACES.index(interface)
    atom_id = first_atom_id
    for oi, origin in enumerate(ORIGINS):
        for node in range(1, nodes + 1):
            for horizon in range(1, HORIZONS + 1):
                target0 = (SEED + model * 11 + sid * 7 + node * 3 + oi * 5
                           + horizon) % 1000
                e0 = (SEED + model + node + horizon + sid + oi + mode * 2) % 7
                ek = (SEED + model * 3 + node * 2 + horizon + sid * 2 + oi
                      + mode) % 11
                targetk = target0 + sid % 4 + 1
                direction = 1 if (node + horizon + oi) % 2 == 0 else -1
                yield (atom_id, run_id, 0, 1000, origin, node, horizon,
                       target0 + direction * e0, target0, 1)
                yield (atom_id + 1, run_id, 1, 1000 + sid, origin, node, horizon,
                       targetk - direction * ek, targetk, 1)
                atom_id += 2


def generate(directory, nodes):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "forecast_atoms.csv"
    expected = []
    atom_id = 1
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(ATOM_COLUMNS)
        for run in metadata()["evaluation_run"]:
            s0 = sk = count = 0
            for row in rows_for_run(run, nodes, atom_id):
                writer.writerow(row)
                if row[2] == 0:
                    s0 += abs(row[7] - row[8])
                    count += 1
                else:
                    sk += abs(row[7] - row[8])
                atom_id = row[0] + 1
            if run[2] == "common":
                expected.append((run[0], count, s0, sk, s0/count, sk/count,
                                 (sk-s0)/(s0+count)))
    (directory / "expected_results.json").write_text(
        json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    info = {"nodes_per_run": nodes, "forecast_atoms": atom_id - 1,
            "selected_interface": "common", "selected_runs": len(expected),
            "expected_pairs": sum(row[1] for row in expected),
            "csv_sha256": sha.hexdigest(), "csv_bytes": path.stat().st_size}
    (directory / "input_manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    return path, expected, info

