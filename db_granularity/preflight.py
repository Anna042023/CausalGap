#!/usr/bin/env python3
"""Execute untimed correctness checks before freezing the measurement run."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import sys

sys.dont_write_bytecode = True
from queries import GRAINS, query_sql
from run_granularity import CONFIGS, ROOT, open_database, setup_databases, sha, verify, write_json, utcnow


def main():
    expected = json.loads((ROOT / "inputs/expected_manifest.json").read_text())
    work = Path(tempfile.mkdtemp(prefix="causalgap_r4_preflight_", dir="/tmp"))
    records = []
    try:
        artifacts = work / "artifacts"
        artifacts.mkdir()
        db_paths = setup_databases(work, ROOT / "inputs", artifacts)
        for config in CONFIGS:
            temporary = work / ("query_" + config["config"])
            temporary.mkdir()
            connection, settings = open_database(config, db_paths[config["engine"]], temporary)
            try:
                for grain in GRAINS:
                    rows = connection.execute(query_sql(grain["grain"])).fetchall()
                    result = verify(rows, expected[grain["grain"]])
                    records.append({"config": config["config"], "grain": grain["grain"],
                                    "status": "pass", **result})
                    del rows
            finally:
                connection.close()
        result = {"completed_utc": utcnow(), "purpose": "Untimed functional check; not included in timing statistics",
                  "queries": records, "query_count": len(records), "all_passed": len(records) == 12,
                  "source_hashes": {f: sha(ROOT / f) for f in [
                      "prepare_inputs.py", "queries.py", "run_granularity.py", "preflight.py"]}}
        write_json(ROOT / "preflight.json", result)
        print(json.dumps({"query_count": len(records), "all_passed": len(records) == 12}))
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
