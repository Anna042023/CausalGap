#!/usr/bin/env python3
"""Build the deterministic relational workload used in the SQL benchmarks.

The generated data are not fitted-model predictions. Run this script from any
working directory. It writes only to the selected output directory.
"""
import argparse
import gzip
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "db_experiments"))
from workload import SCALES, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=SCALES, default="large")
    parser.add_argument("--output", type=Path, default=ROOT / "db_extension" / "inputs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path, expected, manifest = generate(args.output, SCALES[args.scale])
    packed = args.output / "forecast_atoms.csv.gz"
    with csv_path.open("rb") as source, packed.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as zipped:
            shutil.copyfileobj(source, zipped)
    csv_path.unlink()
    print(f"Generated {manifest['forecast_atoms']:,} records in {packed}")
    print(f"Validated comparison pairs: {manifest['expected_pairs']:,}")


if __name__ == "__main__":
    main()
