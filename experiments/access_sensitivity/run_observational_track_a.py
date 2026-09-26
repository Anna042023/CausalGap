#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the seven observational topology-input stress tests on Track A.

This shim reuses the existing Section 5.1 scheduler but makes two Section-5.8-specific
changes for Track A only:
  1) passes --force-include-adjacency-only so graph-free models are retained as the
     intended zero-response implementation sanity check;
  2) uses a Track-A completion check rather than the Track-B information-equivalence gate.
No core CausalGap file is modified.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = CODE_ROOT / "experiments" / "observational_diagnostics" / "run_diagnostics.py"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def load_base():
    if not BASE_PATH.exists():
        raise FileNotFoundError(BASE_PATH)
    spec = importlib.util.spec_from_file_location("section51_base_runner", BASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


BASE = load_base()
ORIG_BUILD = BASE.build_intervene_cmd
ORIG_COMPLETE = BASE.intervention_result_complete


def build_intervene_cmd_tracka(job, args, intervention_seeds):
    cmd = ORIG_BUILD(job, args, intervention_seeds)
    if str(job.track).upper() == "A" and "--force-include-adjacency-only" not in cmd:
        cmd.append("--force-include-adjacency-only")
    return cmd


def complete_tracka(path: Path, job, intervention_seeds):
    if str(job.track).upper() != "A":
        return ORIG_COMPLETE(path, job, intervention_seeds)
    csv_path = path / "causalgap_results.csv"
    if not csv_path.exists():
        return False, "missing causalgap_results.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"unreadable csv: {e}"
    counts = BASE.expected_counts(intervention_seeds)
    expected_total = sum(counts.values())
    if len(df) != expected_total:
        return False, f"row count {len(df)} != expected {expected_total}"
    req = {"dataset","model","evaluation_track","intervention_type","intervention_seed","DMIS","RD","environment_distance","representation_response_raw","paper_intensity_level","leakage_detected"}
    missing=sorted(req-set(df.columns))
    if missing:
        return False, f"missing columns: {missing}"
    if set(df["evaluation_track"].astype(str).str.upper()) != {"A"}:
        return False, "track mismatch"
    if set(pd.to_numeric(df["intervention_seed"],errors="coerce").dropna().astype(int)) != set(intervention_seeds):
        return False, "intervention seed coverage mismatch"
    keys=df[["intervention_type","paper_intensity_level","intervention_seed"]].astype(str)
    if keys.duplicated().any():
        return False, "duplicate intervention keys"
    for c in ["DMIS","RD","environment_distance","representation_response_raw"]:
        if not np.isfinite(pd.to_numeric(df[c],errors="coerce")).all():
            return False, f"non-finite {c}"
    if BASE._bool_series(df["leakage_detected"]).any():
        return False, "leakage_detected=True present"
    return True, "complete Track-A input-stress diagnostic"


BASE.build_intervene_cmd = build_intervene_cmd_tracka
BASE.intervention_result_complete = complete_tracka

if __name__ == "__main__":
    BASE.main()
