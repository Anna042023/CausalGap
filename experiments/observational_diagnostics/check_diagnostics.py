#!/usr/bin/env python3
"""Audit Section 5.1 fast-v2 job coverage without rerunning anything."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_LEVELS = {
    "edge_removal": {"s=0.05", "s=0.10", "s=0.20"},
    "weight_modification": {"gamma=0.8", "gamma=0.5", "gamma=0.2"},
    "node_isolation": {"s=0.10", "s=0.20", "s=0.40", "s=0.60"},
}

def bool_mask(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.experiment_root.expanduser().resolve()
    manifest_path = root / "experiment_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_per_job = int(manifest["expected_rows_per_job"])
    expected_seeds = set(int(x) for x in manifest["intervention_seeds"])
    rows = []
    for dataset in manifest["datasets"]:
        for model in manifest["models"]:
            for model_seed in manifest["model_seeds"]:
                rd = root / "work_runs" / dataset / model / f"track{manifest['track']}" / f"seed{model_seed}"
                f = rd / "causalgap_results.csv"
                rec = {"dataset":dataset,"model":model,"track":manifest["track"],"model_seed":model_seed,
                       "result_csv":str(f),"status":"MISSING","rows":0,"finite_dmis_rd":False,
                       "seed_coverage_ok":False,"level_coverage_ok":False,"validity_ok":False,
                       "diagnostic_subset_ok":False,"diagnostic_windows":np.nan}
                if f.exists():
                    try:
                        df = pd.read_csv(f)
                        rec["rows"] = len(df)
                        required = {"DMIS","RD","intervention_seed","intervention_type","paper_intensity_level",
                                    "include_in_information_equivalent_structural","include_in_main_structural",
                                    "diagnostic_subset_active","diagnostic_selected_test_windows"}
                        missing = required.difference(df.columns)
                        if missing:
                            raise ValueError(f"missing columns {sorted(missing)}")
                        rec["finite_dmis_rd"] = bool(
                            np.isfinite(pd.to_numeric(df["DMIS"], errors="coerce")).all() and
                            np.isfinite(pd.to_numeric(df["RD"], errors="coerce")).all()
                        )
                        rec["seed_coverage_ok"] = set(df["intervention_seed"].astype(int).tolist()) == expected_seeds
                        level_ok = True
                        for kind, levels in EXPECTED_LEVELS.items():
                            g = df[df["intervention_type"].astype(str).eq(kind)]
                            # With multiple seeds each level repeats; compare unique level labels.
                            if set(g["paper_intensity_level"].astype(str).tolist()) != levels:
                                level_ok = False
                        rec["level_coverage_ok"] = level_ok
                        if manifest["track"] == "B":
                            rec["validity_ok"] = bool(bool_mask(df["include_in_information_equivalent_structural"]).all())
                        else:
                            rec["validity_ok"] = bool(bool_mask(df["include_in_main_structural"]).all())
                        diag = bool_mask(df["diagnostic_subset_active"])
                        win = pd.to_numeric(df["diagnostic_selected_test_windows"], errors="coerce")
                        rec["diagnostic_subset_ok"] = bool(diag.all() and win.notna().all() and win.nunique() == 1)
                        if win.notna().any(): rec["diagnostic_windows"] = int(win.dropna().iloc[0])
                        ok = (len(df)==expected_per_job and rec["finite_dmis_rd"] and rec["seed_coverage_ok"] and
                              rec["level_coverage_ok"] and rec["validity_ok"] and rec["diagnostic_subset_ok"])
                        rec["status"] = "PASS" if ok else "INCOMPLETE"
                    except Exception as e:
                        rec["status"]="ERROR"; rec["error"]=repr(e)
                rows.append(rec)
    out = pd.DataFrame(rows)
    audit_dir = root / "analysis" / "diagnostics"; audit_dir.mkdir(parents=True, exist_ok=True)
    out_path = audit_dir / "job_coverage.csv"; out.to_csv(out_path, index=False)
    counts = out["status"].value_counts().to_dict()
    print(out.groupby("status").size().to_string())
    print(f"\nExpected jobs: {len(out)}")
    print(f"PASS         : {counts.get('PASS',0)}")
    print(f"Missing/etc  : {len(out)-counts.get('PASS',0)}")
    print(f"Audit saved  : {out_path}")
    by_ds = out[out["status"].eq("PASS")].groupby("dataset")["diagnostic_windows"].min()
    if not by_ds.empty:
        print("\nDiagnostic windows by dataset:\n" + by_ds.to_string())
    bad = out[out["status"] != "PASS"]
    if not bad.empty:
        print("\nFirst incomplete jobs:")
        print(bad[["dataset","model","model_seed","status","rows"]].head(30).to_string(index=False))
        raise SystemExit(2)

if __name__ == "__main__": main()
