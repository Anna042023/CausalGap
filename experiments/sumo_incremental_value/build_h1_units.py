#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from protocol import (  # noqa: E402
    ALL_MODELS,
    DEFAULT_EXPERIMENT_ROOT,
    H1_LEVELS,
    MODEL_FAMILIES,
    STRUCTURAL_TYPES,
    family_name,
    intensity_lookup,
    json_dump,
    normalized_weak_strength,
    FOUNDATION_MODELS,
    resolve_foundation_parameter_count,
)


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def empirical_calibrated_distance(value: float, reference: np.ndarray) -> float:
    """Appendix A.1 calibration rank with mid-ties and +1 denominator correction."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        raise ValueError("Empty calibration distance reference")
    v = float(value)
    less = int(np.sum(ref < v))
    equal = int(np.sum(np.isclose(ref, v, rtol=1e-10, atol=1e-12)))
    return float(min((1.0 + less + 0.5 * equal) / (len(ref) + 1.0), 1.0))


def normalized_weak_magnitude(value: float, reference: np.ndarray) -> float:
    """Map a weak intervention magnitude monotonically to [0,1] using frozen calibration bounds.

    This is the s_ij coordinate used only for the weak-RD trajectory slope.
    It is intentionally distinct from the empirical-rank distance used in DMIS_cal.
    """
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        raise ValueError("Empty calibration distance reference")
    lo, hi = float(np.min(ref)), float(np.max(ref))
    if hi <= lo + 1e-12:
        return 0.0
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


def load_calibration_refs(root: Path) -> dict[tuple[str, str], np.ndarray]:
    refs = {}
    for setting, folder in [("synthetic", "synthetic"), ("sumo_metrla", "sumo_metrla")]:
        p = root / "environments" / folder / "calibration_distance_reference.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        for kind, g in d.groupby("intervention_type"):
            refs[(setting, str(kind))] = pd.to_numeric(g["environment_distance"], errors="coerce").dropna().to_numpy(float)
    return refs


def collect_raw(root: Path, settings: list[str], models: list[str], eval_seeds: set[int] | None):
    rows = []
    audits = []
    controls = []
    for setting in settings:
        for model in models:
            base = root / "runs" / setting / model / "trackB"
            if not base.exists():
                audits.append(dict(setting=setting, model=model, status="MISSING_RUN_DIR", detail=str(base)))
                continue
            seed_dirs = sorted(p for p in base.glob("seed*") if p.is_dir())
            if not seed_dirs:
                audits.append(dict(setting=setting, model=model, status="MISSING_MODEL_SEED", detail=str(base)))
                continue
            for sd in seed_dirs:
                try:
                    model_seed = int(sd.name.replace("seed", ""))
                except Exception:
                    continue
                cp = sd / "model_controls.csv"
                if cp.exists():
                    c = pd.read_csv(cp)
                    if len(c):
                        c = c.iloc[[0]].copy(); c["setting"] = setting; c["model_seed"] = model_seed
                        controls.append(c)
                p = sd / "causalgap_results.csv"
                if not p.exists():
                    audits.append(dict(setting=setting, model=model, status="MISSING_RESULTS", detail=str(p)))
                    continue
                try:
                    d = pd.read_csv(p)
                except Exception as e:
                    audits.append(dict(setting=setting, model=model, status="UNREADABLE_RESULTS", detail=repr(e)))
                    continue
                d["setting"] = setting
                d["model_seed"] = model_seed
                if eval_seeds is not None:
                    d = d[pd.to_numeric(d["intervention_seed"], errors="coerce").astype("Int64").isin(eval_seeds)]
                rows.append(d)
                audits.append(dict(setting=setting, model=model, status="FOUND", detail=f"rows={len(d)}"))
    raw = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ctr = pd.concat(controls, ignore_index=True) if controls else pd.DataFrame()
    return raw, ctr, pd.DataFrame(audits)


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0
    return float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))


def main():
    ap = argparse.ArgumentParser(description="Build Section 5.2 H1 V2 weak-history -> strong-RD prediction units.")
    ap.add_argument("--source-experiment-root", type=Path,
                    default=None,
                    help="Raw H1 source root. Must be passed explicitly for SUMO-only or combined analysis.")
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT,
                    help="V2 output root. Raw runs are not copied or modified.")
    ap.add_argument("--project-root", type=Path, default=Path("/data/wanganna/CausalGap"))
    ap.add_argument("--foundation-counts-csv", type=Path, default=None,
                    help="Optional audited override CSV with columns model,parameter_count. Used only if desired.")
    ap.add_argument("--settings", nargs="+", choices=["synthetic", "sumo_metrla"], default=["synthetic"])
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--eval-seeds", nargs="*", type=int, default=None,
                    help="Optional filter; default uses all completed intervention seeds.")
    args = ap.parse_args()

    if args.source_experiment_root is None:
        args.source_experiment_root = args.experiment_root

    analysis = args.experiment_root / "analysis" / "h1"
    inter = analysis / "intermediate"
    tables = analysis / "tables"
    diagnostics = analysis / "diagnostics"
    for p in (inter, tables, diagnostics): p.mkdir(parents=True, exist_ok=True)

    eval_seed_set = None if args.eval_seeds is None else set(map(int, args.eval_seeds))
    raw, controls, audit = collect_raw(args.source_experiment_root, args.settings, args.models, eval_seed_set)
    audit.to_csv(diagnostics / "run_coverage.csv", index=False)
    if raw.empty:
        raise RuntimeError("No H1 causalgap_results.csv files found.")

    required = {
        "model", "evaluation_track", "intervention_type", "intervention_seed",
        "paper_intensity_level", "RD", "representation_response_raw", "environment_distance",
        "IPE0_global", "causal_valid", "leakage_detected",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"H1 raw results missing columns: {missing}")

    raw = raw[raw["model"].astype(str).isin(args.models)].copy()
    raw = raw[raw["setting"].isin(args.settings)].copy()
    raw = raw[raw["evaluation_track"].astype(str).str.upper().eq("B")].copy()
    raw = raw[raw["intervention_type"].astype(str).isin(STRUCTURAL_TYPES)].copy()
    raw = raw[bool_series(raw["causal_valid"]) & (~bool_series(raw["leakage_detected"]))].copy()

    refs = load_calibration_refs(args.source_experiment_root)
    if not refs:
        raise RuntimeError("No calibration_distance_reference.csv found. Prepare environments first.")

    phases, strengths, norms = [], [], []
    dcal, dmcal = [], []
    for r in raw.itertuples(index=False):
        spec = intensity_lookup(str(r.intervention_type), str(r.paper_intensity_level))
        phases.append(spec["phase"]); strengths.append(float(spec["strength"]))
        ref = refs.get((str(r.setting), str(r.intervention_type)))
        if ref is None:
            raise RuntimeError(f"Missing calibration reference for {(r.setting, r.intervention_type)}")
        norms.append(
            normalized_weak_magnitude(float(r.environment_distance), ref)
            if spec["phase"] == "weak" else np.nan
        )
        dc = empirical_calibrated_distance(float(r.environment_distance), ref)
        dcal.append(dc)
        dmcal.append(float(r.representation_response_raw) / max(dc, 1e-6))
    raw["h1_phase"] = phases
    raw["declared_strength"] = strengths
    raw["weak_intensity_s"] = norms
    raw["distance_calibrated"] = dcal
    raw["DMIS_cal"] = dmcal
    raw["DMIS_raw"] = pd.to_numeric(raw.get("DMIS", np.nan), errors="coerce")
    raw.to_csv(inter / "01_h1_raw_calibrated_rows.csv", index=False)

    # First average all completed seed replicates within each fixed
    # model-setting-intervention-family-intensity configuration.
    group = ["setting", "model", "intervention_type", "paper_intensity_level", "h1_phase", "declared_strength"]
    numeric = [
        "RD", "DMIS_cal", "DMIS_raw", "representation_response_raw", "environment_distance",
        "distance_calibrated", "weak_intensity_s", "IPE0_global", "MAEk", "IPEk",
    ]
    numeric = [c for c in numeric if c in raw.columns]
    agg = raw.groupby(group, dropna=False)[numeric].mean().reset_index()
    counts = raw.groupby(group, dropna=False).size().rename("n_seed_rows").reset_index()
    agg = agg.merge(counts, on=group, how="left")
    agg.to_csv(inter / "02_configuration_seed_means.csv", index=False)

    if controls.empty:
        raise RuntimeError("No model_controls.csv files found; H1 C controls cannot be built.")
    # V2: repair Foundation parameter-count controls without rerunning any model.
    # Non-foundation controls are retained exactly from the completed raw runs.
    controls.to_csv(inter / "03_model_controls_original.csv", index=False)
    explicit_counts = {}
    if args.foundation_counts_csv is not None:
        odf = pd.read_csv(args.foundation_counts_csv)
        if not {"model", "parameter_count"}.issubset(odf.columns):
            raise RuntimeError("--foundation-counts-csv requires columns model,parameter_count")
        explicit_counts = {str(r.model): int(r.parameter_count) for r in odf.itertuples(index=False)}

    controls_v2 = controls.copy()
    parameter_audit = []
    for model in sorted(set(controls_v2["model"].astype(str))):
        mask = controls_v2["model"].astype(str).eq(model)
        old_vals = pd.to_numeric(controls_v2.loc[mask, "parameter_count"], errors="coerce").dropna()
        old_count = float(old_vals.mean()) if len(old_vals) else np.nan
        if model in FOUNDATION_MODELS:
            new_count, src = resolve_foundation_parameter_count(model, args.project_root, explicit_counts)
            if new_count is None:
                parameter_audit.append(dict(model=model, old_parameter_count=old_count,
                                            corrected_parameter_count=np.nan, status="UNRESOLVED", source=src))
                continue
            controls_v2.loc[mask, "parameter_count"] = int(new_count)
            controls_v2.loc[mask, "log_parameter_count"] = float(np.log1p(int(new_count)))
            if "parameter_count_source" in controls_v2.columns:
                controls_v2.loc[mask, "parameter_count_source"] = src
            parameter_audit.append(dict(model=model, old_parameter_count=old_count,
                                        corrected_parameter_count=int(new_count), status="CORRECTED", source=src))
        else:
            parameter_audit.append(dict(model=model, old_parameter_count=old_count,
                                        corrected_parameter_count=old_count, status="UNCHANGED", source="completed_run_control"))

    pa = pd.DataFrame(parameter_audit)
    pa.to_csv(diagnostics / "foundation_parameter_count_audit.csv", index=False)
    unresolved = pa[(pa["status"] == "UNRESOLVED") & pa["model"].isin(sorted(FOUNDATION_MODELS))]
    if len(unresolved):
        raise RuntimeError(
            "V2 refuses to reuse wrapper-only Foundation parameter counts. Could not resolve: " +
            ", ".join(unresolved["model"].astype(str)) +
            ". Check /data/wanganna/CausalGap/model_weights or pass --foundation-counts-csv with audited counts."
        )

    controls_v2.to_csv(inter / "03_model_controls_merged.csv", index=False)
    cagg = controls_v2.groupby("model", as_index=False).agg({
        "model_family": "first", "model_family_name": "first",
        "parameter_count": "mean", "log_parameter_count": "mean",
        "native_topology_access": "first", "recurrent_flag": "first",
        "convolutional_flag": "first", "attention_flag": "first",
        "pretraining_flag": "first", "adaptation_level": "first",
    })

    units = []
    rejected = []
    for (setting, model, kind), g in agg.groupby(["setting", "model", "intervention_type"], dropna=False):
        weak = g[g["h1_phase"] == "weak"].sort_values("declared_strength")
        strong = g[g["h1_phase"] == "strong"]
        expected_weak = [x["level"] for x in H1_LEVELS[str(kind)] if x["phase"] == "weak"]
        got_weak = list(weak["paper_intensity_level"].astype(str))
        if set(got_weak) != set(expected_weak) or len(weak) != 3 or len(strong) != 1:
            rejected.append(dict(setting=setting, model=model, intervention_type=kind,
                                 reason="requires_exactly_three_weak_and_one_strong",
                                 got_weak="|".join(got_weak), n_strong=len(strong)))
            continue
        x = weak["weak_intensity_s"].to_numpy(float)
        rd = weak["RD"].to_numpy(float)
        dm = weak["DMIS_cal"].to_numpy(float)
        dmr = weak["DMIS_raw"].to_numpy(float)
        srow = strong.iloc[0]
        units.append({
            "setting": setting,
            "model": model,
            "model_family": int(MODEL_FAMILIES[str(model)]),
            "model_family_name": family_name(str(model)),
            "intervention_type": kind,
            "weak_RD_mean": float(np.mean(rd)),
            "weak_RD_slope": slope(x, rd),
            "weak_RD_max": float(np.max(rd)),
            "weak_DMIScal_mean": float(np.mean(dm)),
            "weak_DMIScal_slope": slope(x, dm),
            "weak_DMIScal_max": float(np.max(dm)),
            "weak_DMISraw_mean": float(np.mean(dmr)),
            "baseline_IPE": float(np.mean(weak["IPE0_global"].to_numpy(float))),
            "strong_intervention_magnitude": float(srow["environment_distance"]),
            "strong_intervention_distance_raw": float(srow["environment_distance"]),
            "strong_intervention_distance_calibrated": float(srow["distance_calibrated"]),
            "strong_RD": float(srow["RD"]),
            "strong_DMIScal": float(srow["DMIS_cal"]),
            "strong_IPE": float(srow["IPEk"]),
            "n_seed_rows_min": int(g["n_seed_rows"].min()),
            "weak_levels": "|".join(expected_weak),
            "strong_level": str(srow["paper_intensity_level"]),
        })

    units = pd.DataFrame(units)
    pd.DataFrame(rejected).to_csv(diagnostics / "rejected_prediction_units.csv", index=False)
    if units.empty:
        raise RuntimeError("No admissible H1 prediction units were built.")
    units = units.merge(cagg, on=["model", "model_family", "model_family_name"], how="left")
    if units["log_parameter_count"].isna().any():
        bad = units.loc[units["log_parameter_count"].isna(), "model"].unique().tolist()
        raise RuntimeError(f"Missing log parameter count for models: {bad}")
    units = units.sort_values(["model_family", "model", "setting", "intervention_type"]).reset_index(drop=True)
    units.insert(0, "unit_id", [f"H1_{i:04d}" for i in range(len(units))])
    units.to_csv(inter / "04_h1_prediction_units.csv", index=False)
    units.to_csv(tables / "H1_prediction_units_for_release.csv", index=False)

    coverage = (
        units.groupby(["setting", "model_family_name"], dropna=False)
        .agg(prediction_units=("unit_id", "count"), models=("model", "nunique"))
        .reset_index()
    )
    coverage.to_csv(diagnostics / "prediction_unit_coverage.csv", index=False)

    json_dump(analysis / "h1_unit_build_summary.json", {
        "n_raw_rows": int(len(raw)),
        "n_configuration_means": int(len(agg)),
        "n_prediction_units": int(len(units)),
        "n_models": int(units["model"].nunique()),
        "settings": sorted(units["setting"].unique().tolist()),
        "intervention_types": sorted(units["intervention_type"].unique().tolist()),
        "model_families": sorted(units["model_family"].unique().astype(int).tolist()),
        "primary_response_feature": "weak_DMIScal_mean",
        "error_history": ["weak_RD_mean", "weak_RD_slope", "weak_RD_max"],
        "target": "strong_RD",
        "seed_aggregation": "mean within model-setting-intervention_type-intensity before H1 unit construction",
        "calibration": "DMIScal uses the frozen empirical-rank weak topology-distance reference; weak-RD slope uses [0,1] min-max magnitude bounds from the same disjoint calibration reference; strong outcomes are never used in calibration",
        "v2_control_fix": "H1 control strong_intervention_magnitude now uses raw structural distance d_T. Calibrated rank remains stored separately and is not used as the magnitude control.",
        "source_experiment_root": str(args.source_experiment_root),
    })
    print(f"[DONE] H1 prediction units: {len(units)} -> {inter / '04_h1_prediction_units.csv'}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
