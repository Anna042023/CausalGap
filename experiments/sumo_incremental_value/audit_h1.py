#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def safe_z(x: float, mu: float, sd: float) -> float:
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.nan
    return float((x - mu) / sd)


def main():
    ap = argparse.ArgumentParser(description="CPU-only audit for Section 5.2 H1 V2 controls and DMIS extrapolation.")
    ap.add_argument("--experiment-root", type=Path,
                    default=Path("/data/wanganna/CausalGap/experiment_results/sumo_incremental_value"))
    args = ap.parse_args()

    root = args.experiment_root
    analysis = root / "analysis" / "h1"
    inter = analysis / "intermediate"
    diag = analysis / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    units_path = inter / "04_h1_prediction_units.csv"
    if not units_path.exists():
        raise FileNotFoundError(units_path)
    u = pd.read_csv(units_path)

    # 1) Strong-control audit: raw structural distance must be used as H1 magnitude control.
    c = u[["unit_id", "model", "setting", "intervention_type",
           "strong_intervention_magnitude", "strong_intervention_distance_raw"]].copy()
    c["abs_difference"] = (c["strong_intervention_magnitude"] - c["strong_intervention_distance_raw"]).abs()
    c["control_matches_raw_dT"] = c["abs_difference"] <= 1e-12
    c.to_csv(diag / "v2_strong_magnitude_control_audit.csv", index=False)
    if not bool(c["control_matches_raw_dT"].all()):
        raise RuntimeError("V2 control audit failed: strong_intervention_magnitude is not raw d_T for all units")

    # 2) Model-level DMIS distribution audit.
    model_summary = (
        u.groupby(["model_family", "model_family_name", "model"], as_index=False)
        .agg(
            n_units=("unit_id", "count"),
            weak_DMIScal_mean=("weak_DMIScal_mean", "mean"),
            weak_DMIScal_min=("weak_DMIScal_mean", "min"),
            weak_DMIScal_max=("weak_DMIScal_mean", "max"),
            weak_DMISraw_mean=("weak_DMISraw_mean", "mean"),
            strong_RD_mean=("strong_RD", "mean"),
            strong_RD_min=("strong_RD", "min"),
            strong_RD_max=("strong_RD", "max"),
        )
    )
    model_summary["all_weak_DMIScal_exact_zero"] = model_summary["weak_DMIScal_max"].abs() <= 1e-15
    model_summary.to_csv(diag / "v2_model_dmis_summary.csv", index=False)

    # 3) Leave-one-family-out extrapolation audit. This is descriptive only and never changes H1 units.
    rows = []
    for fam in sorted(u["model_family"].unique()):
        train = u[u["model_family"] != fam]
        test = u[u["model_family"] == fam]
        mu = float(train["weak_DMIScal_mean"].mean())
        sd = float(train["weak_DMIScal_mean"].std(ddof=0))
        train_min = float(train["weak_DMIScal_mean"].min())
        train_max = float(train["weak_DMIScal_mean"].max())
        for model, g in test.groupby("model"):
            v = float(g["weak_DMIScal_mean"].mean())
            rows.append({
                "held_out_family": int(fam),
                "model": model,
                "model_mean_weak_DMIScal": v,
                "training_mean": mu,
                "training_std": sd,
                "z_vs_training": safe_z(v, mu, sd),
                "training_min": train_min,
                "training_max": train_max,
                "outside_training_range": bool(v < train_min or v > train_max),
            })
    extr = pd.DataFrame(rows)
    extr.to_csv(diag / "v2_family_holdout_dmis_extrapolation.csv", index=False)

    # 4) Strong-RD sign summary by intervention family.
    sign = (
        u.groupby(["setting", "intervention_type"], as_index=False)
        .agg(
            n_units=("unit_id", "count"),
            positive_strong_RD=("strong_RD", lambda x: int(np.sum(np.asarray(x, float) > 0))),
            nonpositive_strong_RD=("strong_RD", lambda x: int(np.sum(np.asarray(x, float) <= 0))),
            mean_strong_RD=("strong_RD", "mean"),
            median_strong_RD=("strong_RD", "median"),
        )
    )
    sign.to_csv(diag / "v2_strong_rd_sign_summary.csv", index=False)

    # 5) Focus models requested by the Synthetic-only diagnostic.
    focus_models = [m for m in ["Autoformer", "FEDformer", "Moirai"] if m in set(u["model"].astype(str))]
    focus = u[u["model"].isin(focus_models)].copy()
    if len(focus):
        keep = [c for c in [
            "unit_id", "model", "model_family", "setting", "intervention_type",
            "weak_RD_mean", "weak_RD_slope", "weak_RD_max",
            "weak_DMIScal_mean", "weak_DMISraw_mean", "strong_RD",
            "parameter_count", "log_parameter_count",
        ] if c in focus.columns]
        focus[keep].to_csv(diag / "v2_focus_models_Autoformer_FEDformer_Moirai.csv", index=False)

    # 6) If OOF files are already available, add model-wise effect audit.
    p0 = diag / "oof_primary_error_history.csv"
    p1 = diag / "oof_primary_error_history_plus_dmis.csv"
    if p0.exists() and p1.exists():
        a = pd.read_csv(p0)[["unit_id", "model", "model_family", "absolute_error"]].rename(
            columns={"absolute_error": "ae_error_history"}
        )
        b = pd.read_csv(p1)[["unit_id", "absolute_error"]].rename(
            columns={"absolute_error": "ae_plus_dmis"}
        )
        m = a.merge(b, on="unit_id", how="inner")
        m["risk_reduction"] = m["ae_error_history"] - m["ae_plus_dmis"]
        mm = m.groupby(["model_family", "model"], as_index=False).agg(
            error_history_MAE=("ae_error_history", "mean"),
            plus_DMIS_MAE=("ae_plus_dmis", "mean"),
            risk_reduction=("risk_reduction", "mean"),
            positive_units=("risk_reduction", lambda x: int(np.sum(np.asarray(x, float) > 0))),
            n_units=("risk_reduction", "size"),
        )
        mm.to_csv(diag / "v2_model_level_oof_effects.csv", index=False)

    print("===== H1 V2 audit complete =====")
    print(f"Units: {len(u)}")
    print("Strong control uses raw dT for all units: PASS")
    print("Foundation parameter count audit:", diag / "foundation_parameter_count_audit.csv")
    print("Model DMIS summary:", diag / "v2_model_dmis_summary.csv")
    print("Family extrapolation audit:", diag / "v2_family_holdout_dmis_extrapolation.csv")
    print("Strong-RD sign summary:", diag / "v2_strong_rd_sign_summary.csv")


if __name__ == "__main__":
    main()
