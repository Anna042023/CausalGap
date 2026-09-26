#!/usr/bin/env python3
"""Corrected Section 5.1 analysis: within-family DMISraw, cross-family DMIScal.

This script NEVER trains or evaluates a model.  It reuses the already-completed
Fast-V2 causalgap_results.csv files and the topology-only calibration reference.
All outputs are written under analysis_calibrated, leaving the original analysis
and all expensive model artifacts untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from causalgap.models.registry import SPECS  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def bool_mask(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def parse_model_seed(path: Path, work_root: Path) -> int:
    rel = path.relative_to(work_root)
    for part in rel.parts:
        if part.startswith("seed"):
            try:
                return int(part[4:])
            except ValueError:
                pass
    raise ValueError(f"Cannot parse model seed from {path}")


def load_rows(work_root: Path) -> pd.DataFrame:
    files = sorted(work_root.rglob("causalgap_results.csv"))
    if not files:
        raise SystemExit(f"No causalgap_results.csv found under {work_root}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["model_seed"] = parse_model_seed(f, work_root)
        d["source_csv"] = str(f)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def ensure_intensity_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "paper_intensity_level" in out.columns:
        out["paper_intensity_level"] = out["paper_intensity_level"].astype(str)
        return out
    # Backwards-compatible fallback. Fast V2 normally already stores this field.
    def derive(r) -> str:
        t = str(r["intervention_type"])
        s = float(r["intervention_strength"])
        if t == "edge_removal":
            return f"edge_s{s:.2f}"
        if t == "weight_modification":
            return f"weight_strength{s:.2f}"
        if t in ("node_isolation", "node_masking"):
            return f"node_s{s:.6f}"
        return f"{t}_{s:.6f}"
    out["paper_intensity_level"] = out.apply(derive, axis=1)
    return out


def validate_columns(df: pd.DataFrame) -> None:
    required = {
        "dataset", "model", "evaluation_track", "intervention_type",
        "intervention_family", "intervention_seed", "paper_intensity_level",
        "intervention_strength", "environment_source", "leakage_detected",
        "include_in_information_equivalent_structural", "include_in_main_structural",
        "DMIS", "RD", "representation_response_raw", "environment_distance",
        "MAE0", "MAEk", "IPE0", "IPEk", "model_seed",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")


def add_model_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    class_map = {name: spec.class_id for name, spec in SPECS.items()}
    unknown = sorted(set(out["model"].dropna().astype(str)) - set(class_map))
    if unknown:
        raise SystemExit(f"Unknown models in result CSVs: {unknown}")
    out["model_class"] = out["model"].map(class_map).astype(int)
    return out


def primary_filter(df: pd.DataFrame, track: str) -> tuple[pd.DataFrame, dict]:
    track = track.upper()
    m = df["evaluation_track"].astype(str).str.upper().eq(track)
    m &= df["intervention_family"].astype(str).eq("structural")
    m &= ~bool_mask(df["leakage_detected"])
    if "diagnostic_subset_active" in df.columns:
        m &= bool_mask(df["diagnostic_subset_active"])
    if track == "B":
        m &= bool_mask(df["include_in_information_equivalent_structural"])
        validity = "include_in_information_equivalent_structural"
    else:
        m &= bool_mask(df["include_in_main_structural"])
        validity = "include_in_main_structural"
    d = df[m].copy()
    for col in ["DMIS", "RD", "representation_response_raw", "environment_distance"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    finite = (
        np.isfinite(d["RD"].to_numpy(float))
        & np.isfinite(d["representation_response_raw"].to_numpy(float))
        & np.isfinite(d["environment_distance"].to_numpy(float))
        & (d["environment_distance"].to_numpy(float) > 0)
    )
    dropped = int((~finite).sum())
    d = d.loc[finite].copy()
    report = {
        "track": track,
        "validity_column": validity,
        "raw_rows": int(len(df)),
        "valid_rows": int(len(d)),
        "dropped_nonfinite_or_nonpositive_distance": dropped,
        "environment_sources": d["environment_source"].value_counts(dropna=False).to_dict(),
    }
    return d, report


def raw_dmis_audit(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = primary.copy()
    d["DMIS_raw_recomputed"] = d["representation_response_raw"] / d["environment_distance"]
    d["DMIS_saved"] = pd.to_numeric(d["DMIS"], errors="coerce")
    d["DMIS_raw_abs_diff"] = np.abs(d["DMIS_raw_recomputed"] - d["DMIS_saved"])
    finite = np.isfinite(d["DMIS_raw_abs_diff"].to_numpy(float))
    report = {
        "max_abs_difference": float(d.loc[finite, "DMIS_raw_abs_diff"].max()) if finite.any() else None,
        "median_abs_difference": float(d.loc[finite, "DMIS_raw_abs_diff"].median()) if finite.any() else None,
        "n_compared": int(finite.sum()),
    }
    d["DMIS_raw"] = d["DMIS_raw_recomputed"]
    return d, report


def empirical_magnitude_map(value: float, reference: np.ndarray) -> tuple[float, int, int]:
    """Paper mid-rank CDF mapping with the +1/(N+1) boundary correction."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        raise ValueError("Empty magnitude calibration reference")
    # Distances are produced by deterministic floating operations. Use a tight
    # numerical tolerance so exact node-isolation ties are handled as ties while
    # distinct graph magnitudes are not spuriously merged.
    atol = max(1e-12, 1e-9 * max(1.0, abs(float(value))))
    lt = int(np.count_nonzero(ref < float(value) - atol))
    eq = int(np.count_nonzero(np.isclose(ref, float(value), rtol=1e-9, atol=atol)))
    mapped = (1.0 + lt + 0.5 * eq) / (len(ref) + 1.0)
    return float(min(mapped, 1.0)), lt, eq


def apply_cross_family_calibration(primary: pd.DataFrame, ref: pd.DataFrame, eps_d: float) -> pd.DataFrame:
    d = primary.copy()
    ref = ref.copy()
    ref["environment_distance"] = pd.to_numeric(ref["environment_distance"], errors="coerce")
    ref = ref[np.isfinite(ref["environment_distance"].to_numpy(float))].copy()
    refs: dict[tuple[str, str], np.ndarray] = {}
    for (dataset, itype), g in ref.groupby(["dataset", "intervention_type"], dropna=False):
        refs[(str(dataset), str(itype))] = g["environment_distance"].to_numpy(float)

    mapped, nlt, neq, nref = [], [], [], []
    missing_keys = []
    for r in d.itertuples(index=False):
        key = (str(r.dataset), str(r.intervention_type))
        arr = refs.get(key)
        if arr is None or len(arr) == 0:
            missing_keys.append(key)
            mapped.append(np.nan); nlt.append(0); neq.append(0); nref.append(0)
            continue
        m, a, b = empirical_magnitude_map(float(r.environment_distance), arr)
        mapped.append(m); nlt.append(a); neq.append(b); nref.append(len(arr))
    if missing_keys:
        raise SystemExit(f"Missing calibration references for keys: {sorted(set(missing_keys))}")

    d["environment_distance_calibrated"] = np.asarray(mapped, dtype=float)
    d["calibration_n_less"] = np.asarray(nlt, dtype=int)
    d["calibration_n_equal"] = np.asarray(neq, dtype=int)
    d["calibration_reference_n"] = np.asarray(nref, dtype=int)
    d["DMIS_cal"] = d["representation_response_raw"] / np.maximum(
        d["environment_distance_calibrated"].to_numpy(float), float(eps_d)
    )
    if not np.isfinite(d["DMIS_cal"].to_numpy(float)).all():
        raise SystemExit("Non-finite DMIS_cal after calibration")
    return d


def aggregate_configurations(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g1 = [
        "dataset", "model", "model_class", "model_seed",
        "intervention_type", "paper_intensity_level",
    ]
    per_seed = (
        primary.groupby(g1, dropna=False)
        .agg(
            DMIS_raw=("DMIS_raw", "mean"),
            DMIS_cal=("DMIS_cal", "mean"),
            RD=("RD", "mean"),
            representation_response_raw=("representation_response_raw", "mean"),
            environment_distance=("environment_distance", "mean"),
            environment_distance_calibrated=("environment_distance_calibrated", "mean"),
            intervention_strength=("intervention_strength", "mean"),
            n_intervention_seeds=("intervention_seed", "nunique"),
            n_rows=("intervention_seed", "size"),
            MAE0=("MAE0", "mean"),
            MAEk=("MAEk", "mean"),
        ).reset_index()
    )
    g2 = ["dataset", "model", "model_class", "intervention_type", "paper_intensity_level"]
    config = (
        per_seed.groupby(g2, dropna=False)
        .agg(
            DMIS_raw=("DMIS_raw", "mean"),
            DMIS_cal=("DMIS_cal", "mean"),
            RD=("RD", "mean"),
            representation_response_raw=("representation_response_raw", "mean"),
            environment_distance=("environment_distance", "mean"),
            environment_distance_calibrated=("environment_distance_calibrated", "mean"),
            intervention_strength=("intervention_strength", "mean"),
            n_model_seeds=("model_seed", "nunique"),
            min_intervention_seeds=("n_intervention_seeds", "min"),
            MAE0=("MAE0", "mean"),
            MAEk=("MAEk", "mean"),
        ).reset_index()
    )
    config["cluster_id"] = config["model"].astype(str) + "@@" + config["dataset"].astype(str)
    return per_seed, config


def family_balanced_points(config: pd.DataFrame) -> pd.DataFrame:
    d = (
        config.groupby(["dataset", "model", "model_class", "intervention_type"], dropna=False)
        .agg(
            DMIS_cal=("DMIS_cal", "mean"),
            DMIS_raw=("DMIS_raw", "mean"),
            RD=("RD", "mean"),
            n_intensities=("paper_intensity_level", "nunique"),
        ).reset_index()
    )
    d["cluster_id"] = d["model"].astype(str) + "@@" + d["dataset"].astype(str)
    return d


def corr_values(x: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, float]]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"Pearson r": (np.nan, np.nan), "Spearman rho": (np.nan, np.nan), "Kendall tau": (np.nan, np.nan)}
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    kt = stats.kendalltau(x, y)
    return {
        "Pearson r": (float(pr.statistic), float(pr.pvalue)),
        "Spearman rho": (float(sr.statistic), float(sr.pvalue)),
        "Kendall tau": (float(kt.statistic), float(kt.pvalue)),
    }


def cluster_bootstrap(df: pd.DataFrame, xcol: str, n_boot: int, seed: int) -> tuple[dict, np.ndarray]:
    clusters = np.asarray(sorted(df["cluster_id"].unique()), dtype=object)
    if len(clusters) < 2:
        raise ValueError("Need at least two clusters")
    by = {c: df[df["cluster_id"] == c] for c in clusters}
    rng = np.random.default_rng(seed)
    names = ["Pearson r", "Spearman rho", "Kendall tau"]
    samples = np.full((n_boot, 3), np.nan, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([by[c] for c in picked], ignore_index=True)
        cv = corr_values(boot[xcol].to_numpy(float), boot["RD"].to_numpy(float))
        for j, name in enumerate(names):
            samples[b, j] = cv[name][0]
    cis = {}
    for j, name in enumerate(names):
        vals = samples[:, j]
        vals = vals[np.isfinite(vals)]
        cis[name] = (float(np.quantile(vals, .025)), float(np.quantile(vals, .975)))
    return cis, samples


def make_table(df: pd.DataFrame, xcol: str, n_boot: int, seed: int, analysis_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    point = corr_values(df[xcol].to_numpy(float), df["RD"].to_numpy(float))
    cis, boot = cluster_bootstrap(df, xcol=xcol, n_boot=n_boot, seed=seed)
    rows = []
    for name in ["Pearson r", "Spearman rho", "Kendall tau"]:
        value, p = point[name]; lo, hi = cis[name]
        rows.append({
            "Analysis": analysis_name,
            "DMIS coordinate": xcol,
            "Metric": name,
            "Value": value,
            "95% CI lower": lo,
            "95% CI upper": hi,
            "p-value": p,
            "N configurations": int(len(df)),
            "N model-dataset clusters": int(df["cluster_id"].nunique()),
        })
    return pd.DataFrame(rows), boot


def subgroup_table(config: pd.DataFrame, group_cols: list[str], xcol: str) -> pd.DataFrame:
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for key, g in config.groupby(grouper, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        cv = corr_values(g[xcol].to_numpy(float), g["RD"].to_numpy(float))
        row = {c: v for c, v in zip(group_cols, key)}
        row.update({"DMIS_coordinate": xcol, "N": int(len(g))})
        for name, (v, p) in cv.items():
            prefix = {"Pearson r": "pearson", "Spearman rho": "spearman", "Kendall tau": "kendall"}[name]
            row[f"{prefix}_value"] = v
            row[f"{prefix}_p"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def regression_band(df: pd.DataFrame, xcol: str, xgrid: np.ndarray, n_boot: int, seed: int):
    x = df[xcol].to_numpy(float); y = df["RD"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = intercept + slope * xgrid
    clusters = np.asarray(sorted(df["cluster_id"].unique()), dtype=object)
    by = {c: df[df["cluster_id"] == c] for c in clusters}
    rng = np.random.default_rng(seed)
    curves = np.full((n_boot, len(xgrid)), np.nan)
    for b in range(n_boot):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([by[c] for c in picked], ignore_index=True)
        bx = boot[xcol].to_numpy(float); byy = boot["RD"].to_numpy(float)
        if np.unique(bx).size < 2:
            continue
        s, i = np.polyfit(bx, byy, 1)
        curves[b] = i + s*xgrid
    return yhat, np.nanquantile(curves, .025, axis=0), np.nanquantile(curves, .975, axis=0)


def plot_primary(config: pd.DataFrame, table: pd.DataFrame, out_pdf: Path, out_png: Path, n_boot: int, seed: int) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 12, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    colors = {1:"#4C78A8",2:"#59A14F",3:"#F28E2B",4:"#B07AA1",5:"#E15759"}
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    for cls in sorted(config["model_class"].unique()):
        g = config[config["model_class"] == cls]
        ax.scatter(g["DMIS_cal"], g["RD"], s=24, alpha=.48, c=colors.get(int(cls),"#777777"),
                   edgecolors="none", label=f"Class {int(cls)}", rasterized=True)
    xmin, xmax = float(config["DMIS_cal"].min()), float(config["DMIS_cal"].max())
    if math.isclose(xmin, xmax): xmax = xmin + 1e-6
    xgrid = np.linspace(xmin, xmax, 240)
    yhat, lo, hi = regression_band(config, "DMIS_cal", xgrid, n_boot, seed+17)
    ax.plot(xgrid, yhat, linewidth=2.0, c="#222222", label="Linear Fit")
    ax.fill_between(xgrid, lo, hi, color="#777777", alpha=.18, linewidth=0, label="95% CI")
    vals = {r["Metric"]: float(r["Value"]) for _,r in table.iterrows()}
    txt = f"r = {vals['Pearson r']:.3f}\nρ = {vals['Spearman rho']:.3f}\nτ = {vals['Kendall tau']:.3f}\nN = {len(config):,}"
    ax.text(.98,.97,txt,transform=ax.transAxes,ha="right",va="top",fontsize=12.5,
            bbox=dict(boxstyle="round,pad=0.35",facecolor="white",edgecolor="0.75",alpha=.92))
    ax.set_xlabel("Calibrated DMIS")
    ax.set_ylabel("Relative Degradation (RD)")
    ax.grid(True, linewidth=.6, alpha=.18)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(out_pdf,bbox_inches="tight"); fig.savefig(out_png,dpi=600,bbox_inches="tight"); plt.close(fig)


def plot_raw_vs_cal(config: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],"font.size":13,"pdf.fonttype":42})
    # Two separate figures are deliberately produced (not subplots) elsewhere if needed;
    # this compact diagnostic is a single figure with two distributions only.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    raw = config["DMIS_raw"].to_numpy(float); cal = config["DMIS_cal"].to_numpy(float)
    qraw = np.quantile(raw, [.5,.9,.95,.99]); qcal = np.quantile(cal, [.5,.9,.95,.99])
    x = np.arange(4); w=.36
    ax.bar(x-w/2, qraw, width=w, label="Raw DMIS")
    ax.bar(x+w/2, qcal, width=w, label="Calibrated DMIS")
    ax.set_xticks(x, ["50th","90th","95th","99th"])
    ax.set_ylabel("DMIS Quantile")
    ax.legend(frameon=False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(out_pdf,bbox_inches="tight"); fig.savefig(out_png,dpi=400,bbox_inches="tight"); plt.close(fig)


def save_markdown(table: pd.DataFrame, path: Path) -> None:
    cols = ["Metric","Value","95% CI lower","95% CI upper","p-value","N configurations","N model-dataset clusters"]
    lines = ["| Metric | Value | 95% CI lower | 95% CI upper | p-value | N configurations | N clusters |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for _,r in table.iterrows():
        lines.append(f"| {r['Metric']} | {r['Value']:.4f} | {r['95% CI lower']:.4f} | {r['95% CI upper']:.4f} | {r['p-value']:.4g} | {int(r['N configurations'])} | {int(r['N model-dataset clusters'])} |")
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def save_latex(table: pd.DataFrame, path: Path) -> None:
    lines=[r"\begin{tabular}{lrrrr}",r"\toprule",r"Metric & Value & 95\% CI lower & 95\% CI upper & $p$-value \\",r"\midrule"]
    for _,r in table.iterrows():
        p=float(r["p-value"]); ptxt="$<10^{-4}$" if np.isfinite(p) and p<1e-4 else f"{p:.4f}"
        metric=str(r["Metric"]).replace("rho",r"$\rho$").replace("tau",r"$\tau$")
        lines.append(f"{metric} & {r['Value']:.3f} & {r['95% CI lower']:.3f} & {r['95% CI upper']:.3f} & {ptxt} \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser(description="Corrected Section 5.1 DMIS--RD analysis")
    ap.add_argument("--experiment-root",type=Path,required=True,help="Existing Fast-V2 experiment root containing work_runs/")
    ap.add_argument("--calibration-reference",type=Path,required=True)
    ap.add_argument("--track",choices=["A","B","a","b"],default="B")
    ap.add_argument("--bootstrap",type=int,default=2000)
    ap.add_argument("--bootstrap-seed",type=int,default=2026)
    ap.add_argument("--eps-d",type=float,default=1e-6,help="Numerical guard only; empirical mapping is strictly positive due to +1/(N+1).")
    args=ap.parse_args()

    root=args.experiment_root.expanduser().resolve(); work=root/"work_runs"
    out=root/"analysis_calibrated"
    inter=out/"intermediate"; tables=out/"tables"; figs=out/"figures"; diag=out/"diagnostics"; meta=out/"metadata"
    for p in [inter,tables,figs,diag,meta]: p.mkdir(parents=True,exist_ok=True)

    raw=ensure_intensity_level(add_model_metadata(load_rows(work))); validate_columns(raw)
    raw.to_csv(inter/"01_all_raw_rows_reused.csv",index=False)
    primary,filter_report=primary_filter(raw,args.track)
    primary,audit=raw_dmis_audit(primary)
    ref=pd.read_csv(args.calibration_reference.expanduser().resolve())
    primary=apply_cross_family_calibration(primary,ref,args.eps_d)
    primary.to_csv(inter/"02_valid_rows_with_DMISraw_DMIScal.csv",index=False)

    per_seed,config=aggregate_configurations(primary)
    per_seed.to_csv(inter/"03_per_model_seed_configuration_means.csv",index=False)
    config.to_csv(inter/"04_configuration_means_corrected.csv",index=False)
    family_bal=family_balanced_points(config); family_bal.to_csv(inter/"05_family_balanced_points_corrected.csv",index=False)

    # Correct primary cross-family result: DMIScal.
    table,boot=make_table(config,"DMIS_cal",args.bootstrap,args.bootstrap_seed,"cross-family primary")
    table.to_csv(tables/"Table1_DMIScal_RD_Association.csv",index=False)
    save_markdown(table,tables/"Table1_DMIScal_RD_Association.md")
    save_latex(table,tables/"Table1_DMIScal_RD_Association.tex")
    np.savez_compressed(inter/"Table1_DMIScal_cluster_bootstrap_samples.npz",correlations=boot,metrics=np.array(["Pearson r","Spearman rho","Kendall tau"],dtype=object))

    # Keep the old/raw pooled statistic only as a diagnostic comparison, not as the paper primary.
    raw_table,raw_boot=make_table(config,"DMIS_raw",args.bootstrap,args.bootstrap_seed+1,"raw pooled diagnostic")
    raw_table.to_csv(diag/"pooled_raw_DMIS_correlation_diagnostic.csv",index=False)
    np.savez_compressed(inter/"pooled_raw_DMIS_bootstrap_samples.npz",correlations=raw_boot)

    robust,robust_boot=make_table(family_bal,"DMIS_cal",args.bootstrap,args.bootstrap_seed+2,"family-balanced calibrated robustness")
    robust.to_csv(tables/"Table1_family_balanced_DMIScal_robustness.csv",index=False)
    np.savez_compressed(inter/"family_balanced_DMIScal_bootstrap_samples.npz",correlations=robust_boot)

    # Paper-aligned within-family diagnostics: raw DMIS.
    within=subgroup_table(config,["intervention_type"],"DMIS_raw")
    within.to_csv(diag/"correlation_within_intervention_family_DMISraw.csv",index=False)
    by_dataset_raw=subgroup_table(config,["dataset"],"DMIS_raw")
    by_dataset_raw.to_csv(diag/"correlation_by_dataset_DMISraw.csv",index=False)
    by_dataset_cal=subgroup_table(config,["dataset"],"DMIS_cal")
    by_dataset_cal.to_csv(diag/"correlation_by_dataset_DMIScal.csv",index=False)
    by_dataset_family=subgroup_table(config,["dataset","intervention_type"],"DMIS_raw")
    by_dataset_family.to_csv(diag/"correlation_by_dataset_and_family_DMISraw.csv",index=False)
    by_class=subgroup_table(config,["model_class"],"DMIS_cal")
    by_class.to_csv(diag/"correlation_by_model_class_DMIScal.csv",index=False)

    # Outlier and scale diagnostics; never used to delete or winsorize paper points.
    ranked=config.sort_values("DMIS_raw",ascending=False).copy()
    ranked["raw_GT20"] = ranked["DMIS_raw"] > 20
    ranked["raw_GT50"] = ranked["DMIS_raw"] > 50
    ranked.head(200).to_csv(diag/"top200_raw_DMIS_outliers.csv",index=False)
    thresholds=pd.DataFrame([
        {"threshold":20,"n_raw_above":int((config["DMIS_raw"]>20).sum()),"n_cal_above":int((config["DMIS_cal"]>20).sum())},
        {"threshold":50,"n_raw_above":int((config["DMIS_raw"]>50).sum()),"n_cal_above":int((config["DMIS_cal"]>50).sum())},
    ])
    thresholds.to_csv(diag/"DMIS_outlier_counts.csv",index=False)
    quant=pd.DataFrame({
        "quantile":[0,.01,.05,.25,.5,.75,.95,.99,1],
        "DMIS_raw":np.quantile(config["DMIS_raw"],[0,.01,.05,.25,.5,.75,.95,.99,1]),
        "DMIS_cal":np.quantile(config["DMIS_cal"],[0,.01,.05,.25,.5,.75,.95,.99,1]),
        "RD":np.quantile(config["RD"],[0,.01,.05,.25,.5,.75,.95,.99,1]),
    })
    quant.to_csv(diag/"distribution_quantiles_raw_vs_calibrated.csv",index=False)

    plot_primary(config,table,figs/"Figure3_DMIScal_vs_RD.pdf",figs/"Figure3_DMIScal_vs_RD.png",args.bootstrap,args.bootstrap_seed)
    plot_raw_vs_cal(config,figs/"DMIS_raw_vs_calibrated_quantiles.pdf",figs/"DMIS_raw_vs_calibrated_quantiles.png")

    comparison=table[["Metric","Value","95% CI lower","95% CI upper","p-value"]].copy().rename(columns={"Value":"DMIScal_value","95% CI lower":"DMIScal_CI_low","95% CI upper":"DMIScal_CI_high","p-value":"DMIScal_p"})
    rr=raw_table[["Metric","Value","95% CI lower","95% CI upper","p-value"]].copy().rename(columns={"Value":"DMISraw_value","95% CI lower":"DMISraw_CI_low","95% CI upper":"DMISraw_CI_high","p-value":"DMISraw_p"})
    comparison=comparison.merge(rr,on="Metric",how="left")
    comparison.to_csv(tables/"Table1_raw_vs_calibrated_comparison.csv",index=False)

    manifest={
        "created_at_utc":utc_now(),
        "source_experiment_root":str(root),
        "calibration_reference":str(args.calibration_reference.expanduser().resolve()),
        "track":args.track.upper(),
        "primary_coordinate":"DMIS_cal",
        "within_family_coordinate":"DMIS_raw",
        "cross_family_mapping":"mid-rank empirical CDF within each dataset x intervention family, using weak calibration magnitudes",
        "eps_d":float(args.eps_d),
        "eps_d_note":"Numerical guard does not bind because the +1/(N_ref+1) mapping is strictly positive.",
        "raw_dmis_audit":audit,
        "filter_report":filter_report,
        "n_raw_rows":int(len(raw)),
        "n_primary_rows":int(len(primary)),
        "n_configuration_rows":int(len(config)),
        "n_model_dataset_clusters":int(config["cluster_id"].nunique()),
        "model_seeds":sorted(primary["model_seed"].astype(int).unique().tolist()),
        "intervention_seeds":sorted(primary["intervention_seed"].astype(int).unique().tolist()),
        "bootstrap_replicates":int(args.bootstrap),
        "ci_method":"percentile cluster bootstrap over model-dataset clusters",
        "scope_warning":(
            "Rows with environment_source=csg_input on observational datasets are topology-input stress tests. "
            "This analysis must not be described as intervention-consistent causal outcome evidence."
        ),
        "no_gpu_rerun":True,
    }
    dump_json(meta/"corrected_analysis_manifest.json",manifest)

    print("\n"+"="*94)
    print("Section 5.1 corrected cross-family analysis complete")
    print("="*94)
    print(table.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("\nWithin-family raw-DMIS correlations:")
    print(within.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print(f"\nCorrected Table 1 : {tables/'Table1_DMIScal_RD_Association.csv'}")
    print(f"Corrected Figure 3: {figs/'Figure3_DMIScal_vs_RD.pdf'}")
    print(f"Diagnostics        : {diag}")
    print("No model was retrained or re-evaluated.")


if __name__=="__main__":
    main()
