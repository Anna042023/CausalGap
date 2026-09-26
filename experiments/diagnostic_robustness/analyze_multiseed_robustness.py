#!/usr/bin/env python3
"""Targeted multi-intervention-seed robustness analysis for Section 5.1.

Design:
  * Representative models: one per model class (default set chosen in run script).
  * Representative datasets: METR-LA and PEMS-BAY.
  * Model seed fixed at 0 (no retraining).
  * Intervention seed 0 is reused from the completed primary run.
  * Intervention seeds 1..4 are read from the dedicated robustness run.
  * Within-family diagnostics use DMIS_raw.
  * Cross-family diagnostics use the fixed dataset×family DMIS_cal calibration.

This script does not run a model or use a GPU.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
            return int(part[4:])
    raise ValueError(f"Cannot parse model seed from {path}")


def load_rows(work_root: Path, source_name: str) -> pd.DataFrame:
    files = sorted(work_root.rglob("causalgap_results.csv"))
    if not files:
        raise SystemExit(f"No causalgap_results.csv found under {work_root}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["model_seed"] = parse_model_seed(f, work_root)
        d["source_csv"] = str(f)
        d["source_experiment"] = source_name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def ensure_intensity_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "paper_intensity_level" in out.columns:
        out["paper_intensity_level"] = out["paper_intensity_level"].astype(str)
        return out
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


def empirical_magnitude_map(value: float, reference: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        raise ValueError("Empty magnitude calibration reference")
    atol = max(1e-12, 1e-9 * max(1.0, abs(float(value))))
    lt = int(np.count_nonzero(ref < float(value) - atol))
    eq = int(np.count_nonzero(np.isclose(ref, float(value), rtol=1e-9, atol=atol)))
    return float(min((1.0 + lt + 0.5 * eq) / (len(ref) + 1.0), 1.0))


def apply_calibration(df: pd.DataFrame, ref: pd.DataFrame, eps_d: float) -> pd.DataFrame:
    refs = {
        (str(ds), str(it)): g["environment_distance"].astype(float).to_numpy()
        for (ds, it), g in ref.groupby(["dataset", "intervention_type"], dropna=False)
    }
    out = df.copy()
    mapped = []
    for r in out.itertuples(index=False):
        key = (str(r.dataset), str(r.intervention_type))
        if key not in refs:
            raise SystemExit(f"Missing calibration reference for {key}")
        mapped.append(empirical_magnitude_map(float(r.environment_distance), refs[key]))
    out["environment_distance_calibrated"] = np.asarray(mapped, dtype=float)
    out["DMIS_cal"] = out["representation_response_raw"].astype(float) / np.maximum(out["environment_distance_calibrated"], eps_d)
    out["DMIS_raw"] = out["representation_response_raw"].astype(float) / out["environment_distance"].astype(float)
    return out


def corr_values(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"Pearson r": (np.nan, np.nan), "Spearman rho": (np.nan, np.nan), "Kendall tau": (np.nan, np.nan)}
    p = stats.pearsonr(x, y); s = stats.spearmanr(x, y); k = stats.kendalltau(x, y)
    return {
        "Pearson r": (float(p.statistic), float(p.pvalue)),
        "Spearman rho": (float(s.statistic), float(s.pvalue)),
        "Kendall tau": (float(k.statistic), float(k.pvalue)),
    }


def cluster_bootstrap(df: pd.DataFrame, xcol: str, n_boot: int, seed: int):
    clusters = np.asarray(sorted(df["cluster_id"].astype(str).unique()), dtype=object)
    by = {c: df[df["cluster_id"].astype(str) == c] for c in clusters}
    rng = np.random.default_rng(seed)
    names = ["Pearson r", "Spearman rho", "Kendall tau"]
    samples = np.full((n_boot, len(names)), np.nan)
    for b in range(n_boot):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([by[c] for c in picked], ignore_index=True)
        cv = corr_values(boot[xcol], boot["RD"])
        for j, n in enumerate(names):
            samples[b, j] = cv[n][0]
    cis = {}
    for j, n in enumerate(names):
        v = samples[:, j]
        v = v[np.isfinite(v)]
        cis[n] = (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (np.nan, np.nan)
    return cis, samples


def make_corr_table(df: pd.DataFrame, xcol: str, analysis: str, n_boot: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    point = corr_values(df[xcol], df["RD"])
    cis, boot = cluster_bootstrap(df, xcol, n_boot, seed)
    rows = []
    for name in ["Pearson r", "Spearman rho", "Kendall tau"]:
        v, p = point[name]; lo, hi = cis[name]
        rows.append({
            "Analysis": analysis,
            "Metric": name,
            "Value": v,
            "95% CI lower": lo,
            "95% CI upper": hi,
            "p-value": p,
            "N configurations": int(len(df)),
            "N model-dataset clusters": int(df["cluster_id"].nunique()),
        })
    return pd.DataFrame(rows), boot


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze targeted multi-intervention-seed robustness for Section 5.1")
    ap.add_argument("--primary-root", type=Path, required=True)
    ap.add_argument("--robustness-root", type=Path, required=True)
    ap.add_argument("--calibration-reference", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=2051)
    ap.add_argument("--eps-d", type=float, default=1e-6)
    args = ap.parse_args()

    primary_root = args.primary_root.expanduser().resolve()
    rob_root = args.robustness_root.expanduser().resolve()
    ref_path = args.calibration_reference.expanduser().resolve()
    out = rob_root / "analysis"
    inter = out / "intermediate"; tables = out / "tables"; figs = out / "figures"; meta = out / "metadata"
    for p in [inter, tables, figs, meta]: p.mkdir(parents=True, exist_ok=True)

    p0 = load_rows(primary_root / "work_runs", "primary_seed0")
    pr = load_rows(rob_root / "work_runs", "robustness_extra_seeds")
    all_rows = pd.concat([p0, pr], ignore_index=True)
    all_rows = ensure_intensity_level(all_rows)

    models = [str(x) for x in args.models]
    datasets = [str(x) for x in args.datasets]
    seeds = sorted(set(int(x) for x in args.seeds))
    m = all_rows["model"].astype(str).isin(models)
    m &= all_rows["dataset"].astype(str).isin(datasets)
    m &= all_rows["model_seed"].astype(int).eq(0)
    m &= all_rows["intervention_seed"].astype(int).isin(seeds)
    m &= all_rows["evaluation_track"].astype(str).str.upper().eq("B")
    m &= all_rows["intervention_family"].astype(str).eq("structural")
    m &= ~bool_mask(all_rows["leakage_detected"])
    m &= bool_mask(all_rows["include_in_information_equivalent_structural"])
    if "diagnostic_subset_active" in all_rows.columns:
        m &= bool_mask(all_rows["diagnostic_subset_active"])
    d = all_rows.loc[m].copy()

    for c in ["RD", "representation_response_raw", "environment_distance"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    finite = np.isfinite(d["RD"]) & np.isfinite(d["representation_response_raw"]) & np.isfinite(d["environment_distance"]) & (d["environment_distance"] > 0)
    d = d.loc[finite].copy()

    # Validate exact coverage: 5 models x 2 datasets x 5 seeds x 10 configs by default.
    expected_keys = {(mo, ds, s) for mo in models for ds in datasets for s in seeds}
    actual_keys = set(zip(d["model"].astype(str), d["dataset"].astype(str), d["intervention_seed"].astype(int)))
    missing = sorted(expected_keys - actual_keys)
    if missing:
        raise SystemExit(f"Missing model-dataset-intervention_seed groups: {missing}")
    counts = d.groupby(["model", "dataset", "intervention_seed"]).size()
    bad = counts[counts != 10]
    if len(bad):
        raise SystemExit(f"Expected 10 structural configurations per model-dataset-seed. Bad groups:\n{bad}")

    ref = pd.read_csv(ref_path)
    d = apply_calibration(d, ref, args.eps_d)
    d["cluster_id"] = d["model"].astype(str) + "@@" + d["dataset"].astype(str)
    d.to_csv(inter / "01_merged_seed0_to4_rows.csv", index=False)

    # One row per configuration per intervention seed.
    group_seed = ["model", "dataset", "intervention_type", "paper_intensity_level", "intervention_seed"]
    seed_cfg = d.groupby(group_seed, dropna=False).agg(
        DMIS_raw=("DMIS_raw", "mean"),
        DMIS_cal=("DMIS_cal", "mean"),
        RD=("RD", "mean"),
        representation_response_raw=("representation_response_raw", "mean"),
        environment_distance=("environment_distance", "mean"),
    ).reset_index()
    seed_cfg["cluster_id"] = seed_cfg["model"].astype(str) + "@@" + seed_cfg["dataset"].astype(str)
    seed_cfg.to_csv(inter / "02_configuration_by_intervention_seed.csv", index=False)

    # Mean over intervention seeds: main targeted robustness statistic.
    group_cfg = ["model", "dataset", "intervention_type", "paper_intensity_level"]
    mean_cfg = seed_cfg.groupby(group_cfg, dropna=False).agg(
        DMIS_raw=("DMIS_raw", "mean"),
        DMIS_cal=("DMIS_cal", "mean"),
        RD=("RD", "mean"),
        DMIS_cal_std=("DMIS_cal", "std"),
        RD_std=("RD", "std"),
        n_intervention_seeds=("intervention_seed", "nunique"),
    ).reset_index()
    mean_cfg["cluster_id"] = mean_cfg["model"].astype(str) + "@@" + mean_cfg["dataset"].astype(str)
    mean_cfg.to_csv(inter / "03_configuration_mean_over_5_seeds.csv", index=False)

    main_table, boot = make_corr_table(mean_cfg, "DMIS_cal", "5-seed mean, representative subset", args.bootstrap, args.bootstrap_seed)
    main_table.to_csv(tables / "Table_R1_multiseed_mean_DMIScal_RD.csv", index=False)
    np.savez_compressed(inter / "Table_R1_cluster_bootstrap_samples.npz", correlations=boot)

    # Seed-wise stability on the same representative subset.
    seedwise_rows = []
    for s, g in seed_cfg.groupby("intervention_seed"):
        cv = corr_values(g["DMIS_cal"], g["RD"])
        for metric, (v, p) in cv.items():
            seedwise_rows.append({"intervention_seed": int(s), "Metric": metric, "Value": v, "p-value": p, "N": int(len(g))})
    seedwise = pd.DataFrame(seedwise_rows)
    seedwise.to_csv(tables / "Table_R2_seedwise_DMIScal_RD.csv", index=False)

    # Compare seed0-only and 5-seed mean within exactly the same subset.
    seed0 = seed_cfg[seed_cfg["intervention_seed"] == 0].copy()
    seed0["cluster_id"] = seed0["model"].astype(str) + "@@" + seed0["dataset"].astype(str)
    seed0_cv = corr_values(seed0["DMIS_cal"], seed0["RD"])
    mean_cv = corr_values(mean_cfg["DMIS_cal"], mean_cfg["RD"])
    cmp = []
    for metric in ["Pearson r", "Spearman rho", "Kendall tau"]:
        a = seed0_cv[metric][0]; b = mean_cv[metric][0]
        cmp.append({
            "Metric": metric,
            "seed0_only_value": a,
            "mean_over_seeds0_4_value": b,
            "difference_mean_minus_seed0": b - a,
            "same_sign": bool(np.sign(a) == np.sign(b)) if np.isfinite(a) and np.isfinite(b) else False,
        })
    pd.DataFrame(cmp).to_csv(tables / "Table_R3_seed0_vs_5seed_mean.csv", index=False)

    # Within-family raw-DMIS robustness, paper-aligned.
    fam_rows = []
    for fam, g in mean_cfg.groupby("intervention_type"):
        cv = corr_values(g["DMIS_raw"], g["RD"])
        for metric, (v, p) in cv.items():
            fam_rows.append({"intervention_type": fam, "Metric": metric, "Value": v, "p-value": p, "N": int(len(g))})
    pd.DataFrame(fam_rows).to_csv(tables / "Table_R4_within_family_5seed_mean_DMISraw_RD.csv", index=False)

    # Sign stability across the 5 intervention seeds.
    stab = []
    for metric, g in seedwise.groupby("Metric"):
        vals = g["Value"].to_numpy(float)
        stab.append({
            "Metric": metric,
            "n_seeds": int(len(vals)),
            "n_positive": int((vals > 0).sum()),
            "n_negative": int((vals < 0).sum()),
            "n_zero": int((vals == 0).sum()),
            "fraction_positive": float((vals > 0).mean()),
            "median_seedwise_value": float(np.median(vals)),
            "min_seedwise_value": float(np.min(vals)),
            "max_seedwise_value": float(np.max(vals)),
        })
    pd.DataFrame(stab).to_csv(tables / "Table_R5_seed_sign_stability.csv", index=False)

    mean_cfg[["model", "dataset", "intervention_type", "paper_intensity_level", "DMIS_cal", "DMIS_cal_std", "RD", "RD_std", "n_intervention_seeds"]].to_csv(
        tables / "configuration_seed_variability.csv", index=False
    )

    # Supplementary seed-wise correlation figure, default font.
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for metric in ["Pearson r", "Spearman rho", "Kendall tau"]:
        g = seedwise[seedwise["Metric"] == metric].sort_values("intervention_seed")
        ax.plot(g["intervention_seed"], g["Value"], marker="o", linewidth=1.6, label=metric)
    ax.axhline(0, color="0.35", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Intervention Seed")
    ax.set_ylabel("DMIS–RD Correlation")
    ax.set_xticks(seeds)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "Figure_Robustness_seedwise_correlations.pdf", bbox_inches="tight")
    fig.savefig(figs / "Figure_Robustness_seedwise_correlations.png", dpi=500, bbox_inches="tight")
    plt.close(fig)

    dump_json(meta / "robustness_manifest.json", {
        "created_at_utc": utc_now(),
        "primary_root": str(primary_root),
        "robustness_root": str(rob_root),
        "calibration_reference": str(ref_path),
        "models": models,
        "datasets": datasets,
        "model_seed": 0,
        "intervention_seeds": seeds,
        "expected_raw_rows": len(models) * len(datasets) * len(seeds) * 10,
        "actual_raw_rows": int(len(d)),
        "main_coordinate": "DMIS_cal for cross-family; DMIS_raw for within-family",
        "no_retraining_in_analysis": True,
        "interpretation_guard": "This targeted robustness check tests intervention-seed stability on a representative subset; it does not replace the 1,680-point primary analysis.",
    })

    print("\n" + "=" * 92)
    print("Section 5.1 targeted multi-intervention-seed robustness complete")
    print("=" * 92)
    print(main_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSeed-wise correlations:")
    print(seedwise.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
