#!/usr/bin/env python3
"""Analyze Section 5.1 DMIS--RD association results.

Outputs are intentionally split into intermediate, tables, figures, diagnostics,
and metadata folders so later experiments can reuse the raw results without
rerunning model inference.
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
        df = pd.read_csv(f)
        df["model_seed"] = parse_model_seed(f, work_root)
        df["source_csv"] = str(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def validate_columns(df: pd.DataFrame) -> None:
    required = {
        "dataset", "model", "evaluation_track", "intervention_type",
        "intervention_family", "intervention_seed", "paper_intensity_level", "intervention_strength",
        "diagnostic_subset_active", "diagnostic_selected_test_windows", "diagnostic_original_test_windows",
        "environment_source", "leakage_detected",
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
    out["model_class"] = out["model"].map(class_map)
    unknown = sorted(out.loc[out["model_class"].isna(), "model"].astype(str).unique())
    if unknown:
        raise SystemExit(f"Unknown models in results: {unknown}")
    out["model_class"] = out["model_class"].astype(int)
    return out


def primary_filter(df: pd.DataFrame, track: str) -> tuple[pd.DataFrame, dict]:
    track = track.upper()
    m = df["evaluation_track"].astype(str).str.upper().eq(track)
    m &= df["intervention_family"].astype(str).eq("structural")
    m &= ~bool_mask(df["leakage_detected"])
    m &= bool_mask(df["diagnostic_subset_active"])
    if track == "B":
        m &= bool_mask(df["include_in_information_equivalent_structural"])
        validity_column = "include_in_information_equivalent_structural"
    else:
        m &= bool_mask(df["include_in_main_structural"])
        validity_column = "include_in_main_structural"
    d = df[m].copy()
    d["DMIS"] = pd.to_numeric(d["DMIS"], errors="coerce")
    d["RD"] = pd.to_numeric(d["RD"], errors="coerce")
    finite = np.isfinite(d["DMIS"].to_numpy()) & np.isfinite(d["RD"].to_numpy())
    dropped_nonfinite = int((~finite).sum())
    d = d.loc[finite].copy()
    report = {
        "track": track,
        "validity_column": validity_column,
        "raw_rows": int(len(df)),
        "primary_rows_after_validity": int(len(d)),
        "dropped_nonfinite": dropped_nonfinite,
        "environment_sources": d["environment_source"].value_counts(dropna=False).to_dict(),
    }
    return d, report


def aggregate_configurations(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1) average intervention-sampling repetitions within each trained model seed
    g1_cols = [
        "dataset", "model", "model_class", "model_seed",
        "intervention_type", "paper_intensity_level",
    ]
    per_model_seed = (
        primary.groupby(g1_cols, dropna=False)
        .agg(
            DMIS=("DMIS", "mean"),
            RD=("RD", "mean"),
            DMIS_sd_intervention=("DMIS", "std"),
            RD_sd_intervention=("RD", "std"),
            n_intervention_seeds=("intervention_seed", "nunique"),
            n_rows=("intervention_seed", "size"),
            representation_response_raw=("representation_response_raw", "mean"),
            intervention_strength=("intervention_strength", "mean"),
            environment_distance=("environment_distance", "mean"),
            diagnostic_selected_test_windows=("diagnostic_selected_test_windows", "min"),
            MAE0=("MAE0", "mean"),
            MAEk=("MAEk", "mean"),
        )
        .reset_index()
    )

    # 2) average model-training repetitions if more than one seed was requested.
    # The resulting model-dataset-family-intensity point is the unit used in
    # Table 1/Figure 3, preventing repeated random seeds from acting as pseudo-
    # independent observations.
    g2_cols = ["dataset", "model", "model_class", "intervention_type", "paper_intensity_level"]
    config = (
        per_model_seed.groupby(g2_cols, dropna=False)
        .agg(
            DMIS=("DMIS", "mean"),
            RD=("RD", "mean"),
            DMIS_sd_model_seed=("DMIS", "std"),
            RD_sd_model_seed=("RD", "std"),
            n_model_seeds=("model_seed", "nunique"),
            min_intervention_seeds=("n_intervention_seeds", "min"),
            representation_response_raw=("representation_response_raw", "mean"),
            intervention_strength=("intervention_strength", "mean"),
            environment_distance=("environment_distance", "mean"),
            diagnostic_selected_test_windows=("diagnostic_selected_test_windows", "min"),
            MAE0=("MAE0", "mean"),
            MAEk=("MAEk", "mean"),
        )
        .reset_index()
    )
    config["cluster_id"] = config["model"].astype(str) + "@@" + config["dataset"].astype(str)
    return per_model_seed, config


def family_balanced_points(config: pd.DataFrame) -> pd.DataFrame:
    # Equalize intervention-family contribution by first averaging intensities
    # within each family for each model-dataset pair.
    cols = ["dataset", "model", "model_class", "intervention_type"]
    d = (
        config.groupby(cols, dropna=False)
        .agg(
            DMIS=("DMIS", "mean"),
            RD=("RD", "mean"),
            n_intensities=("paper_intensity_level", "nunique"),
        )
        .reset_index()
    )
    d["cluster_id"] = d["model"].astype(str) + "@@" + d["dataset"].astype(str)
    return d


def corr_values(x: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, float]]:
    if len(x) < 3:
        return {"Pearson r": (np.nan, np.nan), "Spearman rho": (np.nan, np.nan), "Kendall tau": (np.nan, np.nan)}
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    kt = stats.kendalltau(x, y)
    return {
        "Pearson r": (float(pr.statistic), float(pr.pvalue)),
        "Spearman rho": (float(sr.statistic), float(sr.pvalue)),
        "Kendall tau": (float(kt.statistic), float(kt.pvalue)),
    }


def cluster_bootstrap_correlations(df: pd.DataFrame, n_boot: int, seed: int) -> tuple[dict[str, tuple[float, float]], np.ndarray]:
    clusters = np.asarray(sorted(df["cluster_id"].unique()), dtype=object)
    if len(clusters) < 2:
        raise ValueError("Need at least two model-dataset clusters for cluster bootstrap.")
    by_cluster = {c: df[df["cluster_id"] == c] for c in clusters}
    rng = np.random.default_rng(seed)
    names = ["Pearson r", "Spearman rho", "Kendall tau"]
    samples = np.full((n_boot, len(names)), np.nan, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = [by_cluster[c] for c in picked]
        boot = pd.concat(pieces, ignore_index=True)
        cv = corr_values(boot["DMIS"].to_numpy(float), boot["RD"].to_numpy(float))
        for j, name in enumerate(names):
            samples[b, j] = cv[name][0]
    cis = {}
    for j, name in enumerate(names):
        vals = samples[:, j]
        vals = vals[np.isfinite(vals)]
        cis[name] = (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
    return cis, samples


def make_table(df: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    x = df["DMIS"].to_numpy(float)
    y = df["RD"].to_numpy(float)
    point = corr_values(x, y)
    cis, boot = cluster_bootstrap_correlations(df, n_boot=n_boot, seed=seed)
    rows = []
    for name in ["Pearson r", "Spearman rho", "Kendall tau"]:
        value, p = point[name]
        lo, hi = cis[name]
        rows.append({
            "Metric": name,
            "Value": value,
            "95% CI lower": lo,
            "95% CI upper": hi,
            "p-value": p,
            "N configurations": int(len(df)),
            "N model-dataset clusters": int(df["cluster_id"].nunique()),
        })
    return pd.DataFrame(rows), boot


def save_latex_table(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & Value & 95\% CI lower & 95\% CI upper & $p$-value \\",
        r"\midrule",
    ]
    for _, r in table.iterrows():
        p = float(r["p-value"])
        ptxt = "$<10^{-4}$" if np.isfinite(p) and p < 1e-4 else (f"{p:.4f}" if np.isfinite(p) else "--")
        metric = str(r["Metric"]).replace("rho", r"$\rho$").replace("tau", r"$\tau$")
        lines.append(
            f"{metric} & {float(r['Value']):.3f} & {float(r['95% CI lower']):.3f} & "
            f"{float(r['95% CI upper']):.3f} & {ptxt} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def regression_bootstrap_band(df: pd.DataFrame, xgrid: np.ndarray, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df["DMIS"].to_numpy(float)
    y = df["RD"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = intercept + slope * xgrid

    clusters = np.asarray(sorted(df["cluster_id"].unique()), dtype=object)
    by_cluster = {c: df[df["cluster_id"] == c] for c in clusters}
    rng = np.random.default_rng(seed)
    curves = np.full((n_boot, len(xgrid)), np.nan, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([by_cluster[c] for c in picked], ignore_index=True)
        bx = boot["DMIS"].to_numpy(float)
        by = boot["RD"].to_numpy(float)
        if np.unique(bx).size < 2:
            continue
        bs, bi = np.polyfit(bx, by, 1)
        curves[b] = bi + bs * xgrid
    low = np.nanquantile(curves, 0.025, axis=0)
    high = np.nanquantile(curves, 0.975, axis=0)
    return yhat, low, high


def plot_figure3(config: pd.DataFrame, table: pd.DataFrame, out_pdf: Path, out_png: Path,
                 n_boot: int, seed: int) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 15,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    class_colors = {
        1: "#4C78A8",
        2: "#59A14F",
        3: "#F28E2B",
        4: "#B07AA1",
        5: "#E15759",
    }
    class_labels = {
        1: "Class I",
        2: "Class II",
        3: "Class III",
        4: "Class IV",
        5: "Class V",
    }

    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    for cls in sorted(config["model_class"].unique()):
        d = config[config["model_class"] == cls]
        ax.scatter(
            d["DMIS"], d["RD"], s=24, alpha=0.48,
            c=class_colors.get(int(cls), "#777777"),
            edgecolors="none", label=class_labels.get(int(cls), f"Class {cls}"),
            rasterized=True,
        )

    xmin, xmax = float(config["DMIS"].min()), float(config["DMIS"].max())
    if math.isclose(xmin, xmax):
        xmax = xmin + 1e-6
    xgrid = np.linspace(xmin, xmax, 240)
    yhat, low, high = regression_bootstrap_band(config, xgrid, n_boot=n_boot, seed=seed + 17)
    ax.plot(xgrid, yhat, linewidth=2.0, c="#222222", label="Linear Fit")
    ax.fill_between(xgrid, low, high, color="#777777", alpha=0.18, linewidth=0, label="95% CI")

    pearson = table.loc[table["Metric"] == "Pearson r", "Value"].iloc[0]
    spearman = table.loc[table["Metric"] == "Spearman rho", "Value"].iloc[0]
    kendall = table.loc[table["Metric"] == "Kendall tau", "Value"].iloc[0]
    txt = f"r = {pearson:.3f}\nρ = {spearman:.3f}\nτ = {kendall:.3f}\nN = {len(config):,}"
    ax.text(
        0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
        fontsize=12.5, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.92),
    )

    ax.set_xlabel("DMIS")
    ax.set_ylabel("Relative Degradation (RD)")
    ax.grid(True, linewidth=0.6, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def subgroup_correlations(config: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for key, g in config.groupby(col, dropna=False):
        if len(g) < 3:
            continue
        cv = corr_values(g["DMIS"].to_numpy(float), g["RD"].to_numpy(float))
        row = {col: key, "N": len(g)}
        for name, (v, p) in cv.items():
            safe = name.replace(" ", "_").replace("rho", "spearman").replace("tau", "kendall").replace("r", "pearson", 1)
            row[f"{safe}_value"] = v
            row[f"{safe}_p"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze CausalGap Section 5.1 DMIS--RD association")
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--track", choices=["A", "B", "a", "b"], default="B")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="Cluster-bootstrap replicates for 95%% CI. Statistical only; does not rerun models.")
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    args = ap.parse_args()

    root = args.experiment_root.expanduser().resolve()
    work_root = root / "work_runs"
    out_intermediate = root / "analysis" / "intermediate"
    out_tables = root / "analysis" / "tables"
    out_figures = root / "analysis" / "figures"
    out_diag = root / "analysis" / "diagnostics"
    out_meta = root / "analysis" / "metadata"
    for p in [out_intermediate, out_tables, out_figures, out_diag, out_meta]:
        p.mkdir(parents=True, exist_ok=True)

    raw = add_model_metadata(load_rows(work_root))
    validate_columns(raw)
    raw.to_csv(out_intermediate / "01_all_raw_rows.csv", index=False)

    primary, filter_report = primary_filter(raw, args.track)
    if primary.empty:
        raise SystemExit(
            f"No valid Track-{args.track.upper()} structural rows found. "
            "For the paper primary analysis use Track B and verify the DSAI observability gate."
        )
    primary.to_csv(out_intermediate / "02_primary_valid_structural_rows.csv", index=False)

    per_model_seed, config = aggregate_configurations(primary)
    per_model_seed.to_csv(out_intermediate / "03_per_model_seed_configuration_means.csv", index=False)
    config.to_csv(out_intermediate / "04_configuration_means_for_table1_figure3.csv", index=False)

    family_bal = family_balanced_points(config)
    family_bal.to_csv(out_intermediate / "05_family_balanced_points_robustness.csv", index=False)

    table, boot = make_table(config, n_boot=args.bootstrap, seed=args.bootstrap_seed)
    table.to_csv(out_tables / "Table1_DMIS_RD_Association.csv", index=False)
    md_lines = [
        "| Metric | Value | 95% CI lower | 95% CI upper | p-value | N configurations | N model-dataset clusters |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        md_lines.append(
            f"| {r['Metric']} | {float(r['Value']):.4f} | {float(r['95% CI lower']):.4f} | "
            f"{float(r['95% CI upper']):.4f} | {float(r['p-value']):.4g} | "
            f"{int(r['N configurations'])} | {int(r['N model-dataset clusters'])} |"
        )
    (out_tables / "Table1_DMIS_RD_Association.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    save_latex_table(table, out_tables / "Table1_DMIS_RD_Association.tex")
    np.savez_compressed(
        out_intermediate / "Table1_cluster_bootstrap_samples.npz",
        correlations=boot,
        metrics=np.array(["Pearson r", "Spearman rho", "Kendall tau"], dtype=object),
    )

    robust_table, robust_boot = make_table(family_bal, n_boot=args.bootstrap, seed=args.bootstrap_seed + 1)
    robust_table.to_csv(out_tables / "Table1_family_balanced_robustness.csv", index=False)
    np.savez_compressed(
        out_intermediate / "family_balanced_cluster_bootstrap_samples.npz",
        correlations=robust_boot,
        metrics=np.array(["Pearson r", "Spearman rho", "Kendall tau"], dtype=object),
    )

    plot_figure3(
        config,
        table,
        out_figures / "Figure3_DMIS_vs_RD.pdf",
        out_figures / "Figure3_DMIS_vs_RD.png",
        n_boot=args.bootstrap,
        seed=args.bootstrap_seed,
    )

    subgroup_correlations(config, "dataset").to_csv(out_diag / "correlation_by_dataset.csv", index=False)
    subgroup_correlations(config, "model_class").to_csv(out_diag / "correlation_by_model_class.csv", index=False)
    subgroup_correlations(config, "intervention_type").to_csv(out_diag / "correlation_by_intervention_type.csv", index=False)

    coverage = {
        "created_at_utc": utc_now(),
        "track": args.track.upper(),
        "raw_csv_files": int(raw["source_csv"].nunique()),
        "raw_rows": int(len(raw)),
        "primary_valid_rows": int(len(primary)),
        "configuration_rows": int(len(config)),
        "family_balanced_rows": int(len(family_bal)),
        "models": sorted(config["model"].unique().tolist()),
        "datasets": sorted(config["dataset"].unique().tolist()),
        "model_seeds": sorted(primary["model_seed"].astype(int).unique().tolist()),
        "intervention_seeds": sorted(primary["intervention_seed"].astype(int).unique().tolist()),
        "intervention_types": sorted(primary["intervention_type"].unique().tolist()),
        "environment_sources": primary["environment_source"].value_counts(dropna=False).to_dict(),
        "diagnostic_subset_active": True,
        "diagnostic_selected_windows_by_dataset": {
            str(k): int(v) for k, v in primary.groupby("dataset")["diagnostic_selected_test_windows"].min().to_dict().items()
        },
        "diagnostic_protocol_note": (
            "Compute-reduced Section 5.1 main association: deterministic uniform whole-test-span subset, "
            "one model seed and one intervention seed unless manifest states otherwise."
        ),
        "filter_report": filter_report,
        "bootstrap_replicates": int(args.bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "correlation_unit": "model-dataset-intervention-family-intensity after seed averaging",
        "ci_method": "95% percentile cluster bootstrap resampling model-dataset clusters",
        "figure_band_method": "cluster-bootstrap linear-regression mean line band",
    }
    dump_json(out_meta / "analysis_manifest.json", coverage)

    print("\n" + "=" * 84)
    print("Section 5.1 analysis complete")
    print("=" * 84)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nPrimary configuration rows: {len(config):,}")
    print(f"Model-dataset clusters     : {config['cluster_id'].nunique():,}")
    print(f"Table 1: {out_tables/'Table1_DMIS_RD_Association.csv'}")
    print(f"Figure 3: {out_figures/'Figure3_DMIS_vs_RD.pdf'}")
    print(f"All intermediate results preserved under: {root/'analysis'/'intermediate'}")


if __name__ == "__main__":
    main()
