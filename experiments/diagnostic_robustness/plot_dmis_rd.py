#!/usr/bin/env python3
"""Final Section 5.1 Figure 3.

Uses the corrected V3 configuration-level results (DMIS_cal, RD) and creates a
full-range scatter with an inset zoom.  No model evaluation is performed.

Important plotting policy:
* Correlations are read from the untransformed calibrated-DMIS Table 1.
* No points are deleted, clipped, winsorized, or log-transformed for statistics.
* The inset is visualization-only and uses the same raw calibrated-DMIS axis.
* Matplotlib's DEFAULT font family is used (no font.family override).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def _roman(n: int) -> str:
    return {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}.get(int(n), str(n))


def _regression_band(df: pd.DataFrame, xgrid: np.ndarray, n_boot: int, seed: int):
    x = df["DMIS_cal"].to_numpy(float)
    y = df["RD"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = intercept + slope * xgrid

    clusters = np.asarray(sorted(df["cluster_id"].astype(str).unique()), dtype=object)
    by = {c: df[df["cluster_id"].astype(str) == c] for c in clusters}
    rng = np.random.default_rng(seed)
    curves = np.full((n_boot, len(xgrid)), np.nan, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([by[c] for c in picked], ignore_index=True)
        bx = boot["DMIS_cal"].to_numpy(float)
        byy = boot["RD"].to_numpy(float)
        if np.unique(bx).size < 2:
            continue
        s, i = np.polyfit(bx, byy, 1)
        curves[b] = i + s * xgrid
    lo = np.nanquantile(curves, 0.025, axis=0)
    hi = np.nanquantile(curves, 0.975, axis=0)
    return yhat, lo, hi


def _auto_inset_y(df: pd.DataFrame, xmax: float) -> tuple[float, float]:
    g = df[(df["DMIS_cal"] >= 0) & (df["DMIS_cal"] <= xmax)]
    if len(g) < 20:
        g = df
    y = g["RD"].to_numpy(float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return -0.1, 0.5
    # Zoom the dense central region while retaining almost all points in that x-range.
    lo, hi = np.quantile(y, [0.01, 0.99]) if len(y) >= 100 else (float(np.min(y)), float(np.max(y)))
    if math.isclose(lo, hi):
        lo, hi = lo - 0.05, hi + 0.05
    pad = 0.08 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot final Section 5.1 Figure 3 with full range + inset")
    ap.add_argument("--experiment-root", type=Path, required=True,
                    help="Primary Fast-V2 experiment root.")
    ap.add_argument("--corrected-analysis-dir", type=Path, default=None,
                    help="Defaults to <experiment-root>/analysis_calibrated")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Defaults to <experiment-root>/analysis_final/figures")
    ap.add_argument("--inset-xmax", type=float, default=5.0,
                    help="Inset x-axis upper bound. Use 10 if desired; does not affect statistics.")
    ap.add_argument("--inset-ymin", type=float, default=None)
    ap.add_argument("--inset-ymax", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="Cluster-bootstrap curves used only for the displayed regression CI band.")
    ap.add_argument("--bootstrap-seed", type=int, default=2041)
    args = ap.parse_args()

    root = args.experiment_root.expanduser().resolve()
    corrected = (root / "analysis_calibrated") if args.corrected_analysis_dir is None else args.corrected_analysis_dir.expanduser().resolve()
    out = (root / "analysis_final" / "figures") if args.output_dir is None else args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    config_path = corrected / "intermediate" / "04_configuration_means_corrected.csv"
    table_path = corrected / "tables" / "Table1_DMIScal_RD_Association.csv"
    if not config_path.exists():
        raise SystemExit(f"Missing corrected configuration file: {config_path}\nRun corrected V3 analysis first.")
    if not table_path.exists():
        raise SystemExit(f"Missing corrected Table 1: {table_path}\nRun corrected V3 analysis first.")

    df = pd.read_csv(config_path)
    table = pd.read_csv(table_path)
    for c in ["DMIS_cal", "RD", "model_class"]:
        if c not in df.columns:
            raise SystemExit(f"Missing column {c} in {config_path}")
    if "cluster_id" not in df.columns:
        df["cluster_id"] = df["model"].astype(str) + "@@" + df["dataset"].astype(str)
    finite = np.isfinite(pd.to_numeric(df["DMIS_cal"], errors="coerce")) & np.isfinite(pd.to_numeric(df["RD"], errors="coerce"))
    df = df.loc[finite].copy()
    df["DMIS_cal"] = pd.to_numeric(df["DMIS_cal"])
    df["RD"] = pd.to_numeric(df["RD"])
    df["model_class"] = pd.to_numeric(df["model_class"], errors="raise").astype(int)

    # Default Matplotlib font: intentionally do NOT set font.family/font.serif.
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    classes = sorted(df["model_class"].unique())
    color_map = {cls: default_colors[i % len(default_colors)] if default_colors else None for i, cls in enumerate(classes)}

    for cls in classes:
        g = df[df["model_class"] == cls]
        ax.scatter(
            g["DMIS_cal"], g["RD"],
            s=22, alpha=0.46, edgecolors="none",
            color=color_map[cls], label=f"Class {_roman(cls)}", rasterized=True,
        )

    xmin = float(df["DMIS_cal"].min())
    xmax = float(df["DMIS_cal"].max())
    if math.isclose(xmin, xmax):
        xmax = xmin + 1e-6
    xgrid = np.linspace(xmin, xmax, 260)
    yhat, lo, hi = _regression_band(df, xgrid, args.bootstrap, args.bootstrap_seed)
    ax.plot(xgrid, yhat, linewidth=1.8, color="0.15", label="Linear Fit")
    ax.fill_between(xgrid, lo, hi, color="0.5", alpha=0.18, linewidth=0, label="95% CI")

    vals = {str(r["Metric"]): float(r["Value"]) for _, r in table.iterrows()}
    txt = (
        f"r = {vals.get('Pearson r', np.nan):.3f}\n"
        f"ρ = {vals.get('Spearman rho', np.nan):.3f}\n"
        f"τ = {vals.get('Kendall tau', np.nan):.3f}\n"
        f"N = {len(df):,}"
    )
    ax.text(
        0.985, 0.985, txt, transform=ax.transAxes,
        ha="right", va="top", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.72", alpha=0.94),
        zorder=20,
    )
    ax.set_xlabel("Calibrated DMIS")
    ax.set_ylabel("Relative Degradation (RD)")
    ax.grid(True, linewidth=0.6, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.61, 0.90))

    # Inset: same untransformed x coordinate, no point deletion. It only magnifies the dense region.
    inset_xmax = float(args.inset_xmax)
    inset_ymin, inset_ymax = _auto_inset_y(df, inset_xmax)
    if args.inset_ymin is not None:
        inset_ymin = float(args.inset_ymin)
    if args.inset_ymax is not None:
        inset_ymax = float(args.inset_ymax)

    axins = inset_axes(ax, width="46%", height="42%", loc="lower right", borderpad=1.25)
    for cls in classes:
        g = df[df["model_class"] == cls]
        axins.scatter(g["DMIS_cal"], g["RD"], s=13, alpha=0.48, edgecolors="none", color=color_map[cls], rasterized=True)
    mask_grid = xgrid <= inset_xmax
    if mask_grid.sum() >= 2:
        axins.plot(xgrid[mask_grid], yhat[mask_grid], linewidth=1.4, color="0.15")
        axins.fill_between(xgrid[mask_grid], lo[mask_grid], hi[mask_grid], color="0.5", alpha=0.16, linewidth=0)
    axins.set_xlim(max(0.0, xmin), inset_xmax)
    axins.set_ylim(inset_ymin, inset_ymax)
    axins.grid(True, linewidth=0.45, alpha=0.16)
    axins.tick_params(labelsize=8.5)
    axins.set_title(f"Zoom: DMIS ≤ {inset_xmax:g}", fontsize=9.5)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.55", lw=0.8, alpha=0.7)

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.98)
    pdf = out / "Figure3_DMIScal_vs_RD.pdf"
    png = out / "Figure3_DMIScal_vs_RD.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Save exact figure provenance/settings.
    meta = pd.DataFrame([{
        "source_config": str(config_path),
        "source_table": str(table_path),
        "N": int(len(df)),
        "x_min": xmin,
        "x_max": xmax,
        "inset_x_min": max(0.0, xmin),
        "inset_x_max": inset_xmax,
        "inset_y_min": inset_ymin,
        "inset_y_max": inset_ymax,
        "font_policy": "Matplotlib default font family; no font.family override",
        "statistical_transform": "none",
        "point_filtering": "none beyond non-finite safety check",
    }])
    meta.to_csv(out / "Figure3_provenance.csv", index=False)
    print(f"[DONE] {pdf}")
    print(f"[DONE] {png}")
    print(f"Inset: x=[{max(0.0, xmin):.3g}, {inset_xmax:.3g}], y=[{inset_ymin:.3g}, {inset_ymax:.3g}]")


if __name__ == "__main__":
    main()
