#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats


CALIBRATED_X_COLUMNS = [
    "DMIS_cal",
    "DMIScal",
    "dmis_cal",
    "DMIS_calibrated",
    "calibrated_DMIS",
]


def csv_has_calibrated_dmis(path: Path) -> bool:
    """Check the CSV header without loading the whole file."""
    try:
        cols = list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return any(c in cols for c in CALIBRATED_X_COLUMNS)


def find_input_csv(project_root: Path) -> Path:
    """
    Prefer the configuration-level output produced by corrected V3 analysis.

    Important:
    The old V6 script accidentally looked for a non-existent corrected filename,
    then fell back to the pre-calibration Fast-V2 CSV whose only DMIS column was
    raw `DMIS`. That caused the KeyError seen by the user.
    """
    base = project_root / "experiment_results/observational_diagnostics"

    candidates = [
        # Corrected V3: this is the intended source.
        base / "analysis_calibrated/intermediate/04_configuration_means_corrected.csv",

        # Future/alternate corrected-analysis locations, if present.
        base / "analysis_final/intermediate/04_configuration_means_corrected.csv",
        base / "analysis_final/intermediate/04_configuration_means_corrected.csv",
        base / "analysis_final/intermediate/04_configuration_means_corrected.csv",
    ]

    for p in candidates:
        if p.exists() and csv_has_calibrated_dmis(p):
            return p

    # Robust fallback: search only corrected/final analysis folders and require
    # that the CSV header actually contains a calibrated-DMIS column.
    search_roots = [
        base / "analysis_calibrated",
        base / "analysis_final",
        base / "analysis_final",
        base / "analysis_final",
    ]
    for sr in search_roots:
        if not sr.exists():
            continue
        for p in sorted(sr.rglob("*.csv")):
            if "configuration" in p.name.lower() and csv_has_calibrated_dmis(p):
                return p

    raise FileNotFoundError(
        "Could not find the corrected configuration-level CSV containing DMIS_cal.\n"
        "Expected, for example:\n"
        f"  {base / 'analysis_calibrated/intermediate/04_configuration_means_corrected.csv'}\n"
        "Please rerun the corrected V3 analysis or pass --input-csv explicitly."
    )


def pick_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find {label} column. Tried: {candidates}. "
        f"Existing columns: {list(df.columns)}"
    )


def class_sort_key(label: str) -> Tuple[int, str]:
    text = str(label).strip()
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
    if text.startswith("Class "):
        suffix = text.split("Class ", 1)[1].strip()
        if suffix in roman:
            return roman[suffix], text
        if suffix.isdigit():
            return int(suffix), text
    if text in roman:
        return roman[text], text
    return 999, text


def normalize_class_series(s: pd.Series) -> pd.Series:
    roman_from_int = {
        1: "Class I",
        2: "Class II",
        3: "Class III",
        4: "Class IV",
        5: "Class V",
    }
    out = []
    for x in s.astype(str):
        t = x.strip()
        if t in roman_from_int.values():
            out.append(t)
            continue
        if t.startswith("Class "):
            tail = t.split("Class ", 1)[1].strip()
            if tail.isdigit():
                out.append(roman_from_int.get(int(tail), t))
                continue
            if tail in ["I", "II", "III", "IV", "V"]:
                out.append("Class " + tail)
                continue
        if t.isdigit():
            out.append(roman_from_int.get(int(t), t))
            continue
        out.append(t)
    return pd.Series(out, index=s.index)


def fit_line_and_ci(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    alpha: float = 0.05,
):
    n = len(x)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_hat = intercept + slope * x_grid

    if n < 3 or np.allclose(x.var(), 0):
        nan = np.full_like(x_grid, np.nan)
        return y_hat, nan, nan

    y_fit = intercept + slope * x
    residuals = y - y_fit
    s_err = np.sqrt(np.sum(residuals ** 2) / max(n - 2, 1))
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)
    if np.isclose(sxx, 0):
        nan = np.full_like(x_grid, np.nan)
        return y_hat, nan, nan

    t_val = stats.t.ppf(1 - alpha / 2, max(n - 2, 1))
    conf = t_val * s_err * np.sqrt(
        1 / n + ((x_grid - x_mean) ** 2) / sxx
    )
    return y_hat, y_hat - conf, y_hat + conf


def main():
    parser = argparse.ArgumentParser(
        description="Final two-panel Figure 3 using calibrated DMIS."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/data/wanganna/CausalGap"),
    )
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--zoom-xmax", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--alpha", type=float, default=0.62)
    args = parser.parse_args()

    input_csv = (
        args.input_csv
        if args.input_csv is not None
        else find_input_csv(args.project_root)
    )
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    if not csv_has_calibrated_dmis(input_csv):
        raise SystemExit(
            f"The selected input CSV does not contain calibrated DMIS: {input_csv}\n"
            f"Columns: {list(pd.read_csv(input_csv, nrows=0).columns)}\n"
            "Use analysis_calibrated/intermediate/"
            "04_configuration_means_corrected.csv instead."
        )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.project_root
        / "experiment_results/observational_diagnostics/"
          "analysis_final/figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    x_col = pick_column(df, CALIBRATED_X_COLUMNS, "x/DMIS_cal")
    y_col = pick_column(
        df,
        ["RD", "rd", "relative_degradation"],
        "y/RD",
    )
    class_col = pick_column(
        df,
        ["model_class", "class", "Class", "architecture_class"],
        "model class",
    )

    plot_df = df[[x_col, y_col, class_col]].copy()
    plot_df = plot_df.rename(
        columns={
            x_col: "DMIS_cal",
            y_col: "RD",
            class_col: "model_class",
        }
    )
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    plot_df["model_class"] = normalize_class_series(
        plot_df["model_class"]
    )

    x = plot_df["DMIS_cal"].to_numpy(float)
    y = plot_df["RD"].to_numpy(float)
    n = len(plot_df)

    pearson_r, _ = stats.pearsonr(x, y)
    spearman_rho, _ = stats.spearmanr(x, y)
    kendall_tau, _ = stats.kendalltau(x, y)

    classes = sorted(
        plot_df["model_class"].dropna().unique(),
        key=class_sort_key,
    )
    # Publication-friendly, high-contrast five-family palette.
    # The hues are distinct in both print and screen viewing, while avoiding
    # the overly saturated look of Matplotlib's default tab colors.
    colors = {
        "Class I": "#4C78A8",   # muted blue
        "Class II": "#F28E2B",  # warm orange
        "Class III": "#59A14F", # green
        "Class IV": "#E15759",  # coral red
        "Class V": "#9C6BC1",   # soft purple
    }

    family_names = {
        "Class I": "Family I: Forecasting STGNNs",
        "Class II": "Family II: Robust and Structure-Aware Models",
        "Class III": "Family III: Structural-Dynamics Models",
        "Class IV": "Family IV: Generic Time-Series Models",
        "Class V": "Family V: Foundation Models",
    }

    # Default Matplotlib font; only sizes are adjusted.
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.titlesize": 21,
            "axes.titleweight": "bold",
            "axes.labelsize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
        }
    )

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(16.4, 6.6)
    )

    # 给顶部图例单独留出空间，同时控制两个子图间距
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.14,
        top=0.72,
        wspace=0.18
    )

    for cls in classes:
        sub = plot_df[plot_df["model_class"] == cls]
        ax_full.scatter(
            sub["DMIS_cal"],
            sub["RD"],
            s=args.point_size,
            alpha=args.alpha,
            color=colors.get(cls),
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )

    x_grid = np.linspace(float(x.min()), float(x.max()), 300)
    fit, lo, hi = fit_line_and_ci(x, y, x_grid)
    ax_full.plot(
        x_grid, fit, color="black", linewidth=2.3, zorder=5
    )
    ax_full.fill_between(
        x_grid, lo, hi, color="gray", alpha=0.18, zorder=1
    )
    ax_full.set_title("(a) Full Range", loc="center", pad=12)
    ax_full.set_xlabel("Calibrated DMIS")
    ax_full.set_ylabel("RD")
    ax_full.grid(True, linestyle="--", linewidth=0.7, alpha=0.18)
    ax_full.set_axisbelow(True)

    stats_text = (
        f"r = {pearson_r:.3f}\n"
        f"ρ = {spearman_rho:.3f}\n"
        f"τ = {kendall_tau:.3f}\n"
        f"N = {n:,}"
    )
    ax_full.text(
        0.70,
        0.70,
        stats_text,
        transform=ax_full.transAxes,
        fontsize=18,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="0.78",
            linewidth=0.8,
            alpha=0.94,
        ),
    )

    zoom_df = plot_df[
        plot_df["DMIS_cal"] <= args.zoom_xmax
    ].copy()

    for cls in classes:
        sub = zoom_df[zoom_df["model_class"] == cls]
        if len(sub) == 0:
            continue
        ax_zoom.scatter(
            sub["DMIS_cal"],
            sub["RD"],
            s=args.point_size,
            alpha=args.alpha,
            color=colors.get(cls),
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )

    zoom_xmin = min(
        float(zoom_df["DMIS_cal"].min())
        if len(zoom_df)
        else 0.0,
        0.0,
    )
    x_grid_zoom = np.linspace(
        zoom_xmin, args.zoom_xmax, 300
    )
    fit_z, lo_z, hi_z = fit_line_and_ci(
        x, y, x_grid_zoom
    )
    ax_zoom.plot(
        x_grid_zoom,
        fit_z,
        color="black",
        linewidth=2.3,
        zorder=5,
    )
    ax_zoom.fill_between(
        x_grid_zoom,
        lo_z,
        hi_z,
        color="gray",
        alpha=0.18,
        zorder=1,
    )

    ax_zoom.set_title(
        "(b) Central Range",
        loc="center",
        pad=12,
    )
    ax_zoom.set_xlabel("Calibrated DMIS")
    ax_zoom.set_ylabel("RD")
    ax_zoom.set_xlim(zoom_xmin, args.zoom_xmax)
    ax_zoom.grid(True, linestyle="--", linewidth=0.7, alpha=0.18)
    ax_zoom.set_axisbelow(True)

    if len(zoom_df):
        y_low = float(np.quantile(zoom_df["RD"], 0.01))
        y_high = float(np.quantile(zoom_df["RD"], 0.99))
        pad = max(
            0.02, 0.08 * (y_high - y_low + 1e-9)
        )
        ax_zoom.set_ylim(y_low - pad, y_high + pad)

    # Larger legend dots.
    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="w",
            markerfacecolor=colors.get(cls, "gray"),
            markeredgewidth=0,
            markersize=11.0,
            alpha=args.alpha,
            label=family_names.get(cls, cls),
        )
        for cls in classes
    ]
    fit_handle = Line2D(
        [0], [0], color="black", linewidth=2.3, label="OLS Fit"
    )
    ci_handle = Patch(
        facecolor="gray",
        edgecolor="none",
        alpha=0.18,
        label="95% CI",
    )

    handles = class_handles + [fit_handle, ci_handle]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",

        # Five long family names + OLS/CI are intentionally split over
        # two rows to preserve readability at publication size.
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),

        # Spacious but compact academic-style legend.
        columnspacing=1.10,
        handlelength=1.45,
        handletextpad=0.45,
        labelspacing=0.75,

        # Top legend is enlarged by four points relative to the old 13.5.
        fontsize=17.5,
    )

    png = output_dir / "Figure3_DMIScal_vs_RD_two_panel.png"
    pdf = output_dir / "Figure3_DMIScal_vs_RD_two_panel.pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "input_csv": str(input_csv),
        "n_points": int(n),
        "calibrated_dmis_column": x_col,
        "zoom_xmax": float(args.zoom_xmax),
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "kendall_tau": float(kendall_tau),
        "notes": [
            "Uses corrected configuration-level calibrated DMIS.",
            "Default Matplotlib font with all figure text enlarged by two points.",
            "Top legend enlarged by four points and split into two rows.",
            "Larger scatter and legend markers.",
            "Panel (b) title changed to Central Range.",
            "Five model classes are shown as named Families I-V.",
            "Publication-friendly five-family color palette.",
            "Centered subplot titles.",
            "YAxis label is RD.",
            "OLS Fit replaces the old Linear Fit label.",
            "No points are deleted for the full-range panel.",
        ],
    }
    (
        output_dir
        / "Figure3_two_panel_provenance.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("[INPUT]", input_csv)
    print("[CHECK] x column:", x_col)
    print(
        f"[CHECK] N={n:,}, r={pearson_r:.3f}, "
        f"rho={spearman_rho:.3f}, tau={kendall_tau:.3f}"
    )
    print("[DONE]", png)
    print("[DONE]", pdf)


if __name__ == "__main__":
    main()
