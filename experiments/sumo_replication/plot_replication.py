#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-ready Section 5.2 figures from EXISTING CausalGap CSVs.

Changes relative to the uploaded script:
* Keep the enlarged typography and all original data calculations/checks.
* Keep the original five-item Family legend below the rank panels.
* Put only each panel's correlation-statistics box inside the panel, using a
  lighter and more transparent blue background.
* Label every positive stacked-bar segment at its actual center, including
  very small fractions; no separate note or displaced segment.
* Remove white outlines from Figure 6 stacked bars and use the same figure
  height for Figures 4 and 5.
* Increase every text element in Figures 4–6 by another 6 pt (12 pt above
  the original script), including titles, labels, ticks, legends and annotations.
* Place Figure 4 point estimates below their corresponding CI segments and
  widen the gap between its panels so the right y-axis labels remain separate.
* Use a still lighter, borderless blue background for Figure 5 statistic boxes.
* Increase all Figure 4–6 text by a further 4 pt, except the two Figure 5
  in-panel statistic boxes, whose text increases by 2 pt.
* Use short Figure 4 y labels, a narrower canvas, and a one-row legend above
  the vertical bar values so the legend and annotations do not overlap.
* Increase Figure 5 scatter-marker diameters by 3 pt in both panels and legend.

Run all figures:
    python -u plot_section52_compact_markers.py --only all
Override paths with --spec-csv, --replication-dir, --out-dir if necessary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from scipy.stats import kendalltau, spearmanr

PROJECT = Path("/data/wanganna/CausalGap")
REP = PROJECT / "experiment_results/sumo_replication/analysis/replication"
SPEC = PROJECT / "experiment_results/sumo_h1_protocol/analysis/h1/Table2_H1_weak_history_intensity_specification.csv"
OUT = PROJECT / "experiment_results/sumo_replication/analysis/figures"

REGIMES = [
    "stable_invariance",
    "adaptive_response",
    "pathological_rigidity",
    "over_reactive_instability",
]
REGIME_LABELS = [
    "Stable\nInvariance",
    "Adaptive\nResponse",
    "Pathological\nRigidity",
    "Over-Reactive\nInstability",
]
REGIME_SHORT_LABELS = [
    "Stable Invariance",
    "Adaptive Response",
    "Pathological Rigidity",
    "Over-Reactive Instability",
]
FAMILY_LABELS = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}
SPEC_LABELS = {
    "primary_mean_slope_max_distance": "Primary: Mean + Slope + Max",
    "history_mean_max_no_slope": "Mean + Max (No Slope)",
    "history_mean_only": "Mean Only",
    "nominal_intensity_slope": "Nominal Intensity Slope",
}
PAIR_COLORS = {"history": "#D5E7F1", "history_dmis": "#F4DFBF"}
PAIR_EDGE_COLORS = {"history": "#6595B4", "history_dmis": "#BF8F53"}
FAMILY_COLORS = {
    1: "#6FA8DC", 2: "#7CCBA2", 3: "#F2C572",
    4: "#E69A8D", 5: "#C7A0D8",
}
REGIME_COLORS = {
    "stable_invariance": "#79a3cd",
    "adaptive_response": "#b5dceb",
    "pathological_rigidity": "#eef8ed",
    "over_reactive_instability": "#feeeb1",
}
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "soft_teal_blue",
    ["#F8FBFD", "#E7F1F4", "#CFE3E8", "#A9CDD6", "#7EB2C3", "#5E96B3"],
)

# Default Matplotlib text size is 10 pt, axes title ~12 pt.
# All regular Figure 4–6 text is now 16 pt above the original script.
# The two Figure 5 in-panel statistic boxes are 14 pt above the original,
# because this revision increases them by 2 pt while all other text gains 4 pt.
FONT_DELTA = 16
RANK_STAT_DELTA = 14
FS_BASE = 10 + FONT_DELTA
FS_TITLE = 12 + FONT_DELTA
FS_SMALL = 8 + FONT_DELTA
FS_MODEL = 8.5 + FONT_DELTA
FS_LEGEND = 9 + FONT_DELTA
FS_RANK_STAT = 9 + RANK_STAT_DELTA
FS_VALUE = 10 + FONT_DELTA

# Shared styling for in-panel legends and correlation boxes.
STAT_FACE = "#EFF8FD"
STAT_EDGE = "none"
STAT_ALPHA = 0.38  # lighter and more transparent


def set_publication_fonts() -> None:
    plt.rcdefaults()                   # keep the original Matplotlib font family
    plt.rcParams.update({
        "font.size": FS_BASE,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_BASE,
        "xtick.labelsize": FS_BASE,
        "ytick.labelsize": FS_BASE,
        "legend.fontsize": FS_BASE,
        "figure.titlesize": FS_TITLE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def read_csv(path: Path, needed: set[str], description: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} is missing:\n  {path}\nDo not replace it with placeholder data."
        )
    d = pd.read_csv(path)
    missing = needed.difference(d.columns)
    if missing:
        raise ValueError(
            f"{description}: missing columns {sorted(missing)}; "
            f"found {list(d.columns)}\nFile: {path}"
        )
    return d


def numeric(d: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    out = d.copy()
    for c in columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if not np.isfinite(out[c].to_numpy(float)).all():
            raise ValueError(
                f"{label}: {c} has NaN/inf/non-numeric entries. "
                "Do not plot incomplete results."
            )
    return out


def save(fig, path: Path) -> None:
    for ext in ("pdf", "svg", "png"):
        target = path.with_suffix("." + ext)
        fig.savefig(target, bbox_inches="tight", dpi=350, facecolor="white")
        print(f"[SAVED] {target}", flush=True)
    plt.close(fig)


def axes_style(ax, axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=axis, alpha=0.22, linewidth=0.75, color="#BFBFBF")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", pad=7)


# =========================================================
# Figure 4: H1 specification sensitivity (risk + CI)
# =========================================================
def figure_h1(spec_csv: Path, out: Path) -> dict:
    cols = {
        "specification", "risk_error_history", "risk_plus_dmis",
        "risk_reduction", "ci95_lower", "ci95_upper",
    }
    d = numeric(
        read_csv(spec_csv, cols, "Synthetic-only H1 specification table"),
        ["risk_error_history", "risk_plus_dmis", "risk_reduction",
         "ci95_lower", "ci95_upper"],
        "H1",
    )
    if d.specification.duplicated().any():
        raise ValueError("Repeated H1 specification names.")
    order = list(SPEC_LABELS)
    absent = set(order) - set(d.specification)
    if absent:
        raise ValueError(f"Expected four completed specifications, missing: {sorted(absent)}")
    d = d.set_index("specification").loc[order].reset_index()
    if not np.allclose(
        d.risk_error_history - d.risk_plus_dmis,
        d.risk_reduction, atol=1e-7, rtol=1e-5,
    ):
        raise ValueError("H1 risk reductions do not equal risk_error_history - risk_plus_dmis.")
    if ((d.risk_reduction < d.ci95_lower) | (d.risk_reduction > d.ci95_upper)).any():
        raise ValueError("Some point estimates lie outside their stated confidence intervals.")

    labels_full = [SPEC_LABELS[x] for x in d.specification]
    labels_short = ["Primary", "No Slope", "Mean Only", "Nominal"]
    xx = np.arange(len(d))

    # Original: (14.6, 5.2). Larger panels leave room for the +6 pt typography.
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(20.0, 10.0),
        gridspec_kw={"width_ratios": [1.00, 1.12]},
    )
    # Short y labels allow a narrower canvas and a balanced inter-panel gap.
    # Both axes use the same bottom boundary.  The larger bottom margin gives
    # the rotated left-panel tick labels room without shifting either panel.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.23, wspace=0.32)

    a.bar(
        xx - 0.19, d.risk_error_history * 1e3, width=0.38,
        label="Error History", facecolor=PAIR_COLORS["history"],
        edgecolor=PAIR_EDGE_COLORS["history"], linewidth=1.0, hatch="///",
    )
    a.bar(
        xx + 0.19, d.risk_plus_dmis * 1e3, width=0.38,
        label="Error History + DMIS", facecolor=PAIR_COLORS["history_dmis"],
        edgecolor=PAIR_EDGE_COLORS["history_dmis"], linewidth=1.0, hatch="...",
    )
    ymax = float(max(d.risk_error_history.max(), d.risk_plus_dmis.max()) * 1e3)
    text_offset = ymax * 0.022
    for i, row in d.iterrows():
        for offset, raw in ((-0.19, row.risk_error_history),
                            (0.19, row.risk_plus_dmis)):
            a.text(
                i + offset, raw * 1e3 + text_offset, f"{raw:.6f}",
                ha="center", va="bottom", rotation=90,
                fontsize=FS_SMALL, clip_on=False,
            )
    # Reserve a clean band above the vertical values for the two-row legend.
    a.set_ylim(0, ymax * 1.85)
    a.set_xticks(
        xx, labels_short,
        rotation=28, ha="right", rotation_mode="anchor",
    )
    a.set_ylabel(r"Strong-RD Macro-MAE ($\times 10^{-3}$)")
    a.legend(
        loc="upper left", bbox_to_anchor=(0.02, 0.985), ncol=1,
        frameon=False, fontsize=FS_LEGEND, borderaxespad=0,
        labelspacing=0.35, handletextpad=0.55,
    )
    a.set_title("(a) Strong-RD Prediction Risk", loc="center", fontweight="bold", pad=17)
    axes_style(a)

    yy = np.arange(len(d))
    scale = 1e5
    estimate = d.risk_reduction.to_numpy(float) * scale
    left = (d.risk_reduction - d.ci95_lower).to_numpy(float) * scale
    right = (d.ci95_upper - d.risk_reduction).to_numpy(float) * scale
    b.errorbar(
        estimate, yy, xerr=np.vstack([left, right]),
        fmt="D", color="#4E79A7", ecolor="#81AACA",
        markerfacecolor="#4E83A8", markeredgecolor="#426F8F",
        markeredgewidth=0.9, elinewidth=1.8, capsize=5,
        markersize=8.5, zorder=3,
    )
    b.axvline(0, ls="--", lw=1.1, color="0.45")
    b.set_yticks(yy, labels_short)
    b.set_ylim(len(d) - 0.32, -0.60)  # more vertical breathing room, inverted axis
    b.set_xlabel(r"Risk Reduction ($\times 10^{-5}$; Positive = Lower Risk)", labelpad=12)
    b.set_title("(b) Incremental Value and 95% CIs", loc="center",
                fontweight="bold", pad=17)
    minx = float(d.ci95_lower.min() * scale)
    maxx = float(d.ci95_upper.max() * scale)
    span = max(maxx - minx, 1.0)
    b.set_xlim(minx - 0.10 * span, maxx + 0.10 * span)

    # Put each point estimate directly below its own CI segment. Because the
    # y-axis is inverted, increasing y moves the label downward on the page.
    for y, x_est, v in zip(yy, estimate, d.risk_reduction):
        b.annotate(
            f"{v:+.6f}", xy=(x_est, y + 0.28),
            ha="center", va="top", fontsize=FS_SMALL,
            bbox={"boxstyle": "round,pad=0.10", "fc": "white",
                  "ec": "none", "alpha": 0.88},
        )
    axes_style(b, "x")
    save(fig, out / "Fig52_H1_risk_and_CI")
    return {
        "h1_specifications": len(d),
        "h1_CIs_include_zero": int(((d.ci95_lower <= 0) & (d.ci95_upper >= 0)).sum()),
    }


# =========================================================
# Figure 5: rank correspondence
# =========================================================
def figure_rank(rep: Path, out: Path) -> dict:
    j = read_csv(
        rep / "cross_setting_model_means.csv",
        {"model", "model_family", "DMIS_sumo", "DMIS_synthetic",
         "RD_sumo", "RD_synthetic"},
        "Matched Synthetic–SUMO model means",
    )
    j = numeric(j, ["DMIS_sumo", "DMIS_synthetic", "RD_sumo", "RD_synthetic"],
                "Matched model means")
    if j.model.duplicated().any():
        raise ValueError("Cross-setting table must contain each model only once; duplicate names found.")
    stats = read_csv(
        rep / "ranking_replication_sumo_vs_synthetic.csv",
        {"metric", "n_models", "spearman_rho", "spearman_p", "kendall_tau", "kendall_p"},
        "Replication correlation table",
    )
    stats = numeric(stats, ["n_models", "spearman_rho", "spearman_p",
                            "kendall_tau", "kendall_p"], "Correlations")
    if len(j) != 21:
        raise ValueError(f"Expected 21 matched models, found {len(j)}.")
    if set(stats.metric) != {"DMIS", "RD"} or len(stats) != 2:
        raise ValueError("Correlation table must have exactly one DMIS and one RD row.")

    fig, axs = plt.subplots(1, 2, figsize=(20.0, 10.0))
    # Matplotlib scatter size is marker area in pt^2. Increasing the original
    # diameter sqrt(86) by 3 pt gives this area and updates legend handles too.
    rank_marker_size = (np.sqrt(86.0) + 3.0) ** 2
    # Reserve the lower margin for the original shared five-item Family legend.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.205, wspace=0.33)
    checks = {}
    for ax, metric, name in zip(axs, ["DMIS", "RD"], ["(a) DMIS", "(b) RD"]):
        x = j[f"{metric}_synthetic"].rank(ascending=False, method="average")
        y = j[f"{metric}_sumo"].rank(ascending=False, method="average")
        sp, kp = spearmanr(x, y), kendalltau(x, y)
        row = stats.set_index("metric").loc[metric]
        if (
            int(row.n_models) != len(j)
            or not np.isclose(sp.statistic, row.spearman_rho, atol=2e-5)
            or not np.isclose(kp.statistic, row.kendall_tau, atol=2e-5)
        ):
            raise ValueError(
                f"{metric} correlations disagree with model means. "
                "Re-run 03_analyze_replication.sh."
            )
        for fam in sorted(j.model_family.unique()):
            m = j.model_family.astype(str).eq(str(fam))
            ax.scatter(
                x[m], y[m], label=f"Family {FAMILY_LABELS.get(int(fam), fam)}",
                color=FAMILY_COLORS.get(int(fam), "#808080"),
                s=rank_marker_size, alpha=0.92, edgecolors="white", linewidths=0.8,
            )
        ax.plot([1, 21], [1, 21], "--", color="0.55", lw=1.1, zorder=0)
        ax.set_xlim(0.25, 21.75)
        ax.set_ylim(21.75, 0.25)
        ax.invert_xaxis()
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.set_xlabel(f"Synthetic {metric} Rank (1 = Highest)", labelpad=10)
        ax.set_ylabel(f"SUMO {metric} Rank (1 = Highest)", labelpad=10)
        ax.set_title(name, loc="center", fontweight="bold", pad=15)
        axes_style(ax)
        # Axes-relative coordinates keep this box inside the plot.
        ax.text(
            0.975, 0.025,
            f"Spearman $\\rho$ = {sp.statistic:+.3f}, p = {sp.pvalue:.3g}\n"
            f"Kendall $\\tau$ = {kp.statistic:+.3f}, p = {kp.pvalue:.3g}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS_RANK_STAT, zorder=6,
            bbox={"boxstyle": "round,pad=0.35", "fc": STAT_FACE,
                  "ec": "none", "lw": 0, "alpha": STAT_ALPHA},
        )
        checks[metric] = {
            "rho": float(sp.statistic), "p": float(sp.pvalue), "n_models": len(j),
        }
    # Preserve the original shared five-item Family legend below both panels.
    handles, legend_labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center",
        bbox_to_anchor=(0.5, 0.034), ncol=5,
        frameon=False, fontsize=FS_LEGEND,
    )
    save(fig, out / "Fig52_Synthetic_SUMO_rank_correspondence")
    return checks


def read_regimes(path: Path, label: str) -> pd.DataFrame:
    d = read_csv(path, {"model_family", *REGIMES}, label)
    d = numeric(d, REGIMES, label)
    d["model_family"] = pd.to_numeric(d["model_family"], errors="raise").astype(int)
    if len(d) != 5 or set(d.model_family) != set(range(1, 6)):
        raise ValueError(f"{label}: exactly one row for each family 1..5 required")
    d = d.set_index("model_family").loc[list(range(1, 6)), REGIMES]
    if (d < -1e-9).any().any() or not np.allclose(d.sum(axis=1), 1, atol=0.0002):
        raise ValueError(f"{label}: four fractions per family must be non-negative and sum to 1")
    return d


# =========================================================
# Figure 6: SUMO stacked regimes, optional Synthetic/SUMO heatmaps
# =========================================================
def figure_regimes(rep: Path, out: Path) -> dict:
    s = read_regimes(rep / "sumo_four_quadrant_distribution.csv", "SUMO regime fractions")
    synthetic = rep / "synthetic_four_quadrant_distribution.csv"
    t = read_regimes(synthetic, "Synthetic regime fractions") if synthetic.is_file() else None

    fig, ax = plt.subplots(figsize=(19.0, 9.2))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.95, bottom=0.19)
    bottom = np.zeros(5)
    # Keep actual segment heights and center every nonzero percentage normally.
    for col, lab in zip(REGIMES, REGIME_SHORT_LABELS):
        vals = s[col].to_numpy(float) * 100
        ax.bar(
            np.arange(5), vals, bottom=bottom, width=0.72,
            label=lab, color=REGIME_COLORS[col],
            edgecolor="none", linewidth=0,
        )
        for x, (val, old) in enumerate(zip(vals, bottom)):
            if val > 0:
                ax.text(
                    x, old + val / 2, f"{val:.2f}%",
                    ha="center", va="center", fontsize=FS_VALUE,
                    color="#354052", fontweight="medium",
                )
        bottom += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("Fraction Of Interventions (%)", labelpad=12)
    ax.set_xticks(range(5), [f"Family {FAMILY_LABELS[i]}" for i in range(1, 6)])
    axes_style(ax)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center",
        bbox_to_anchor=(0.5, 0.035), ncol=4,
        frameon=False, fontsize=FS_LEGEND,
    )
    save(fig, out / "Fig53_SUMO_regime_stacked")

    if t is not None:
        fig, axarr = plt.subplots(1, 2, figsize=(22.0, 9.0), sharey=True)
        fig.subplots_adjust(left=0.085, right=0.88, bottom=0.19, top=0.91, wspace=0.20)
        for ax, mat, name in zip(axarr, [t, s], ["(a) Synthetic", "(b) SUMO"]):
            im = ax.imshow(
                mat.to_numpy(float) * 100,
                vmin=0, vmax=100, aspect="auto", cmap=HEATMAP_CMAP,
            )
            ax.set_xticks(range(4), REGIME_LABELS, fontsize=FS_LEGEND)
            ax.set_yticks(range(5),
                          [f"Family {FAMILY_LABELS[i]}" for i in range(1, 6)])
            ax.set_title(name, loc="center", fontweight="bold", pad=15)
            for (row, col), value in np.ndenumerate(mat.to_numpy(float) * 100):
                ax.text(col, row, f"{value:.1f}", ha="center", va="center",
                        fontsize=FS_LEGEND,
                        color="white" if value >= 55 else "#334455")
            ax.tick_params(length=0, axis="x", pad=9)
            ax.tick_params(length=0, axis="y", pad=9)
        cax = fig.add_axes([0.91, 0.20, 0.017, 0.69])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Fraction (%)", labelpad=10)
        cb.ax.tick_params(labelsize=FS_BASE)
        save(fig, out / "Fig53_Synthetic_SUMO_regime_heatmaps")
    else:
        print("[SKIP] Synthetic regime heatmap: synthetic_four_quadrant_distribution.csv not present")
    return {"sumo_families": len(s), "synthetic_regimes_available": t is not None}


# =========================================================
# Supplement: model-level response/error scatter
# =========================================================
def _nonoverlapping_model_labels(fig, ax, d: pd.DataFrame, chosen: set[int]) -> None:
    """Try multiple label offsets, checking rendered bounds against earlier labels.

    This is dependency-free, unlike adjustText.  Text stays within axes, and
    the deterministic placement tries to avoid all 21 plotted points as well.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer=renderer)
    points_px = ax.transData.transform(d[["DMIS", "RD"]].to_numpy(float))
    used = []
    candidates = [
        (13, 16), (13, -18), (-13, 16), (-13, -18),
        (30, 27), (30, -31), (-30, 27), (-30, -31),
        (58, 12), (58, -16), (-58, 12), (-58, -16),
        (12, 44), (12, -46), (-12, 44), (-12, -46),
        (76, 36), (76, -38), (-76, 36), (-76, -38),
    ]
    for ix in sorted(chosen, key=lambda i: -len(str(d.loc[i, "model"]))):
        r = d.loc[ix]
        name = str(r.model)
        best = None
        for dx, dy in candidates:
            ann = ax.annotate(
                name, (r.DMIS, r.RD), xytext=(dx, dy), textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                fontsize=FS_MODEL, color="#354052",
                bbox={"fc": "white", "ec": "none", "alpha": 0.86, "pad": 0.15},
            )
            fig.canvas.draw()
            rect = ann.get_window_extent(renderer=renderer).expanded(1.07, 1.18)
            outside = max(0, bounds.x0 - rect.x0) + max(0, rect.x1 - bounds.x1)
            outside += max(0, bounds.y0 - rect.y0) + max(0, rect.y1 - bounds.y1)
            label_overlap = sum(
                max(0, min(rect.x1, prev.x1) - max(rect.x0, prev.x0))
                * max(0, min(rect.y1, prev.y1) - max(rect.y0, prev.y0))
                for prev in used
            )
            point_overlap = sum(
                rect.x0 - 6 < px < rect.x1 + 6
                and rect.y0 - 6 < py < rect.y1 + 6
                for px, py in points_px
            )
            score = outside * 10000 + label_overlap * 100 + point_overlap * 1000
            score += abs(dx) + abs(dy)
            if best is None or score < best[0]:
                best = (score, dx, dy, rect)
            ann.remove()
            if score < 100:
                break
        assert best is not None
        _, dx, dy, _ = best
        final = ax.annotate(
            name, (r.DMIS, r.RD), xytext=(dx, dy), textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=FS_MODEL, color="#354052",
            bbox={"fc": "white", "ec": "none", "alpha": 0.86, "pad": 0.15},
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "0.6"}
                if abs(dx) + abs(dy) > 65 else None,
        )
        fig.canvas.draw()
        used.append(final.get_window_extent(renderer=renderer).expanded(1.07, 1.18))


def figure_model_response(rep: Path, out: Path) -> dict:
    d = read_csv(
        rep / "sumo_model_response_error.csv",
        {"model_family", "model", "DMIS", "RD"},
        "Model-level SUMO DMIS/RD",
    )
    d = numeric(d, ["DMIS", "RD"], "SUMO model means")
    if len(d) != 21 or d.model.duplicated().any():
        raise ValueError("Expected one row per each of 21 model names")
    fig, ax = plt.subplots(figsize=(14.5, 10.0))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.91, bottom=0.235)
    for family, subset in d.groupby("model_family", sort=True):
        color = FAMILY_COLORS.get(int(family), "#808080")
        ax.scatter(
            subset.DMIS, subset.RD, color=color, s=90,
            edgecolor="white", linewidth=0.8,
            label=f"Family {FAMILY_LABELS.get(int(family), family)}",
            alpha=0.95,
        )
    ax.margins(x=0.16, y=0.18)
    ax.set_xlabel("Mean DMIS (SUMO)", labelpad=10)
    ax.set_ylabel("Mean RD (SUMO)", labelpad=10)
    ax.set_title("(a) Model-Level DMIS and RD", loc="left",
                 fontweight="bold", pad=15)
    axes_style(ax)
    chosen = set(d.nlargest(3, "RD").index) | set(d.nlargest(3, "DMIS").index)
    _nonoverlapping_model_labels(fig, ax, d, chosen)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=5, frameon=False, fontsize=FS_LEGEND,
        loc="lower center", bbox_to_anchor=(0.5, 0.075),
    )
    save(fig, out / "FigS_SUMO_model_DMIS_RD")
    if (d.model.eq("Lag-Llama") & d.DMIS.eq(0)).any():
        print("[AUDIT WARNING] Lag-Llama DMIS = 0 in existing CSV. "
              "Verify representation extraction before interpreting it.")
    return {
        "models": len(d),
        "lag_llama_zero_dmis": bool((d.model.eq("Lag-Llama") & d.DMIS.eq(0)).any()),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spec-csv", type=Path, default=SPEC)
    p.add_argument("--replication-dir", type=Path, default=REP)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--only", choices=["all", "h1", "rank", "regime", "model"],
                   default="all")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_publication_fonts()
    results = {
        "inputs": {"spec_csv": str(args.spec_csv),
                   "replication_dir": str(args.replication_dir)},
        "completed": {},
    }
    jobs = {
        "h1": lambda: figure_h1(args.spec_csv, args.out_dir),
        "rank": lambda: figure_rank(args.replication_dir, args.out_dir),
        "regime": lambda: figure_regimes(args.replication_dir, args.out_dir),
        "model": lambda: figure_model_response(args.replication_dir, args.out_dir),
    }
    for name in (jobs if args.only == "all" else [args.only]):
        print(f"\n[DRAW] {name}", flush=True)
        results["completed"][name] = jobs[name]()
    (args.out_dir / "figure_source_audit.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n[PASS] Saved plots and audit in {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"\n[STOP] {exc}", file=sys.stderr)
        sys.exit(2)
