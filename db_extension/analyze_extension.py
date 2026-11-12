#!/usr/bin/env python3
"""Analyze retained R3 timings without rerunning queries; produce paper artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
MEASUREMENTS = ROOT / "measurements"
BOOTSTRAPS = 20000
BOOTSTRAP_SEED = 2026110704
VARIANTS = ["sqlite_auto", "sqlite_indexed", "duckdb_scan"]
VARIANT_LABELS = {"sqlite_auto": "SQLite auto", "sqlite_indexed": "SQLite indexed",
                  "duckdb_scan": "DuckDB 1T"}
QUERY_LABELS = {"naive": "Naive", "checked": "Checked", "cached": "Cached"}
HARDWARE = ["t1_m64", "t1_m256", "t1_m1024", "t2_m1024", "t4_m1024"]


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(set(r) for r in rows)))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def stable_rng(label):
    value = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return np.random.default_rng(BOOTSTRAP_SEED ^ value)


def interval(values, statistic, label):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None
    indices = stable_rng(label).integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    estimates = statistic(values[indices], axis=1)
    return [float(v) for v in np.quantile(estimates, [0.025, 0.975])]


def group_summary(rows, keys, prefix):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary = []
    for group, group_rows in grouped.items():
        base = dict(zip(keys, group))
        successful = [r for r in group_rows if r["status"] == "ok"]
        item = dict(base, attempted=len(group_rows), successful=len(successful),
                    failures=len(group_rows)-len(successful))
        for metric in ["wall_ms", "cpu_ms", "peak_rss_kib"]:
            values = [float(r[metric]) for r in successful if r.get(metric) is not None]
            if not values:
                continue
            label = prefix + "/" + "/".join(group) + "/" + metric
            lo, hi = interval(values, np.median, label)
            item.update({f"{metric}_median": float(np.median(values)),
                         f"{metric}_q25": float(np.quantile(values, 0.25)),
                         f"{metric}_q75": float(np.quantile(values, 0.75)),
                         f"{metric}_median_ci_low": lo,
                         f"{metric}_median_ci_high": hi,
                         f"{metric}_mean": float(np.mean(values)),
                         f"{metric}_min": float(min(values)),
                         f"{metric}_max": float(max(values))})
        summary.append(item)
    return summary


def paired_effect(rows_a, rows_b, label, identifiers):
    a = {r["block"]: r for r in rows_a if r["status"] == "ok"}
    b = {r["block"]: r for r in rows_b if r["status"] == "ok"}
    common = sorted(set(a) & set(b))
    if not common:
        return dict(identifiers, paired_blocks=0)
    numerator = np.array([a[k]["wall_ms"] for k in common], float)
    denominator = np.array([b[k]["wall_ms"] for k in common], float)
    logs = np.log(numerator / denominator)
    lo_log, hi_log = interval(logs, np.mean, label)
    sd = float(np.std(logs, ddof=1)) if len(logs) > 1 else None
    return dict(identifiers, paired_blocks=len(common),
        geometric_mean_ratio=float(np.exp(np.mean(logs))),
        ci95_low=float(np.exp(lo_log)), ci95_high=float(np.exp(hi_log)),
        median_block_ratio=float(np.median(numerator / denominator)),
        percent_change=100*float(np.expm1(np.mean(logs))),
        mean_log_ratio=float(np.mean(logs)),
        paired_log_cohen_dz=float(np.mean(logs) / sd) if sd and sd > 0 else None,
        median_paired_difference_ms=float(np.median(numerator - denominator)),
        numerator_slower_blocks=int(np.sum(numerator > denominator)),
        numerator_faster_blocks=int(np.sum(numerator < denominator)),
        equal_blocks=int(np.sum(numerator == denominator)),
        paired_blocks_used=";".join(map(str, common)))


def fmt(value, digits=1):
    return f"{value:.{digits}f}" if value is not None else "--"


def ratio_text(effect):
    if not effect or not effect.get("paired_blocks"):
        return "--"
    return f"{effect['geometric_mean_ratio']:.2f} [{effect['ci95_low']:.2f}, {effect['ci95_high']:.2f}]"


def create_figure(primary_summary, effects, hardware_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.titlesize": 8.3, "axes.labelsize": 7.7, "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0, "legend.fontsize": 6.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.1})
    fig, axes = plt.subplots(1, 3, figsize=(7.18, 2.48))
    hindex = {(r["config"], r["query"]): r for r in hardware_summary}
    eindex = {(r["variant"], r["contrast"]): r for r in effects}
    colors = {"naive": "#526271", "checked": "#a55124", "cached": "#167d8d"}
    ax = axes[0]
    for j, (contrast, label, color) in enumerate([
        ("checked_over_naive", "Checked / naive", "#a55124"),
        ("checked_over_cached", "Checked / cached", "#167d8d")]):
        rows = [eindex[(v, contrast)] for v in VARIANTS]
        x = np.array([r["geometric_mean_ratio"] for r in rows])
        lo = np.array([r["ci95_low"] for r in rows])
        hi = np.array([r["ci95_high"] for r in rows])
        y = np.arange(3) + (j-.5)*.22
        ax.errorbar(x, y, xerr=[x-lo, hi-x], fmt="o", markersize=4,
                    color=color, capsize=2, label=label)
    ax.axvline(1, color="#777777", linestyle="--", linewidth=.8)
    ax.set_xscale("log")
    ax.set_yticks(range(3), ["SQLite\nauto", "SQLite\nindexed", "DuckDB\n1T"])
    ax.invert_yaxis()
    ax.set_xlabel("Paired ratio (95% CI)")
    ax.set_title("(a) Paired effects", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.22, linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=6.0, handletextpad=.3,
              labelspacing=.25)
    ax.set_ylim(2.7, -1.1)

    for ax, configs, labels, title, xlabel in [
        (axes[1], ["t1_m1024", "t2_m1024", "t4_m1024"], ["1", "2", "4"],
         "(b) Threads at 1 GiB", "DuckDB threads"),
        (axes[2], ["t1_m64", "t1_m256", "t1_m1024"], ["64", "256", "1024"],
         "(c) Memory at 1 thread", "DuckDB memory limit (MiB)")]:
        for j, query in enumerate(["checked", "cached"]):
            available = [(i, hindex[(config, query)]) for i, config in enumerate(configs)
                         if hindex[(config, query)]["successful"]]
            x = np.array([i for i, _ in available], float) + (j-.5)*.08
            rows = [r for _, r in available]
            y = np.array([r["wall_ms_median"] for r in rows])
            lo = np.array([r["wall_ms_median_ci_low"] for r in rows])
            hi = np.array([r["wall_ms_median_ci_high"] for r in rows])
            ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt="o-", markersize=3.5,
                        capsize=2, color=colors[query], label=QUERY_LABELS[query])
            for i, config in enumerate(configs):
                row = hindex[(config, query)]
                if row["failures"]:
                    ax.annotate(f"{query[0].upper()}: {row['failures']}/{row['attempted']} fail",
                                (i, .96-j*.11), xycoords=("data", "axes fraction"),
                                ha="center", va="top", fontsize=5.8, color=colors[query])
        ax.set_xticks(range(3), labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Wall time (ms)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=.22, linewidth=.5)
        ax.legend(loc="center right", frameon=False, fontsize=6.8)
        ax.set_yscale("log")
        ax.set_ylim(1.8, 160)
    fig.subplots_adjust(left=.085, right=.985, bottom=.24, top=.88, wspace=.58)
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "database_extension.pdf", metadata={"Title": "R3 database timing and resource sensitivity"})
    fig.savefig(figures / "database_extension.png", dpi=220)
    plt.close(fig)


def create_table(primary_summary, primary_effects, hardware_summary, hardware_cache_effects):
    p = {(r["variant"], r["query"]): r for r in primary_summary}
    h = {(r["config"], r["query"]): r for r in hardware_summary}
    e = {(r["variant"], r["contrast"]): r for r in primary_effects}
    he = {r["config"]: r for r in hardware_cache_effects}
    lines = [r"\begin{table*}[t]", r"\centering", r"\fontsize{8}{9.3}\selectfont",
        r"\setlength{\tabcolsep}{4.5pt}", r"\renewcommand{\arraystretch}{1.05}",
        r"\caption{New controlled relational experiment at 1,197,504 atoms. Medians use 20 randomized blocks; ratios are paired geometric means with 95\% bootstrap confidence intervals.}",
        r"\label{tab:db-r3-extension}", r"\rowcolors{2}{tablerow}{white}",
        r"\begin{tabular}{@{}lrrrcc@{}}", r"\toprule", r"\rowcolor{tablehead}",
        r"Configuration & Naive (ms) & Checked (ms) & Cached (ms) & Checked / naive & Checked / cached \\",
        r"\midrule"]
    for variant in VARIANTS:
        lines.append(" & ".join([VARIANT_LABELS[variant],
            *[fmt(p[(variant,q)]["wall_ms_median"]) for q in ["naive", "checked", "cached"]],
            ratio_text(e[(variant,"checked_over_naive")]),
            ratio_text(e[(variant,"checked_over_cached")])]) + r" \\")
    lines += [r"\midrule", r"\rowcolor{tablehead}",
        r"DuckDB: threads / MiB & Checked wall/CPU & Cached wall/CPU & Peak RSS C/V (MiB) & Checked / cached & Failures C/V \\",
        r"\midrule"]
    for config in HARDWARE:
        c, v = h[(config,"checked")], h[(config,"cached")]
        label = config.replace("t", "", 1).replace("_m", " / ")
        def pair(row):
            return fmt(row.get("wall_ms_median")) + "/" + fmt(row.get("cpu_ms_median"))
        def rss(row):
            return fmt(row.get("peak_rss_kib_median", 0)/1024) if row["successful"] else "--"
        lines.append(" & ".join([label, pair(c), pair(v), rss(c)+"/"+rss(v),
            ratio_text(he.get(config)), f"{c['failures']}/20; {v['failures']}/20"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{2pt}",
        r"\begin{minipage}{0.98\textwidth}\fontsize{8}{9.3}\selectfont",
        r"C/V denote checked/cached. CPU and wall times are in ms. Primary DuckDB uses \texttt{1GB}; the resource sweep uses explicit MiB (1024 MiB = 1 GiB). Peak RSS is the median query-window resident high-water mark, including warm resident state; a database memory limit is not a process RSS cap. Resource queries use a fresh process and one same-query warmup. These results measure configuration sensitivity on one shared host.",
        r"\end{minipage}", r"\end{table*}"]
    (ROOT / "database_extension_table.tex").write_text("\n".join(lines)+"\n")


def main():
    completion = json.loads((MEASUREMENTS / "completion.json").read_text())
    if completion["fatal_error"]:
        raise RuntimeError(completion["fatal_error"])
    primary_all = load_jsonl(MEASUREMENTS / "primary_raw.jsonl")
    hardware_all = load_jsonl(MEASUREMENTS / "hardware_raw.jsonl")
    primary = [r for r in primary_all if r["phase"] == "measured"]
    hardware = [r for r in hardware_all if r["phase"] == "measured"]
    if len(primary) != 180 or len(hardware) != 200:
        raise AssertionError((len(primary), len(hardware)))
    expected = json.loads((MEASUREMENTS / "input_verification.json").read_text())
    successes = [r for r in primary_all + hardware_all if r["status"] == "ok"]
    if any(r["output_sha256"] != expected["expected_output_sha256"] or
           r["scored_pairs"] != expected["expected_pairs"] or
           r["output_runs"] != expected["selected_runs"] or r["oracle_status"] != "pass"
           for r in successes):
        raise AssertionError("Oracle records are not consistent")
    psummary = group_summary(primary, ["variant", "query"], "primary")
    hsummary = group_summary(hardware, ["config", "query"], "hardware")
    effects = []
    for variant in VARIANTS:
        for a,b in [("checked","naive"),("checked","cached"),("naive","cached")]:
            contrast = f"{a}_over_{b}"
            effects.append(paired_effect(
                [r for r in primary if r["variant"] == variant and r["query"] == a],
                [r for r in primary if r["variant"] == variant and r["query"] == b],
                "primary/"+variant+"/"+contrast,
                {"variant":variant,"contrast":contrast,"numerator":a,"denominator":b}))
    hardware_effects = []
    hardware_cache_effects = []
    for config in HARDWARE:
        hardware_cache_effects.append(paired_effect(
            [r for r in hardware if r["config"]==config and r["query"]=="checked"],
            [r for r in hardware if r["config"]==config and r["query"]=="cached"],
            "hardware/cache/"+config,{"config":config,"contrast":"checked_over_cached"}))
        if config == "t1_m1024":
            continue
        for query in ["checked","cached"]:
            hardware_effects.append(paired_effect(
                [r for r in hardware if r["config"]=="t1_m1024" and r["query"]==query],
                [r for r in hardware if r["config"]==config and r["query"]==query],
                "hardware/scaling/"+config+"/"+query,
                {"config":config,"query":query,"contrast":"reference_over_config",
                 "reference":"t1_m1024"}))
    failures = [r for r in primary_all+hardware_all if r["status"] != "ok"]
    analysis = ROOT / "analysis"
    analysis.mkdir(exist_ok=True)
    for name,rows in [("primary_summary",psummary),("hardware_summary",hsummary),
                      ("paired_primary_effects",effects),("hardware_resource_effects",hardware_effects),
                      ("hardware_cache_effects",hardware_cache_effects),("failures",failures)]:
        write_json(analysis / f"{name}.json",rows)
        write_csv(analysis / f"{name}.csv",rows)
    protocol = {"resampling_unit":"randomized complete block, paired within each contrast",
        "bootstrap_draws":BOOTSTRAPS,"bootstrap_base_seed":BOOTSTRAP_SEED,
        "interval":"two-sided 95% percentile interval; no multiplicity adjustment",
        "ratio_estimator":"exp(mean(log(numerator_ms / denominator_ms))) across complete paired blocks",
        "additional_effect_sizes":"percent change; paired log Cohen dz; median paired ms difference; directional block count",
        "failure_handling":"retain every failure; no imputation; only complete successful pairs in ratios",
        "interval_scope":"repeatability conditional on this input, warm-cache protocol and shared host; not input-population or hardware-generalization uncertainty",
        "primary_measured":len(primary),"hardware_measured":len(hardware),
        "successful_with_warmups":len(successes),"failures_with_warmups":len(failures),
        "primary_successes":sum(r["status"]=="ok" for r in primary),
        "hardware_successes":sum(r["status"]=="ok" for r in hardware),
        "hwm_reset_failures_measured":sum(r.get("hwm_reset_status")!="ok" for r in hardware),
        "expected_output_sha256":expected["expected_output_sha256"]}
    write_json(analysis / "analysis_protocol.json",protocol)
    create_figure(psummary,effects,hsummary)
    create_table(psummary,effects,hsummary,hardware_cache_effects)
    print(json.dumps({"protocol":protocol,"figure":"figures/database_extension.pdf",
                      "table":"database_extension_table.tex"},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", default="measurements")
    arguments = parser.parse_args()
    MEASUREMENTS = ROOT / arguments.measurements
    main()
