#!/usr/bin/env python3
"""Generate the paper's database plots/tables from the successful raw pass."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
VARIANTS = ("sqlite_auto", "sqlite_indexed", "duckdb_scan")
LABELS = ("SQLite: automatic", "SQLite: composite", "DuckDB: scan")
COLORS = ("#0072B2", "#D55E00", "#009E73")
SCALES = ("small", "medium", "large")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", default="measurements")
    args = parser.parse_args()
    results = ROOT / args.measurements
    summary = read_csv(results / "latency_summary.csv")
    setup = read_csv(results / "setup_costs.csv")
    breaks = read_csv(results / "materialization_break_even.csv")
    lookup = {(r["scale"], r["variant"], r["query"]): r for r in summary}
    costs = {(r["scale"], r["variant"]): r for r in setup}
    qstar = {(r["scale"], r["variant"]): float(r["derived_q_star"]) for r in breaks}
    assert len(summary) == 27 and all(int(r["repeats"]) >= 5 for r in summary)
    raw = read_csv(results / "latency_repeats.csv")
    assert len(raw) == 162 and all(r["status"] == "ok" for r in raw)
    for scale in SCALES:
        assert len({r["output_sha256"] for r in raw if r["scale"] == scale}) == 1

    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5,
                         "axes.titlesize": 9, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15))
    x = np.array([36288, 290304, 1197504], dtype=float)
    for axis, query, title in zip(axes, ("naive", "checked", "cached"),
                                  ("(a) Naive join", "(b) Checked pairs", "(c) Cached pairs")):
        for variant, label, color, marker in zip(VARIANTS, LABELS, COLORS, ("o", "s", "^")):
            data = [lookup[(scale, variant, query)] for scale in SCALES]
            medians = np.array([float(r["median_ms"]) for r in data])
            lower = medians - np.array([float(r["q25_ms"]) for r in data])
            upper = np.array([float(r["q75_ms"]) for r in data]) - medians
            axis.errorbar(x, medians, yerr=np.array([lower, upper]), color=color,
                          marker=marker, linewidth=1.25, markersize=3.6, capsize=2,
                          label=label)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(x, ("36k", "290k", "1.20M"))
        axis.set_xlabel("Forecast atoms")
        axis.set_title(title, loc="left")
        axis.grid(axis="y", which="major", alpha=0.2)
    axes[0].set_ylabel("Query latency (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.28, wspace=0.30)
    fig.savefig(figures / "database_experiments.pdf", bbox_inches="tight")
    fig.savefig(figures / "database_experiments.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    def latency(variant, query):
        row = lookup[("large", variant, query)]
        return rf"\shortstack{{{float(row['median_ms']):.1f}\\({float(row['iqr_ms']):.1f})}}"

    table_rows = []
    for variant, label in zip(VARIANTS, ("SQLite auto", "SQLite indexed", "DuckDB")):
        table_rows.append(label + " & " + " & ".join(latency(variant, query)
                         for query in ("naive", "checked", "cached")) + r" \\")
    index_speedup = (float(lookup[("large", "sqlite_auto", "naive")]["median_ms"])
                     / float(lookup[("large", "sqlite_indexed", "naive")]["median_ms"]))
    before = [int(costs[("large", v)]["db_bytes_before_materialization"])
              / (1024**2) for v in VARIANTS]
    after = [int(costs[("large", v)]["db_bytes_after_materialization"])
             / (1024**2) for v in VARIANTS]
    builds = [float(costs[("large", v)]["materialize_ms"]) for v in VARIANTS]
    cache_gain = [float(lookup[("large", v, "checked")]["median_ms"])
                  / float(lookup[("large", v, "cached")]["median_ms"])
                  for v in VARIANTS]
    findings = {"source": str(results.relative_to(ROOT)),
                "complete_measured_query_calls": 135, "warmup_calls": 27,
                "sqlite_index_naive_speedup_large": index_speedup,
                "large_materialization_ms": dict(zip(VARIANTS, builds)),
                "large_database_mib_before_cache": dict(zip(VARIANTS, before)),
                "large_database_mib_after_cache": dict(zip(VARIANTS, after)),
                "large_checked_to_cached_ratio": dict(zip(VARIANTS, cache_gain)),
                "large_derived_q_star": {v:qstar[("large", v)] for v in VARIANTS},
                "plot_library": matplotlib.__version__}
    (ROOT / "findings.json").write_text(json.dumps(findings, indent=2) + "\n")

    setup_rows = []
    for i, (variant, label) in enumerate(zip(VARIANTS, ("SQLite auto", "SQLite indexed", "DuckDB"))):
        cost = costs[("large", variant)]
        index_cell = f"{float(cost['index_ms']):.1f}" if variant == "sqlite_indexed" else "--"
        setup_rows.append(label + f" & {index_cell} & {builds[i]:.1f} & {after[i]:.2f} & "
                          + f"{qstar[('large', variant)]:.2f}" + r" \\")
    latex = r"""% Generated by db_experiments/analyze.py from a completed measurement run.
% New controlled relational experiments. No original forecasting runs are implied.
\subsection{Query Plans and Physical Design}
\label{sec:db_performance}

Figure~\ref{fig:db_relational} reports all three scales.
At 1,197,504 atoms, the composite index reduces the SQLite self-join median
by __INDEX_SPEEDUP__$\times$, while checked grouping becomes slower
(Table~\ref{tab:db_latency}). Captured plans show an automatic partial
covering index for the first self-join and persistent index probes for the
second. Both checked SQLite plans materialize key groups and use temporary
grouping structures. The index therefore benefits pair lookup without
benefiting the complete audit. DuckDB uses sequential scans and hash
operators. Because naive and checked queries use different algorithms,
their latency difference does not isolate the marginal cost of validation.
The declared engine configurations also use different memory settings.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{db_experiments/figures/database_experiments.pdf}
\caption{New controlled relational experiments. Each panel shows median
query latency and interquartile range from five measured repetitions after
one warm-up. Both axes use logarithmic scales. Automatic SQLite indexes remain
enabled. Cached queries use validated pairs from an unchanged snapshot.}
\label{fig:db_relational}
\end{figure*}

\begin{table}[t]
\centering
\fontsize{8}{9.5}\selectfont
\caption{Largest-scale query latency: median (interquartile range), in
milliseconds, for 1,197,504 atoms and 299,376 selected pairs.}
\label{tab:db_latency}
\begin{tabular}{lrrr}
\toprule
Configuration & Naive & Checked & Cached \\
\midrule
__LATENCY_ROWS__
\bottomrule
\end{tabular}
\end{table}

Index construction costs __INDEX_MS__\,ms and raises the largest SQLite database
from __SQLITE_BASE__ to __SQLITE_INDEX__\,MiB before caching.
Table~\ref{tab:db_setup} includes the validated-pair build cost and final
storage. Setup costs are single measurements. For an unchanged snapshot,
let $B$ denote the measured build cost and $t_c,t_m$ the checked and cached
query medians. Comparing $qt_c$ with $B+qt_m$ gives the derived crossover
\begin{equation}
 q^*=\frac{B}{t_c-t_m},\qquad t_c>t_m.
\end{equation}
The largest-scale values are __Q_VALUES__ for SQLite automatic,
SQLite indexed, and DuckDB. These values are calculated from measured
components, not from an observed sequence of repeated workloads. Values
near one are sensitive to build-time and latency variation and do not
establish guaranteed first-query savings.

\begin{table}[t]
\centering
\fontsize{8}{9.5}\selectfont
\caption{Largest-scale setup costs and storage. Index and validated-pair
build costs are milliseconds. MiB is total database size after caching.
$q^*$ is a derived point estimate.}
\label{tab:db_setup}
\begin{tabular}{lrrrr}
\toprule
Configuration & Index & Build & MiB & $q^*$ \\
\midrule
__SETUP_ROWS__
\bottomrule
\end{tabular}
\end{table}

All 135 measured query calls and 27 warm-ups agree with independently
generated integer error sums, pair counts, MAE/RD, and cross-engine output
hashes. Raw repetitions, plans, load costs, settings, and compressed inputs
are retained. The findings concern a single-host, warm-cache workload with
one selected interface. Update maintenance, concurrent clients, cold starts,
and broader selectivities remain unmeasured. These database measurements
supply no additional forecasting-accuracy or H1/H2 evidence.
"""
    replacements = {
        "__LATENCY_ROWS__": "\n".join(table_rows),
        "__SETUP_ROWS__": "\n".join(setup_rows),
        "__INDEX_SPEEDUP__": f"{index_speedup:.2f}",
        "__INDEX_MS__": f"{float(costs[('large', 'sqlite_indexed')]['index_ms']):.1f}",
        "__SQLITE_BASE__": f"{before[0]:.2f}",
        "__SQLITE_INDEX__": f"{before[1]:.2f}",
        "__BUILD_VALUES__": ", ".join(f"{v:.1f}" for v in builds),
        "__SIZE_VALUES__": ", ".join(f"{v:.2f}" for v in after),
        "__Q_VALUES__": ", ".join(f"{qstar[('large', v)]:.2f}" for v in VARIANTS),
    }
    for token, value in replacements.items():
        latex = latex.replace(token, value)
    output = ROOT.parent / "retained/db_results.tex"
    output.parent.mkdir(exist_ok=True)
    output.write_text(latex, encoding="utf-8")
    print("Verified 135 measured calls and 27 warm-ups; wrote plots, findings, and", output)


if __name__ == "__main__":
    main()

