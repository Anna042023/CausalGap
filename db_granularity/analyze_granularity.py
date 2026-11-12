#!/usr/bin/env python3
"""Analyze frozen randomized complete blocks; never rerun measured queries."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from queries import GRAINS

ROOT = Path(__file__).resolve().parent
CONFIGS = ("sqlite_default", "duckdb_1t", "duckdb_4t")
LABELS = {"sqlite_default": "SQLite default", "duckdb_1t": "DuckDB 1T", "duckdb_4t": "DuckDB 4T"}
COLORS = {"sqlite_default": "#305A91", "duckdb_1t": "#318571", "duckdb_4t": "#C87B30"}
BOOTSTRAP_SEED = 2026110706
BOOTSTRAP_RESAMPLES = 20000


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, records):
    keys = list(records[0]) if records else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def main():
    output = ROOT / "analysis"
    output.mkdir(exist_ok=True)
    records = [json.loads(line) for line in (ROOT / "measurements/raw.jsonl").read_text().splitlines()]
    measured = [r for r in records if r["phase"] == "measured"]
    warmups = [r for r in records if r["phase"] == "warmup"]
    protocol = json.loads((ROOT / "measurements/protocol.json").read_text())
    expected_count = protocol["blocks"] * len(CONFIGS) * len(GRAINS)
    assert len(measured) == expected_count == len(warmups)
    keyed = {(r["block"], r["config"], r["grain"]): r for r in measured}
    assert len(keyed) == expected_count
    all_cases = [(c, g["grain"]) for c in CONFIGS for g in GRAINS]
    blocks = sorted({r["block"] for r in measured})
    complete = [b for b in blocks if all(keyed[b, c, g]["status"] == "ok"
                 and keyed[b, c, g]["oracle_status"] == "pass" for c, g in all_cases)]
    excluded = sorted(set(blocks) - set(complete))
    assert len(complete) >= 2, "Insufficient complete successful blocks for paired inference"
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(complete), size=(BOOTSTRAP_RESAMPLES, len(complete)))

    def values(config, grain, field):
        return np.array([keyed[b, config, grain][field] for b in complete], dtype=float)

    def describe(array):
        q1, q3 = np.percentile(array, [25, 75])
        lo, hi = np.percentile(np.median(array[indices], axis=1), [2.5, 97.5])
        return {"n": len(array), "median": float(np.median(array)),
                "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
                "minimum": float(np.min(array)), "maximum": float(np.max(array)),
                "ci95_low": float(lo), "ci95_high": float(hi)}

    def paired_ratio(numerator, denominator):
        assert np.all(numerator > 0) and np.all(denominator > 0)
        logs = np.log(numerator / denominator)
        estimates = np.exp(np.mean(logs[indices], axis=1))
        lo, hi = np.percentile(estimates, [2.5, 97.5])
        return {"n_complete_blocks": len(logs), "geometric_mean_ratio": float(np.exp(np.mean(logs))),
                "ci95_low": float(lo), "ci95_high": float(hi),
                "median_paired_ratio": float(np.median(np.exp(logs))),
                "minimum_paired_ratio": float(np.min(np.exp(logs))),
                "maximum_paired_ratio": float(np.max(np.exp(logs)))}

    fields = ["wall_ms", "cpu_ms", "execute_wall_ms", "execute_cpu_ms", "fetch_wall_ms", "fetch_cpu_ms"]
    summaries = []
    flat = []
    for g in GRAINS:
        for config in CONFIGS:
            summary = {"config": config, "grain": g["grain"], "group_keys": list(g["keys"]),
                       "output_rows": g["expected_groups"], "complete_blocks_used": len(complete),
                       "measured_failure_count": sum(r["status"] != "ok" for r in measured
                         if r["config"] == config and r["grain"] == g["grain"])}
            for field in fields:
                summary[field] = describe(values(config, g["grain"], field))
            summary["peak_rss_mib"] = describe(values(config, g["grain"], "peak_rss_kib") / 1024)
            summary["rss_start_mib"] = describe(values(config, g["grain"], "rss_start_kib") / 1024)
            summary["additional_peak_rss_mib"] = describe((values(config, g["grain"], "peak_rss_kib")
                                                           - values(config, g["grain"], "rss_start_kib")) / 1024)
            summary["fetch_wall_fraction"] = describe(values(config, g["grain"], "fetch_wall_ms") /
                                                       values(config, g["grain"], "wall_ms"))
            summary["cpu_wall_ratio"] = describe(values(config, g["grain"], "cpu_ms") /
                                                  values(config, g["grain"], "wall_ms"))
            summaries.append(summary)
            row = {"config": config, "grain": g["grain"], "output_rows": g["expected_groups"],
                   "n_complete_blocks": len(complete), "measured_failure_count": summary["measured_failure_count"]}
            for field, stats in summary.items():
                if isinstance(stats, dict) and "median" in stats:
                    for key in ["median", "q1", "q3", "ci95_low", "ci95_high"]:
                        row[field + "_" + key] = stats[key]
            flat.append(row)
    thread_effects = []
    for g in GRAINS:
        grain = g["grain"]
        thread_effects.append({"grain": grain, "output_rows": g["expected_groups"],
            "wall_speedup_1t_over_4t": paired_ratio(values("duckdb_1t", grain, "wall_ms"), values("duckdb_4t", grain, "wall_ms")),
            "cpu_cost_4t_over_1t": paired_ratio(values("duckdb_4t", grain, "cpu_ms"), values("duckdb_1t", grain, "cpu_ms")),
            "peak_rss_4t_over_1t": paired_ratio(values("duckdb_4t", grain, "peak_rss_kib"), values("duckdb_1t", grain, "peak_rss_kib"))})
    granularity_effects = []
    for config in CONFIGS:
        granularity_effects.append({"config": config,
            "wall_ratio_full_over_run": paired_ratio(values(config, "horizon", "wall_ms"), values(config, "run", "wall_ms")),
            "cpu_ratio_full_over_run": paired_ratio(values(config, "horizon", "cpu_ms"), values(config, "run", "cpu_ms")),
            "peak_rss_ratio_full_over_run": paired_ratio(values(config, "horizon", "peak_rss_kib"), values(config, "run", "peak_rss_kib"))})
    cross_engine_effects = []
    for g in GRAINS:
        for config in ["duckdb_1t", "duckdb_4t"]:
            cross_engine_effects.append({"grain": g["grain"], "duckdb_config": config,
                "wall_ratio_duckdb_over_sqlite": paired_ratio(values(config, g["grain"], "wall_ms"),
                    values("sqlite_default", g["grain"], "wall_ms"))})
    failures = [r for r in records if r["status"] != "ok"]
    stats_protocol = {
        "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES, "confidence_level": 0.95,
        "method": "Nonparametric percentile bootstrap resampling whole randomized blocks with replacement",
        "paired_contrast": "Geometric mean of within-block ratios; exponentiated mean log ratio",
        "point_summaries": "Median per case over the same complete successful blocks",
        "resample_indices_shared_across_all_cases_and_contrasts": True,
        "complete_blocks": complete, "excluded_incomplete_blocks": excluded,
        "exclusion_policy": "Exclude an entire block from inference only if any measured case fails; retain every raw record and failure",
        "outlier_removal": "None", "multiple_comparison_adjustment": "None; reported intervals are marginal",
        "inference_scope": "Repeatability conditional on this fixed generated input, software, warm-query protocol and shared host",
        "source_raw_sha256": sha(ROOT / "measurements/raw.jsonl"),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
    }
    results = {"statistics_protocol": stats_protocol, "summary": summaries,
               "thread_effects": thread_effects, "granularity_effects": granularity_effects,
               "cross_engine_effects": cross_engine_effects,
               "counts": {"formal_queries": len(measured), "warmup_queries": len(warmups),
                          "formal_failures": sum(r["status"] != "ok" for r in measured),
                          "warmup_failures": sum(r["status"] != "ok" for r in warmups),
                          "total_oracle_passes": sum(r.get("oracle_status") == "pass" for r in records),
                          "hwm_reset_successes": sum(r.get("hwm_reset_status") == "ok" for r in records)}}
    save_json(output / "statistics.json", results)
    save_json(output / "statistical_protocol.json", stats_protocol)
    save_json(output / "failures.json", failures)
    csv_write(output / "summary.csv", flat)
    thread_flat = []
    for effect in thread_effects:
        row = {"grain": effect["grain"], "output_rows": effect["output_rows"]}
        for metric, stats in effect.items():
            if isinstance(stats, dict):
                row.update({metric + "_" + key: value for key, value in stats.items()})
        thread_flat.append(row)
    csv_write(output / "thread_effects.csv", thread_flat)
    make_figure(output, summaries, thread_effects)
    print(json.dumps({"counts": results["counts"], "thread_effects": thread_effects,
                      "granularity_effects": granularity_effects}, indent=2))


def make_figure(output, summaries, thread_effects):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "grid.linewidth": 0.4})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.40))
    x = np.arange(4)
    keyed = {(r["config"], r["grain"]): r for r in summaries}
    for ax, field, title, ylabel in [
        (axes[0], "wall_ms", "(a) Complete query latency", "Wall time (ms; log scale)"),
        (axes[2], "peak_rss_mib", "(c) Query-window memory", "Peak process RSS (MiB)"),
    ]:
        for j, config in enumerate(CONFIGS):
            stats = [keyed[config, g["grain"]][field] for g in GRAINS]
            y = np.array([s["median"] for s in stats])
            errors = np.array([[s["median"] - s["ci95_low"] for s in stats],
                               [s["ci95_high"] - s["median"] for s in stats]])
            ax.errorbar(x, y, yerr=errors, label=LABELS[config], color=COLORS[config],
                        marker=["o", "s", "^"][j], markersize=3.6, linewidth=1.1,
                        capsize=2, elinewidth=0.8)
        ax.set_title(title, loc="left", pad=6)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.3, labelspacing=0.2, borderaxespad=0.1)
    ax = axes[1]
    for i, (metric, label, color, marker) in enumerate([
        ("wall_speedup_1t_over_4t", "Wall speedup 1T/4T", "#305A91", "o"),
        ("cpu_cost_4t_over_1t", "CPU cost 4T/1T", "#9B5589", "D"),
    ]):
        stats = [e[metric] for e in thread_effects]
        y = np.array([s["geometric_mean_ratio"] for s in stats])
        errors = np.array([[s["geometric_mean_ratio"] - s["ci95_low"] for s in stats],
                           [s["ci95_high"] - s["geometric_mean_ratio"] for s in stats]])
        ax.errorbar(x + (i - 0.5) * 0.08, y, yerr=errors, label=label, color=color,
                    marker=marker, markersize=3.6, linewidth=1.1, capsize=2, elinewidth=0.8)
    ax.axhline(1, color="#666666", linewidth=0.8, linestyle=":")
    ax.set_title("(b) Thread-setting tradeoff", loc="left", pad=6)
    ax.set_ylabel("Paired geometric ratio", labelpad=2)
    ax.set_ylim(0.90, 1.65)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, handlelength=1.3, labelspacing=0.2, borderaxespad=0.1)
    for ax in axes:
        ax.set_xticks(x, ["252", "756", "24.9k", "299.4k"])
        ax.set_xlabel("Output groups", labelpad=2)
        ax.tick_params(width=0.6, length=3, pad=2)
        ax.margins(x=0.10)
    fig.subplots_adjust(left=0.065, right=0.975, top=0.89, bottom=0.21, wspace=0.42)
    for ext in ["pdf", "svg"]:
        fig.savefig(output / ("granularity_resources." + ext), metadata={"Creator": "Frozen R4 query measurements"})
    fig.savefig(output / "granularity_resources.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
