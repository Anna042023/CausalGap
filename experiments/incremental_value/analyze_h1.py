#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from protocol import DEFAULT_EXPERIMENT_ROOT, LAMBDA_GRID, json_dump  # noqa: E402


CONTROLS_CONT = [
    "strong_intervention_magnitude",
    "baseline_IPE",
    "log_parameter_count",
]
CONTROLS_BIN = [
    "native_topology_access",
    "recurrent_flag",
    "convolutional_flag",
    "attention_flag",
    "pretraining_flag",
]
CONTROLS_CAT = ["setting", "adaptation_level"]

FEATURE_SETS = {
    "mean_only_error": ["weak_RD_mean"],
    "error_history": ["weak_RD_mean", "weak_RD_slope", "weak_RD_max"],
    "error_history_plus_dmis": ["weak_RD_mean", "weak_RD_slope", "weak_RD_max", "weak_DMIScal_mean"],
    "error_history_plus_response_history": [
        "weak_RD_mean", "weak_RD_slope", "weak_RD_max",
        "weak_DMIScal_mean", "weak_DMIScal_slope", "weak_DMIScal_max",
    ],
}


def fit_design(train: pd.DataFrame, test: pd.DataFrame, extra_features: list[str]):
    cont = CONTROLS_CONT + [c for c in extra_features if c not in CONTROLS_BIN and c not in CONTROLS_CAT]
    # Every H_RD / DMIS feature is continuous. Binary controls remain unscaled.
    means = train[cont].astype(float).mean(axis=0)
    stds = train[cont].astype(float).std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    Xtr_parts = [((train[cont].astype(float) - means) / stds).to_numpy()]
    Xte_parts = [((test[cont].astype(float) - means) / stds).to_numpy()]
    names = list(cont)

    Xtr_parts.append(train[CONTROLS_BIN].astype(float).to_numpy())
    Xte_parts.append(test[CONTROLS_BIN].astype(float).to_numpy())
    names.extend(CONTROLS_BIN)

    # Train-fold-only categorical vocabulary: unseen held-out categories map to all zeros.
    for cat in CONTROLS_CAT:
        levels = sorted(train[cat].astype(str).dropna().unique().tolist())
        for lv in levels:
            Xtr_parts.append((train[cat].astype(str).eq(lv).astype(float).to_numpy()[:, None]))
            Xte_parts.append((test[cat].astype(str).eq(lv).astype(float).to_numpy()[:, None]))
            names.append(f"{cat}={lv}")

    Xtr = np.concatenate(Xtr_parts, axis=1).astype(float)
    Xte = np.concatenate(Xte_parts, axis=1).astype(float)
    return Xtr, Xte, names


def ridge_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, lam: float) -> np.ndarray:
    """Small dense ridge solver with an unpenalized intercept.

    H1 has only O(10) predictors and O(100) units, so a direct solve is much
    faster than repeatedly constructing sklearn estimators inside thousands of
    cluster-bootstrap nested CV fits.
    """
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.float64)
    Xa = np.concatenate([np.ones((len(Xtr), 1)), Xtr], axis=1)
    Xea = np.concatenate([np.ones((len(Xte), 1)), Xte], axis=1)
    pen = np.eye(Xa.shape[1], dtype=np.float64) * float(lam)
    pen[0, 0] = 0.0
    lhs = Xa.T @ Xa + pen
    rhs = Xa.T @ ytr
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(lhs) @ rhs
    return Xea @ coef


def inner_lambda(train: pd.DataFrame, feature_key: str, group_col: str, lambdas: list[float]) -> float:
    groups = train[group_col].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 2:
        return float(lambdas[0])
    n_splits = min(5, len(unique))
    cv = GroupKFold(n_splits=n_splits)
    y = train["strong_RD"].to_numpy(float)
    dummy = np.zeros((len(train), 1))

    # Build every standardized fold design exactly once; only the cheap ridge
    # linear solve changes across lambda values.
    cached = []
    for tr_idx, va_idx in cv.split(dummy, y, groups):
        tr = train.iloc[tr_idx]
        va = train.iloc[va_idx]
        Xtr, Xva, _ = fit_design(tr, va, FEATURE_SETS[feature_key])
        cached.append((Xtr, tr["strong_RD"].to_numpy(float), Xva, va["strong_RD"].to_numpy(float)))

    best_lam, best_score = None, float("inf")
    for lam in lambdas:
        fold_mae = []
        for Xtr, ytr, Xva, yva in cached:
            pred = ridge_predict(Xtr, ytr, Xva, float(lam))
            fold_mae.append(float(np.mean(np.abs(yva - pred))))
        score = float(np.mean(fold_mae))
        if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and (best_lam is None or lam < best_lam)):
            best_lam, best_score = float(lam), score
    return float(best_lam)

def split_values(df: pd.DataFrame, split_kind: str):
    if split_kind == "model_family":
        return "model_family", sorted(df["model_family"].unique().tolist())
    if split_kind == "model":
        return "model", sorted(df["model"].astype(str).unique().tolist())
    if split_kind == "setting":
        return "setting", sorted(df["setting"].astype(str).unique().tolist())
    if split_kind == "intervention_type":
        return "intervention_type", sorted(df["intervention_type"].astype(str).unique().tolist())
    raise ValueError(split_kind)


def run_outer(df: pd.DataFrame, split_kind: str, feature_key: str, lambdas: list[float], group_col: str = "model"):
    col, vals = split_values(df, split_kind)
    fold_rows, pred_rows = [], []
    for held in vals:
        mask = df[col].astype(str).eq(str(held))
        test = df[mask].copy()
        train = df[~mask].copy()
        if train.empty or test.empty:
            continue
        lam = inner_lambda(train, feature_key, group_col, lambdas)
        Xtr, Xte, names = fit_design(train, test, FEATURE_SETS[feature_key])
        pred = ridge_predict(Xtr, train["strong_RD"].to_numpy(float), Xte, lam)
        truth = test["strong_RD"].to_numpy(float)
        mae = float(mean_absolute_error(truth, pred))
        fold_rows.append({
            "split": split_kind,
            "held_out": str(held),
            "predictor": feature_key,
            "macro_fold_mae": mae,
            "lambda": lam,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": int(Xtr.shape[1]),
        })
        for j, (_, r) in enumerate(test.iterrows()):
            pred_rows.append({
                "unit_id": r["unit_id"],
                "model": r["model"],
                "model_family": int(r["model_family"]),
                "setting": r["setting"],
                "intervention_type": r["intervention_type"],
                "held_out": str(held),
                "predictor": feature_key,
                "truth": float(truth[j]),
                "prediction": float(pred[j]),
                "absolute_error": float(abs(truth[j] - pred[j])),
                "lambda": lam,
            })
    f = pd.DataFrame(fold_rows)
    p = pd.DataFrame(pred_rows)
    risk = float(f["macro_fold_mae"].mean()) if len(f) else np.nan
    return risk, f, p


def comparison(df: pd.DataFrame, split_kind: str, lambdas: list[float], group_col: str = "model"):
    rb, fb, pb = run_outer(df, split_kind, "error_history", lambdas, group_col)
    rd, fd, pdm = run_outer(df, split_kind, "error_history_plus_dmis", lambdas, group_col)

    # A reduced/smoke protocol may contain only one setting (e.g. synthetic-only).
    # In that case leave-one-setting-out has no admissible train/test fold.  Record
    # the split as unavailable rather than crashing; the final two-setting H1 run
    # still produces the prespecified robustness comparison.
    if fb.empty or fd.empty or "held_out" not in fb.columns or "held_out" not in fd.columns:
        m = pd.DataFrame(columns=["held_out", "risk_error_history", "risk_plus_dmis", "risk_reduction"])
        return {
            "split": split_kind,
            "risk_error_history": np.nan,
            "risk_plus_dmis": np.nan,
            "risk_reduction": np.nan,
            "positive_folds": 0,
            "n_folds": 0,
            "available": False,
        }, m, pb, pdm

    m = fb[["held_out", "macro_fold_mae"]].rename(columns={"macro_fold_mae": "risk_error_history"}).merge(
        fd[["held_out", "macro_fold_mae"]].rename(columns={"macro_fold_mae": "risk_plus_dmis"}), on="held_out", how="inner"
    )
    m["risk_reduction"] = m["risk_error_history"] - m["risk_plus_dmis"]
    return {
        "split": split_kind,
        "risk_error_history": rb,
        "risk_plus_dmis": rd,
        "risk_reduction": float(rb - rd),
        "positive_folds": int((m["risk_reduction"] > 0).sum()),
        "n_folds": int(len(m)),
        "available": True,
    }, m, pb, pdm


def bootstrap_models_within_family(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    chunks = []
    copy_id = 0
    for fam, fg in df.groupby("model_family", sort=True):
        models = sorted(fg["model"].astype(str).unique().tolist())
        sampled = rng.choice(models, size=len(models), replace=True)
        for m in sampled:
            c = fg[fg["model"].astype(str) == str(m)].copy()
            c["bootstrap_original_model"] = str(m)
            c["bootstrap_cluster_id"] = f"F{int(fam)}_copy{copy_id:03d}_{m}"
            # Keep the model feature itself unchanged; it is not an H1 predictor.
            chunks.append(c)
            copy_id += 1
    return pd.concat(chunks, ignore_index=True)


def primary_bootstrap(df: pd.DataFrame, n_boot: int, seed: int, lambdas: list[float]):
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boot = bootstrap_models_within_family(df, rng)
        rec, _, _, _ = comparison(boot, "model_family", lambdas, group_col="bootstrap_cluster_id")
        vals[b] = rec["risk_reduction"]
        if (b + 1) % max(1, n_boot // 20) == 0:
            print(f"[BOOT] {b+1}/{n_boot}", flush=True)
    return vals


def exact_signflip_p(model_effects: np.ndarray, seed: int = 0, mc_samples: int = 1000000) -> tuple[float, str, int]:
    effects = np.asarray(model_effects, dtype=float)
    effects = effects[np.isfinite(effects)]
    n = len(effects)
    if n == 0:
        return np.nan, "none", 0
    observed = abs(float(np.mean(effects)))
    if n <= 22:
        total = 1 << n
        extreme = 0
        chunk = 1 << min(n, 16)
        idx = np.arange(n, dtype=np.uint64)
        for start in range(0, total, chunk):
            ids = np.arange(start, min(start + chunk, total), dtype=np.uint64)[:, None]
            signs = 1.0 - 2.0 * ((ids >> idx[None, :]) & 1).astype(float)
            vals = np.abs((signs @ effects) / n)
            extreme += int(np.sum(vals >= observed - 1e-15))
        return float(extreme / total), "exact", total
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(int(mc_samples), n))
    vals = np.abs(signs @ effects / n)
    return float((1 + np.sum(vals >= observed)) / (len(vals) + 1)), "monte_carlo", int(mc_samples)


def model_level_effects(pb: pd.DataFrame, pdm: pd.DataFrame) -> pd.DataFrame:
    a = pb[["unit_id", "model", "absolute_error"]].rename(columns={"absolute_error": "ae_error_history"})
    b = pdm[["unit_id", "absolute_error"]].rename(columns={"absolute_error": "ae_plus_dmis"})
    m = a.merge(b, on="unit_id", how="inner")
    m["paired_improvement"] = m["ae_error_history"] - m["ae_plus_dmis"]
    return m.groupby("model", as_index=False)["paired_improvement"].mean()


def write_markdown_table(df: pd.DataFrame, path: Path):
    """Write Markdown without requiring the optional `tabulate` package."""
    cols = [str(c) for c in df.columns]
    def fmt(v):
        if pd.isna(v): return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.6g}"
        return str(v)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(df: pd.DataFrame, path: Path):
    path.write_text(df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.6f}"), encoding="utf-8")


def plot_h1(summary: pd.DataFrame, path_png: Path, path_pdf: Path):
    labels = ["Model family\n(primary)", "Model", "Setting", "Intervention\ntype"]
    order = ["model_family", "model", "setting", "intervention_type"]
    vals = []
    rows = []
    for k in order:
        part = summary.loc[summary["split"] == k]
        if part.empty:
            rows.append(None); vals.append(np.nan)
        else:
            row = part.iloc[0]
            rows.append(row); vals.append(float(row.get("risk_reduction", np.nan)))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(order))
    ax.bar(x, vals)

    # Draw percentile intervals as absolute CI segments rather than yerr distances.
    # This remains valid even if a small-bootstrap diagnostic CI does not bracket
    # the point estimate, and avoids negative-yerr failures in reduced smoke runs.
    for xi, row, v in zip(x, rows, vals):
        if row is None or not np.isfinite(v):
            ax.text(xi, 0.0, "N/A", ha="center", va="bottom", fontsize=10)
            continue
        l = float(row.get("ci95_lower", np.nan)); h = float(row.get("ci95_upper", np.nan))
        if np.isfinite(l) and np.isfinite(h):
            low, high = sorted((l, h))
            ax.vlines(xi, low, high, linewidth=1.4, color="black")
            ax.hlines([low, high], xi - 0.07, xi + 0.07, linewidth=1.4, color="black")

    ax.axhline(0.0, linewidth=1.2, color="black")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Strong-RD Macro-MAE Risk Reduction")
    ax.set_title("H1: Incremental Value of Weak-Intervention DMIS")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Nested held-out H1 analysis for CausalGap Section 5.2.")
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    ap.add_argument("--units-csv", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="CPU-only model-identity cluster bootstrap replicates for the primary leave-one-family-out effect.")
    ap.add_argument("--bootstrap-seed", type=int, default=20270915)
    ap.add_argument("--lambda-grid", nargs="+", type=float, default=LAMBDA_GRID)
    args = ap.parse_args()

    analysis = args.experiment_root / "analysis" / "h1"
    inter = analysis / "intermediate"
    tables = analysis / "tables"
    figures = analysis / "figures"
    diagnostics = analysis / "diagnostics"
    for p in (inter, tables, figures, diagnostics): p.mkdir(parents=True, exist_ok=True)
    units_csv = args.units_csv or inter / "04_h1_prediction_units.csv"
    df = pd.read_csv(units_csv)

    required = set(CONTROLS_CONT + CONTROLS_BIN + CONTROLS_CAT + FEATURE_SETS["error_history_plus_response_history"] + ["unit_id", "model", "model_family", "intervention_type", "strong_RD"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Prediction-unit table missing H1 columns: {missing}")
    numeric_cols = list(set(CONTROLS_CONT + CONTROLS_BIN + FEATURE_SETS["error_history_plus_response_history"] + ["strong_RD"]))
    bad = df[numeric_cols].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    if bad.any():
        df.loc[bad].to_csv(diagnostics / "invalid_numeric_units.csv", index=False)
        raise RuntimeError(f"{int(bad.sum())} H1 units contain non-finite predictor/target values")

    split_order = ["model_family", "model", "setting", "intervention_type"]
    summary_rows = []
    fold_tables = []
    primary_pb = primary_pdm = None
    for split_kind in split_order:
        rec, fold, pb, pdm = comparison(df, split_kind, list(args.lambda_grid), group_col="model")
        summary_rows.append(rec)
        fold_tables.append(fold.assign(comparison="paired_H1"))
        if split_kind == "model_family":
            primary_pb, primary_pdm = pb, pdm
        if rec.get("available", True):
            print(
                f"[H1:{split_kind}] error_history={rec['risk_error_history']:.6f} "
                f"+DMIS={rec['risk_plus_dmis']:.6f} reduction={rec['risk_reduction']:.6f} "
                f"positive={rec['positive_folds']}/{rec['n_folds']}", flush=True,
            )
        else:
            print(f"[H1:{split_kind}] unavailable for this reduced protocol (fewer than two held-out groups)", flush=True)

    summary = pd.DataFrame(summary_rows)

    # Primary model-identity cluster bootstrap: within every family, sample model
    # identities with replacement and keep all units from each sampled identity.
    if int(args.bootstrap) > 0:
        boots = primary_bootstrap(df, int(args.bootstrap), int(args.bootstrap_seed), list(args.lambda_grid))
        np.save(inter / "05_primary_model_identity_bootstrap_reductions.npy", boots)
        pd.DataFrame({"replicate": np.arange(len(boots)), "risk_reduction": boots}).to_csv(
            inter / "05_primary_model_identity_bootstrap_reductions.csv", index=False
        )
        q95 = np.quantile(boots, [0.025, 0.975])
        q975 = np.quantile(boots, [0.0125, 0.9875])
        primary_mask = summary["split"].eq("model_family")
        summary.loc[primary_mask, "ci95_lower"] = float(q95[0])
        summary.loc[primary_mask, "ci95_upper"] = float(q95[1])
        summary.loc[primary_mask, "ci97_5_bonf_lower"] = float(q975[0])
        summary.loc[primary_mask, "ci97_5_bonf_upper"] = float(q975[1])
        summary.loc[primary_mask, "bootstrap_positive_fraction"] = float(np.mean(boots > 0))

    # Exact paired model-level sign-flip is a robustness statistic; it is not
    # substituted for the primary bootstrap interval.
    effects = model_level_effects(primary_pb, primary_pdm)
    p_sf, sf_mode, sf_n = exact_signflip_p(effects["paired_improvement"].to_numpy(), seed=args.bootstrap_seed)
    effects.to_csv(diagnostics / "primary_model_level_paired_improvements.csv", index=False)
    pm = summary["split"].eq("model_family")
    summary.loc[pm, "model_level_signflip_p"] = p_sf
    summary.loc[pm, "signflip_mode"] = sf_mode
    summary.loc[pm, "signflip_samples"] = sf_n
    summary.loc[pm, "positive_models"] = int((effects["paired_improvement"] > 0).sum())
    summary.loc[pm, "n_models"] = int(len(effects))

    # Supplementary nested-model decomposition and response-history sensitivity.
    nested_rows = []
    for key in ["mean_only_error", "error_history", "error_history_plus_dmis", "error_history_plus_response_history"]:
        risk, folds, preds = run_outer(df, "model_family", key, list(args.lambda_grid), group_col="model")
        nested_rows.append({"predictor": key, "macro_MAE": risk})
        folds.to_csv(diagnostics / f"folds_primary_{key}.csv", index=False)
        preds.to_csv(diagnostics / f"oof_primary_{key}.csv", index=False)
    nested = pd.DataFrame(nested_rows)
    nested["reduction_vs_preceding"] = nested["macro_MAE"].shift(1) - nested["macro_MAE"]
    nested.to_csv(tables / "Table_A6_H1_nested_model_audit.csv", index=False)

    nonempty_folds = [x for x in fold_tables if x is not None and len(x)]
    if nonempty_folds:
        pd.concat(nonempty_folds, ignore_index=True).to_csv(diagnostics / "h1_all_fold_risks.csv", index=False)
    else:
        pd.DataFrame().to_csv(diagnostics / "h1_all_fold_risks.csv", index=False)
    summary.to_csv(tables / "Table2_H1_incremental_diagnostic_value.csv", index=False)
    write_markdown_table(summary, tables / "Table2_H1_incremental_diagnostic_value.md")
    write_latex_table(summary, tables / "Table2_H1_incremental_diagnostic_value.tex")

    plot_h1(summary, figures / "Figure4_H1_only_risk_reduction.png", figures / "Figure4_H1_only_risk_reduction.pdf")

    primary = summary[summary["split"] == "model_family"].iloc[0].to_dict()
    json_dump(analysis / "H1_summary.json", {
        "primary": primary,
        "interpretation_rule": "Positive risk_reduction = lower held-out strong-RD macro-MAE after adding weak mean DMIScal to the same control + weak-error-history baseline.",
        "confirmatory_split": "leave-one-model-family-out",
        "robustness_splits": ["leave-one-model-out", "leave-one-setting-out", "leave-one-intervention-type-out"],
        "lambda_grid": list(map(float, args.lambda_grid)),
        "inner_cv": "GroupKFold by model identity; all continuous features standardized from inner/outer training data only",
        "identity_indicators": "model/family identities excluded from predictors",
        "bootstrap": "model identities resampled with replacement inside each family; complete H1 pipeline rerun",
        "n_prediction_units": int(len(df)),
    })

    print("\n===== Section 5.2 H1 complete =====")
    print(summary.to_string(index=False))
    print(f"Table: {tables / 'Table2_H1_incremental_diagnostic_value.csv'}")
    print(f"Figure (H1-only, for checking): {figures / 'Figure4_H1_only_risk_reduction.pdf'}")
    print(f"Units: {units_csv}")


if __name__ == "__main__":
    main()
