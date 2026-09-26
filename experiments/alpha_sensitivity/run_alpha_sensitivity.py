#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CausalGap Section 5.9 — DMIS mixing-weight alpha sensitivity.

Required close-out:
  alpha in {0.25, 0.50, 0.75}
  * Controlled Synthetic:
      - DMIS model-rank agreement vs alpha=0.50
      - H1 point-estimate direction using the frozen H1 outer-fold protocol
  * Abstract SUMO:
      - DMIS model-rank agreement vs alpha=0.50
      - operational-regime agreement vs alpha=0.50

No forecasting model is retrained or rerun. The script reconstructs DMIS from
saved cosine_change, magnitude_change, and structural-distance/calibration
columns, then reruns only the CPU-side H1 ridge analysis.

Visualization:
  Matplotlib default font family is kept intentionally. Font sizes are enlarged
  slightly for paper readability. Figures are saved as PDF/SVG/PNG.

Primary output:
  experiment_results/alpha_sensitivity/
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error


ALPHAS = (0.25, 0.50, 0.75)
REFERENCE_ALPHA = 0.50
EPS = 1e-6

LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

STRUCTURAL_TYPES = ("edge_removal", "weight_modification", "node_isolation")
STRONG_LEVEL = {
    "edge_removal": "s=0.20",
    "weight_modification": "gamma=0.2",
    "node_isolation": "s=0.60",
}

REGIMES = [
    "Stable Invariance",
    "Adaptive Response",
    "Pathological Rigidity",
    "Over-Reactive Instability",
]

MODEL_FAMILIES = {
    "STGCN": 1, "DCRNN": 1, "ASTGNN": 1, "GWNet": 1, "AGCRN": 1,
    "STEVE": 2, "DCGCN": 2, "ST-Norm": 2,
    "CCHMM": 3, "CausalGRIT": 3, "TF-MVGNN": 3,
    "PatchTST": 4, "iTransformer": 4, "TimesNet": 4, "FEDformer": 4,
    "DLinear": 4, "Autoformer": 4,
    "Chronos": 5, "TimesFM": 5, "Moirai": 5, "Lag-Llama": 5,
}

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
    "error_history": ["weak_RD_mean", "weak_RD_slope", "weak_RD_max"],
    "error_history_plus_dmis": [
        "weak_RD_mean", "weak_RD_slope", "weak_RD_max", "weak_DMIScal_mean"
    ],
}

# restrained scientific palette; font family is NOT specified
COLORS = {
    "a025": "#4E79A7",
    "a050": "#2A9D8F",
    "a075": "#E0A458",
    "synthetic": "#4E79A7",
    "sumo": "#2A9D8F",
    "grid": "#D9DEE5",
    "text": "#24324A",
    "zero": "#6B7280",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def finite_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def json_clean(obj):
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return json_clean(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_clean(obj), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def normalize_model(x: str) -> str:
    s = str(x).strip()
    aliases = {
        "Graph WaveNet": "GWNet",
        "GraphWaveNet": "GWNet",
        "CausalGRT": "CausalGRIT",
    }
    return aliases.get(s, s)


def alpha_tag(alpha: float) -> str:
    return f"a{int(round(alpha * 100)):03d}"


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2 or np.ptp(x) <= 1e-15:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


# ----------------------------------------------------------------------
# DMIS reconstruction
# ----------------------------------------------------------------------
def add_alpha_dmis(df: pd.DataFrame, distance_col: str, prefix: str) -> pd.DataFrame:
    required = {"cosine_change", "magnitude_change", distance_col}
    miss = sorted(required - set(df.columns))
    if miss:
        raise RuntimeError(
            f"Cannot recompute alpha sensitivity: missing {miss}. "
            "The experiment requires saved cosine_change and magnitude_change; "
            "do not infer alpha sensitivity from DMIS(alpha=.5) alone."
        )

    out = df.copy()
    cd = pd.to_numeric(out["cosine_change"], errors="coerce").to_numpy(float)
    mg = pd.to_numeric(out["magnitude_change"], errors="coerce").to_numpy(float)
    dist = pd.to_numeric(out[distance_col], errors="coerce").to_numpy(float)

    if not (np.isfinite(cd).all() and np.isfinite(mg).all() and np.isfinite(dist).all()):
        raise RuntimeError(f"Non-finite cosine/magnitude/{distance_col} values in alpha source.")

    denom = np.maximum(dist, EPS)
    for a in ALPHAS:
        tag = alpha_tag(a)
        dz = a * cd + (1.0 - a) * mg
        out[f"{prefix}_DZ_{tag}"] = dz
        out[f"{prefix}_DMIS_{tag}"] = dz / denom
    return out


def audit_alpha050_reconstruction(
    df: pd.DataFrame,
    reconstructed_col: str,
    reference_candidates: Iterable[str],
    label: str,
    fail_tol: float = 1e-6,
) -> dict:
    ref = next((c for c in reference_candidates if c in df.columns), None)
    if ref is None:
        return {
            "label": label,
            "reference_column": None,
            "status": "reference_not_available",
        }
    x = pd.to_numeric(df[reconstructed_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[ref], errors="coerce").to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    if not good.any():
        raise RuntimeError(f"{label}: no finite alpha=.5 reconstruction pairs.")
    diff = np.abs(x[good] - y[good])
    scale = np.maximum(np.abs(y[good]), 1e-12)
    audit = {
        "label": label,
        "reference_column": ref,
        "n": int(good.sum()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_relative_diff": float((diff / scale).max()),
    }
    if audit["max_abs_diff"] > fail_tol and audit["max_relative_diff"] > fail_tol:
        raise RuntimeError(
            f"{label}: alpha=.50 reconstruction does not reproduce {ref}. "
            f"max_abs={audit['max_abs_diff']:.6g}, "
            f"max_rel={audit['max_relative_diff']:.6g}. "
            "Stop rather than mix incompatible DMIS definitions."
        )
    audit["status"] = "pass"
    return audit


# ----------------------------------------------------------------------
# Synthetic frozen H1 source
# ----------------------------------------------------------------------
def discover_h1_analysis_root(project: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit

    # Prefer corrected V2 analysis if it contains the necessary frozen files.
    candidates = [
        project / "experiment_results/incremental_value",
        project / "experiment_results/incremental_value",
    ]
    for root in candidates:
        raw = root / "analysis/h1/intermediate/01_h1_raw_calibrated_rows.csv"
        units = root / "analysis/h1/intermediate/04_h1_prediction_units.csv"
        if raw.is_file() and units.is_file():
            return root

    # V2 may reuse V1 raw source while only corrected units live in V2.
    v2 = candidates[0]
    v1 = candidates[1]
    if (v2 / "analysis/h1/intermediate/04_h1_prediction_units.csv").is_file() and \
       (v1 / "analysis/h1/intermediate/01_h1_raw_calibrated_rows.csv").is_file():
        return v2

    raise FileNotFoundError(
        "Cannot find frozen H1 analysis. Expected V2 or V1 analysis/h1/intermediate "
        "with 01_h1_raw_calibrated_rows.csv and 04_h1_prediction_units.csv."
    )


def resolve_h1_files(project: Path, root: Path) -> tuple[Path, Path]:
    units = root / "analysis/h1/intermediate/04_h1_prediction_units.csv"
    raw = root / "analysis/h1/intermediate/01_h1_raw_calibrated_rows.csv"

    if not units.is_file():
        raise FileNotFoundError(units)
    if raw.is_file():
        return raw, units

    # Corrected V2 commonly reuses frozen V1 raw results.
    fallback = (
        project
        / "experiment_results/incremental_value"
        / "analysis/h1/intermediate/01_h1_raw_calibrated_rows.csv"
    )
    if fallback.is_file():
        return fallback, units
    raise FileNotFoundError(raw)


def load_synthetic_h1(project: Path, h1_root: Path | None):
    root = discover_h1_analysis_root(project, h1_root)
    raw_path, units_path = resolve_h1_files(project, root)

    raw = pd.read_csv(raw_path, low_memory=False)
    units = pd.read_csv(units_path, low_memory=False)

    required_raw = {
        "model", "evaluation_track", "intervention_type", "paper_intensity_level",
        "RD", "cosine_change", "magnitude_change", "distance_calibrated",
        "h1_phase", "weak_intensity_s",
    }
    miss = sorted(required_raw - set(raw.columns))
    if miss:
        raise RuntimeError(
            f"Frozen H1 raw snapshot is missing {miss}. File: {raw_path}. "
            "Use the corrected H1 snapshot that retained evaluator response components."
        )

    raw["model"] = raw["model"].map(normalize_model)
    raw = raw[raw["evaluation_track"].astype(str).str.upper().eq("B")].copy()
    raw = raw[raw["intervention_type"].astype(str).isin(STRUCTURAL_TYPES)].copy()
    if "causal_valid" in raw.columns:
        raw = raw[is_true(raw["causal_valid"])].copy()
    if "leakage_detected" in raw.columns:
        raw = raw[~is_true(raw["leakage_detected"])].copy()

    # Keep the frozen synthetic setting only.
    if "setting" in raw.columns:
        syn_mask = raw["setting"].astype(str).str.lower().str.contains("synthetic")
        if syn_mask.any():
            raw = raw[syn_mask].copy()

    # H1 controls/targets are frozen in the original prediction-unit table.
    units["model"] = units["model"].map(normalize_model)
    if "setting" in units.columns:
        syn_mask = units["setting"].astype(str).str.lower().str.contains("synthetic")
        if syn_mask.any():
            units = units[syn_mask].copy()

    if len(units) != 63 or units["model"].nunique() != 21:
        log(
            f"[H1 WARNING] frozen units rows={len(units)}, models={units['model'].nunique()} "
            "(expected 63 and 21 for 21 models × 3 intervention families)."
        )

    raw = add_alpha_dmis(raw, "distance_calibrated", "cal")

    audit = audit_alpha050_reconstruction(
        raw,
        "cal_DMIS_a050",
        ["DMIS_cal"],
        "Synthetic calibrated DMIS",
        fail_tol=1e-6,
    )

    return root, raw_path, units_path, raw, units, audit


def build_alpha_units(raw: pd.DataFrame, base_units: pd.DataFrame, alpha: float) -> pd.DataFrame:
    tag = alpha_tag(alpha)
    dmis_col = f"cal_DMIS_{tag}"

    group_cols = [
        "setting", "model", "intervention_type", "paper_intensity_level",
        "h1_phase",
    ]
    if "declared_strength" in raw.columns:
        group_cols.append("declared_strength")

    numeric = [dmis_col, "RD", "weak_intensity_s"]
    g = raw.groupby(group_cols, dropna=False)[numeric].mean().reset_index()

    rows = []
    for (setting, model, kind), q in g.groupby(
        ["setting", "model", "intervention_type"], dropna=False
    ):
        weak = q[q["h1_phase"].astype(str).eq("weak")].copy()
        if len(weak) != 3:
            raise RuntimeError(
                f"Alpha={alpha}: {model}/{kind} has {len(weak)} weak configurations; expected 3."
            )

        # Preserve original weak-intensity ordering.
        weak = weak.sort_values("weak_intensity_s")
        x = pd.to_numeric(weak["weak_intensity_s"], errors="coerce").to_numpy(float)
        d = pd.to_numeric(weak[dmis_col], errors="coerce").to_numpy(float)
        if not (np.isfinite(x).all() and np.isfinite(d).all()):
            raise RuntimeError(f"Alpha={alpha}: non-finite H1 weak DMIS for {model}/{kind}.")

        rows.append({
            "setting": setting,
            "model": model,
            "intervention_type": kind,
            "weak_DMIScal_mean_new": float(np.mean(d)),
            "weak_DMIScal_slope_new": slope(x, d),
            "weak_DMIScal_max_new": float(np.max(d)),
        })

    f = pd.DataFrame(rows)
    key = ["setting", "model", "intervention_type"]
    out = base_units.merge(f, on=key, how="left", validate="one_to_one")

    if out["weak_DMIScal_mean_new"].isna().any():
        bad = out.loc[out["weak_DMIScal_mean_new"].isna(), key].to_dict("records")
        raise RuntimeError(f"Could not reconstruct alpha={alpha} DMIS for H1 units: {bad[:10]}")

    for c in ["mean", "slope", "max"]:
        out[f"weak_DMIScal_{c}"] = out[f"weak_DMIScal_{c}_new"]
        out.drop(columns=[f"weak_DMIScal_{c}_new"], inplace=True)
    out["alpha"] = alpha
    return out


def audit_h1_alpha050_units(base: pd.DataFrame, rebuilt: pd.DataFrame) -> dict:
    key = ["setting", "model", "intervention_type"]
    cols = ["weak_DMIScal_mean", "weak_DMIScal_slope", "weak_DMIScal_max"]
    a = base[key + cols].copy()
    b = rebuilt[key + cols].copy()
    m = a.merge(b, on=key, suffixes=("_frozen", "_reconstructed"), validate="one_to_one")
    out = {}
    for c in cols:
        x = pd.to_numeric(m[f"{c}_frozen"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(m[f"{c}_reconstructed"], errors="coerce").to_numpy(float)
        diff = np.abs(x - y)
        out[c] = float(np.nanmax(diff))
    maxdiff = max(out.values())
    out["status"] = "pass" if maxdiff <= 1e-6 else "fail"
    if out["status"] != "pass":
        raise RuntimeError(
            "Reconstructed alpha=.50 H1 DMIS features do not reproduce frozen H1 units: "
            + json.dumps(out)
        )
    return out


# ----------------------------------------------------------------------
# Frozen H1 ridge protocol (same model-family-out point estimate)
# ----------------------------------------------------------------------
def fit_design(train: pd.DataFrame, test: pd.DataFrame, extra_features: list[str]):
    required = set(CONTROLS_CONT + CONTROLS_BIN + CONTROLS_CAT + extra_features)
    miss = sorted(required - set(train.columns))
    if miss:
        raise RuntimeError(f"H1 units missing frozen analysis columns: {miss}")

    cont = CONTROLS_CONT + [
        c for c in extra_features if c not in CONTROLS_BIN and c not in CONTROLS_CAT
    ]
    means = train[cont].astype(float).mean(axis=0)
    stds = train[cont].astype(float).std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)

    Xtr_parts = [((train[cont].astype(float) - means) / stds).to_numpy()]
    Xte_parts = [((test[cont].astype(float) - means) / stds).to_numpy()]

    Xtr_parts.append(train[CONTROLS_BIN].astype(float).to_numpy())
    Xte_parts.append(test[CONTROLS_BIN].astype(float).to_numpy())

    for cat in CONTROLS_CAT:
        levels = sorted(train[cat].astype(str).dropna().unique().tolist())
        for lv in levels:
            Xtr_parts.append(train[cat].astype(str).eq(lv).astype(float).to_numpy()[:, None])
            Xte_parts.append(test[cat].astype(str).eq(lv).astype(float).to_numpy()[:, None])

    return (
        np.concatenate(Xtr_parts, axis=1).astype(float),
        np.concatenate(Xte_parts, axis=1).astype(float),
    )


def ridge_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, lam: float):
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


def inner_lambda(train: pd.DataFrame, feature_key: str, group_col: str = "model") -> float:
    groups = train[group_col].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 2:
        return float(LAMBDA_GRID[0])

    n_splits = min(5, len(unique))
    cv = GroupKFold(n_splits=n_splits)
    y = train["strong_RD"].to_numpy(float)
    dummy = np.zeros((len(train), 1))

    cached = []
    for tr_idx, va_idx in cv.split(dummy, y, groups):
        tr = train.iloc[tr_idx]
        va = train.iloc[va_idx]
        Xtr, Xva = fit_design(tr, va, FEATURE_SETS[feature_key])
        cached.append(
            (Xtr, tr["strong_RD"].to_numpy(float), Xva, va["strong_RD"].to_numpy(float))
        )

    best_lam, best_score = None, float("inf")
    for lam in LAMBDA_GRID:
        vals = []
        for Xtr, ytr, Xva, yva in cached:
            pred = ridge_predict(Xtr, ytr, Xva, lam)
            vals.append(float(np.mean(np.abs(yva - pred))))
        score = float(np.mean(vals))
        if (
            score < best_score - 1e-12
            or (abs(score - best_score) <= 1e-12 and (best_lam is None or lam < best_lam))
        ):
            best_lam, best_score = float(lam), score
    return float(best_lam)


def run_outer_model_family(df: pd.DataFrame, feature_key: str):
    folds, preds = [], []
    families = sorted(pd.to_numeric(df["model_family"], errors="raise").astype(int).unique())
    for fam in families:
        mask = pd.to_numeric(df["model_family"], errors="raise").astype(int).eq(int(fam))
        test = df[mask].copy()
        train = df[~mask].copy()
        lam = inner_lambda(train, feature_key, "model")
        Xtr, Xte = fit_design(train, test, FEATURE_SETS[feature_key])
        pred = ridge_predict(Xtr, train["strong_RD"].to_numpy(float), Xte, lam)
        truth = test["strong_RD"].to_numpy(float)
        mae = float(mean_absolute_error(truth, pred))
        folds.append({
            "held_out_family": int(fam),
            "predictor": feature_key,
            "macro_fold_mae": mae,
            "lambda": lam,
            "n_train": len(train),
            "n_test": len(test),
        })
        for j, (_, r) in enumerate(test.iterrows()):
            preds.append({
                "unit_id": r.get("unit_id", f"{r['model']}__{r['intervention_type']}"),
                "model": r["model"],
                "model_family": int(fam),
                "intervention_type": r["intervention_type"],
                "predictor": feature_key,
                "truth": float(truth[j]),
                "prediction": float(pred[j]),
                "absolute_error": float(abs(truth[j] - pred[j])),
            })
    f = pd.DataFrame(folds)
    p = pd.DataFrame(preds)
    return float(f["macro_fold_mae"].mean()), f, p


def h1_point_estimate(units: pd.DataFrame, alpha: float):
    r0, f0, p0 = run_outer_model_family(units, "error_history")
    r1, f1, p1 = run_outer_model_family(units, "error_history_plus_dmis")
    m = f0[["held_out_family", "macro_fold_mae"]].rename(
        columns={"macro_fold_mae": "risk_error_history"}
    ).merge(
        f1[["held_out_family", "macro_fold_mae"]].rename(
            columns={"macro_fold_mae": "risk_plus_dmis"}
        ),
        on="held_out_family",
        validate="one_to_one",
    )
    m["risk_reduction"] = m["risk_error_history"] - m["risk_plus_dmis"]
    m["alpha"] = alpha
    return {
        "alpha": alpha,
        "risk_error_history": r0,
        "risk_plus_dmis": r1,
        "risk_reduction": float(r0 - r1),
        "direction": "Positive" if r0 - r1 > 0 else ("Negative" if r0 - r1 < 0 else "Zero"),
        "positive_folds": int((m["risk_reduction"] > 0).sum()),
        "n_folds": int(len(m)),
    }, m, p0, p1


# ----------------------------------------------------------------------
# Model-ranking summaries
# ----------------------------------------------------------------------
def rank_summary(df: pd.DataFrame, prefix: str, setting: str):
    rows = []
    for a in ALPHAS:
        col = f"{prefix}_DMIS_{alpha_tag(a)}"
        m = df.groupby("model", as_index=False)[col].mean().rename(columns={col: "DMIS"})
        m["alpha"] = a
        m["setting"] = setting
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def compare_ranks(model_alpha: pd.DataFrame, setting: str):
    ref = (
        model_alpha[np.isclose(model_alpha["alpha"], REFERENCE_ALPHA)]
        [["model", "DMIS"]]
        .rename(columns={"DMIS": "DMIS_ref"})
    )
    rows = []
    pair_rows = []
    for a in ALPHAS:
        cur = (
            model_alpha[np.isclose(model_alpha["alpha"], a)]
            [["model", "DMIS"]]
            .rename(columns={"DMIS": "DMIS_alpha"})
        )
        m = ref.merge(cur, on="model", validate="one_to_one")
        sp = spearmanr(m["DMIS_ref"], m["DMIS_alpha"])
        kt = kendalltau(m["DMIS_ref"], m["DMIS_alpha"])
        rows.append({
            "setting": setting,
            "alpha": a,
            "n_models": len(m),
            "spearman_rank_agreement": float(sp.statistic),
            "kendall_tau_b": float(kt.statistic),
        })
        m["alpha"] = a
        m["setting"] = setting
        m["rank_ref"] = m["DMIS_ref"].rank(ascending=False, method="average")
        m["rank_alpha"] = m["DMIS_alpha"].rank(ascending=False, method="average")
        m["rank_shift"] = m["rank_alpha"] - m["rank_ref"]
        pair_rows.append(m)
    return pd.DataFrame(rows), pd.concat(pair_rows, ignore_index=True)


# ----------------------------------------------------------------------
# SUMO raw evaluator rows
# ----------------------------------------------------------------------
def load_sumo_raw(project: Path, explicit: Path | None):
    root = explicit or project / "experiment_results/sumo_replication"
    if not root.exists():
        raise FileNotFoundError(root)

    files = []
    # Prefer the declared current setting path.
    declared = root / "runs/sumo_metrla"
    if declared.exists():
        files = sorted(declared.glob("*/trackB/seed*/causalgap_results.csv"))
    if not files:
        for p in sorted(root.rglob("causalgap_results.csv")):
            s = p.as_posix().lower()
            if "trackb" in s and ("sumo_metrla" in s or "/runs/" in s):
                files.append(p)

    if not files:
        raise FileNotFoundError(
            f"No SUMO Track-B causalgap_results.csv found under {root}. "
            "Alpha sensitivity requires raw saved cosine/magnitude components."
        )

    chunks = []
    for p in files:
        d = pd.read_csv(p, low_memory=False)
        if d.empty:
            continue
        if "evaluation_track" in d.columns:
            d = d[d["evaluation_track"].astype(str).str.upper().eq("B")].copy()
        if "causal_valid" in d.columns:
            d = d[is_true(d["causal_valid"])].copy()
        if "leakage_detected" in d.columns:
            d = d[~is_true(d["leakage_detected"])].copy()
        if "intervention_type" in d.columns:
            d = d[d["intervention_type"].astype(str).isin(STRUCTURAL_TYPES)].copy()

        if "model" not in d.columns or d["model"].isna().all():
            # .../runs/sumo_metrla/MODEL/trackB/seed0/...
            parts = p.parts
            try:
                i = parts.index("sumo_metrla")
                model = parts[i + 1]
            except Exception:
                model = p.parent.parent.parent.name
            d["model"] = model
        d["model"] = d["model"].map(normalize_model)
        d["_source_file"] = str(p)
        chunks.append(d)

    if not chunks:
        raise RuntimeError("SUMO result files exist but no valid structural Track-B rows remain.")

    raw = pd.concat(chunks, ignore_index=True)

    # canonical model seed 0 when the result contains multiple checkpoints
    if "model_seed" in raw.columns:
        ms = pd.to_numeric(raw["model_seed"], errors="coerce")
        if (ms == 0).any():
            raw = raw[ms.eq(0)].copy()

    required = {
        "model", "RD", "cosine_change", "magnitude_change", "environment_distance",
    }
    miss = sorted(required - set(raw.columns))
    if miss:
        raise RuntimeError(f"SUMO raw results missing {miss}.")

    # Detect duplicate evaluator keys rather than silently average historical reruns.
    key = ["model", "intervention_type", "paper_intensity_level", "intervention_seed"]
    if all(c in raw.columns for c in key):
        dup = raw.duplicated(key, keep=False)
        if dup.any():
            preview = raw.loc[dup, key + ["_source_file"]].head(20).to_dict("records")
            raise RuntimeError(
                "SUMO alpha source contains duplicate canonical keys. "
                "Do not silently average historical reruns. Preview: " + json.dumps(preview)
            )

    raw = add_alpha_dmis(raw, "environment_distance", "raw")

    audit = audit_alpha050_reconstruction(
        raw, "raw_DMIS_a050", ["DMIS"], "SUMO raw DMIS", fail_tol=1e-6
    )
    return root, raw, audit


def assign_regime(dmis: pd.Series, rd: pd.Series):
    d = pd.to_numeric(dmis, errors="coerce").to_numpy(float)
    r = pd.to_numeric(rd, errors="coerce").to_numpy(float)
    if not (np.isfinite(d).all() and np.isfinite(r).all()):
        raise RuntimeError("Non-finite values in SUMO regime assignment.")
    dt = float(np.median(d))
    rt = float(np.median(r))
    hi_d = d >= dt
    hi_r = r >= rt
    labels = np.select(
        [
            (~hi_d) & (~hi_r),
            hi_d & (~hi_r),
            (~hi_d) & hi_r,
            hi_d & hi_r,
        ],
        REGIMES,
        default="INVALID",
    )
    return labels.astype(str), dt, rt


def sumo_regime_sensitivity(sumo_model_alpha: pd.DataFrame, project: Path):
    wide = sumo_model_alpha.pivot(index="model", columns="alpha", values="DMIS").reset_index()

    # RD does not depend on alpha.
    # Use all SUMO evaluator instances exactly as primary replication did.
    # This is attached outside after model aggregation.
    return wide


# ----------------------------------------------------------------------
# Output and figures
# ----------------------------------------------------------------------
def write_table25_tex(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Sensitivity to the DMIS mixing weight $\alpha$.}",
        r"\label{tab:alpha_sensitivity}",
        r"\resizebox{0.82\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"$\alpha$ & Synthetic Rank Agreement & H1 Risk Reduction & H1 Direction & SUMO Rank Agreement & SUMO Regime Agreement \\",
        r"\midrule",
    ]
    for r in df.itertuples(index=False):
        if np.isclose(r.alpha, REFERENCE_ALPHA):
            syn = "1.000"
            hdir = "Reference"
            sr = "1.000"
            rg = "100.0\\%"
        else:
            syn = f"{r.synthetic_rank_agreement:.3f}"
            hdir = str(r.synthetic_h1_direction_vs_reference)
            sr = f"{r.sumo_rank_agreement:.3f}"
            rg = f"{100.0*r.sumo_regime_agreement:.1f}\\%"
        lines.append(
            f"{r.alpha:.2f} & {syn} & {r.synthetic_h1_risk_reduction:.6f} & "
            f"{hdir} & {sr} & {rg} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def configure_plotting():
    # Keep Matplotlib default font family. Only enlarge sizes slightly.
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11.5,
        "figure.titlesize": 16,
    })


def style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])


def savefig(fig, stem: Path):
    for ext in ("pdf", "svg", "png"):
        p = stem.with_suffix("." + ext)
        fig.savefig(p, bbox_inches="tight", dpi=350, facecolor="white")
        log(f"[SAVED] {p}")
    plt.close(fig)


def plot_rank_agreement(table: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    x = np.arange(len(table))
    w = 0.34
    ax.bar(
        x - w/2, table["synthetic_rank_agreement"], width=w,
        label="Controlled Synthetic", color=COLORS["synthetic"], alpha=0.88
    )
    ax.bar(
        x + w/2, table["sumo_rank_agreement"], width=w,
        label="Abstract SUMO", color=COLORS["sumo"], alpha=0.88
    )
    ax.set_xticks(x, [f"{a:.2f}" for a in table["alpha"]])
    ax.set_xlabel(r"Mixing Weight $\alpha$")
    ax.set_ylabel("Spearman Rank Agreement")
    ax.set_ylim(0, 1.06)
    ax.set_title(r"DMIS Ranking Stability Across $\alpha$")
    ax.legend(frameon=False, ncol=2, loc="lower center")
    style_ax(ax)
    savefig(fig, out / "Fig59_rank_agreement")


def plot_h1(table: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    ax.plot(
        table["alpha"], table["synthetic_h1_risk_reduction"],
        marker="o", linewidth=2.2, markersize=7.5, color=COLORS["synthetic"]
    )
    ax.axhline(0, linestyle="--", linewidth=1.1, color=COLORS["zero"])
    ax.set_xticks(list(ALPHAS))
    ax.set_xlabel(r"Mixing Weight $\alpha$")
    ax.set_ylabel("H1 Risk Reduction")
    ax.set_title(r"Synthetic H1 Sensitivity to $\alpha$")
    style_ax(ax)
    savefig(fig, out / "Fig59_H1_risk_reduction")


def plot_rank_trajectories(pair_df: pd.DataFrame, setting: str, out: Path, stem: str):
    # Convert pair table to model × alpha actual ranks.
    cur = pair_df.copy()
    if cur.empty:
        return
    # pair_df includes rank_alpha for each alpha, including reference
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    models = sorted(cur["model"].unique())
    cmap = plt.get_cmap("tab20")
    for i, m in enumerate(models):
        g = cur[cur["model"].eq(m)].sort_values("alpha")
        ax.plot(
            g["alpha"], g["rank_alpha"],
            marker="o", markersize=4.0, linewidth=1.2, alpha=0.78,
            color=cmap(i % 20)
        )
        # Label at alpha=.75 only, lightly.
        rr = g.iloc[-1]
        ax.text(float(rr["alpha"]) + 0.006, float(rr["rank_alpha"]), m, fontsize=8.2, va="center")
    ax.invert_yaxis()
    ax.set_xlim(0.235, 0.84)
    ax.set_xticks(list(ALPHAS))
    ax.set_xlabel(r"Mixing Weight $\alpha$")
    ax.set_ylabel("DMIS Rank (1 = Highest)")
    ax.set_title(f"{setting} Model-Rank Trajectories")
    style_ax(ax, grid_axis="both")
    savefig(fig, out / stem)


def plot_sumo_regime_agreement(table: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    vals = 100.0 * table["sumo_regime_agreement"].to_numpy(float)
    colors = [COLORS["a025"], COLORS["a050"], COLORS["a075"]]
    ax.bar([f"{a:.2f}" for a in table["alpha"]], vals, color=colors, alpha=0.9, width=0.62)
    ax.set_xlabel(r"Mixing Weight $\alpha$")
    ax.set_ylabel("Regime Agreement (%)")
    ax.set_ylim(0, 105)
    ax.set_title(r"Abstract SUMO Regime Stability Across $\alpha$")
    style_ax(ax)
    for i, v in enumerate(vals):
        ax.text(i, min(v + 1.2, 103), f"{v:.1f}%", ha="center", va="bottom", fontsize=11.5)
    savefig(fig, out / "Fig59_SUMO_regime_agreement")


def plot_regime_confusion(pairs: pd.DataFrame, alpha: float, out: Path):
    if np.isclose(alpha, REFERENCE_ALPHA):
        return
    g = pairs[np.isclose(pairs["alpha"], alpha)].copy()
    mat = pd.crosstab(g["regime_ref"], g["regime_alpha"]).reindex(
        index=REGIMES, columns=REGIMES, fill_value=0
    )
    arr = mat.to_numpy(int)
    fig, ax = plt.subplots(figsize=(7.3, 6.1))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(4), ["Stable", "Adaptive", "Rigidity", "Over-Reactive"], rotation=25, ha="right")
    ax.set_yticks(range(4), ["Stable", "Adaptive", "Rigidity", "Over-Reactive"])
    ax.set_xlabel(f"Regime at $\\alpha={alpha:.2f}$")
    ax.set_ylabel(r"Reference Regime at $\alpha=0.50$")
    ax.set_title(f"Abstract SUMO Regime Transitions ($\\alpha={alpha:.2f}$)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center",
                    color="white" if arr[i, j] > max(arr.max() * 0.55, 1) else COLORS["text"],
                    fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Number of Models")
    savefig(fig, out / f"Fig59_SUMO_regime_transition_{alpha_tag(alpha)}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path("/data/wanganna/CausalGap"))
    ap.add_argument("--h1-root", type=Path, default=None,
                    help="Optional frozen H1 analysis root. V2 is preferred automatically.")
    ap.add_argument("--sumo-root", type=Path, default=None,
                    help="Optional Abstract SUMO experiment root.")
    ap.add_argument("--sumo-regime-csv", type=Path, default=None,
                    help="Optional saved alpha=.50 SUMO regime assignment CSV for audit.")
    ap.add_argument("--output-root", type=Path, default=None)
    args = ap.parse_args()

    project = args.project_root
    out = args.output_root or project / "experiment_results/alpha_sensitivity"
    inter = out / "intermediate"
    tables = out / "tables"
    figs = out / "figures"
    diag = out / "diagnostics"
    for p in (inter, tables, figs, diag):
        p.mkdir(parents=True, exist_ok=True)

    configure_plotting()

    log("================ CausalGap Section 5.9: Alpha Sensitivity ================")
    log("[INFO] No model training or inference is performed.")

    # ---------------- Synthetic ----------------
    h1_root, raw_path, units_path, syn_raw, base_units, syn_recon_audit = load_synthetic_h1(
        project, args.h1_root
    )
    log(f"[SYNTHETIC] H1 root: {h1_root}")
    log(f"[SYNTHETIC] raw rows={len(syn_raw)} models={syn_raw['model'].nunique()}")
    log(f"[SYNTHETIC] units={len(base_units)} models={base_units['model'].nunique()}")

    # Model-level ranking summary from all canonical structural evaluator rows.
    syn_models = rank_summary(syn_raw, "cal", "Controlled Synthetic")
    syn_rank, syn_rank_pairs = compare_ranks(syn_models, "Controlled Synthetic")

    h1_records = []
    h1_folds = []
    alpha_units = []
    alpha050_unit_audit = None
    for a in ALPHAS:
        u = build_alpha_units(syn_raw, base_units, a)
        if np.isclose(a, REFERENCE_ALPHA):
            alpha050_unit_audit = audit_h1_alpha050_units(base_units, u)
        rec, folds, _, _ = h1_point_estimate(u, a)
        h1_records.append(rec)
        h1_folds.append(folds)
        alpha_units.append(u)
        log(
            f"[H1] alpha={a:.2f} risk_reduction={rec['risk_reduction']:.8g} "
            f"direction={rec['direction']} positive_folds={rec['positive_folds']}/{rec['n_folds']}"
        )

    h1_df = pd.DataFrame(h1_records)
    h1_fold_df = pd.concat(h1_folds, ignore_index=True)
    alpha_units_df = pd.concat(alpha_units, ignore_index=True)

    ref_h1_direction = h1_df.loc[
        np.isclose(h1_df["alpha"], REFERENCE_ALPHA), "direction"
    ].iloc[0]
    h1_df["direction_vs_alpha050"] = np.where(
        np.isclose(h1_df["alpha"], REFERENCE_ALPHA),
        "Reference",
        np.where(h1_df["direction"].eq(ref_h1_direction), "Same", "Different"),
    )

    # ---------------- SUMO ----------------
    sumo_root, sumo_raw, sumo_recon_audit = load_sumo_raw(project, args.sumo_root)
    log(f"[SUMO] root: {sumo_root}")
    log(f"[SUMO] raw rows={len(sumo_raw)} models={sumo_raw['model'].nunique()}")

    sumo_models = rank_summary(sumo_raw, "raw", "Abstract SUMO")
    sumo_rank, sumo_rank_pairs = compare_ranks(sumo_models, "Abstract SUMO")

    # RD model means are alpha-independent.
    rd_model = (
        sumo_raw.groupby("model", as_index=False)["RD"].mean()
        .rename(columns={"RD": "RD_model_mean"})
    )
    sumo_models = sumo_models.merge(rd_model, on="model", validate="many_to_one")

    regime_rows = []
    regime_pair_rows = []
    reference_labels = None
    reference_thresholds = None
    for a in ALPHAS:
        g = sumo_models[np.isclose(sumo_models["alpha"], a)].copy().sort_values("model")
        labels, dthr, rthr = assign_regime(g["DMIS"], g["RD_model_mean"])
        g["regime"] = labels
        g["dmis_threshold"] = dthr
        g["rd_threshold"] = rthr
        if np.isclose(a, REFERENCE_ALPHA):
            reference_labels = dict(zip(g["model"], g["regime"]))
            reference_thresholds = {"dmis": dthr, "rd": rthr}
        regime_rows.append(g)

    sumo_regimes = pd.concat(regime_rows, ignore_index=True)

    for a in ALPHAS:
        g = sumo_regimes[np.isclose(sumo_regimes["alpha"], a)].copy()
        g["regime_ref"] = g["model"].map(reference_labels)
        g["regime_alpha"] = g["regime"]
        g["agreement"] = g["regime_ref"].eq(g["regime_alpha"])
        regime_pair_rows.append(g[[
            "model", "alpha", "DMIS", "RD_model_mean",
            "regime_ref", "regime_alpha", "agreement",
        ]])
    regime_pairs = pd.concat(regime_pair_rows, ignore_index=True)

    sumo_regime_summary = (
        regime_pairs.groupby("alpha", as_index=False)["agreement"].mean()
        .rename(columns={"agreement": "sumo_regime_agreement"})
    )

    # Audit against saved primary alpha=.50 regime labels when available.
    saved_regime = args.sumo_regime_csv
    if saved_regime is None:
        candidate = (
            project / "unified/regime_stability_results/Abstract_SUMO_regime_assignments.csv"
        )
        if candidate.is_file():
            saved_regime = candidate

    sumo_saved_audit = {"status": "saved_primary_not_available"}
    if saved_regime is not None and saved_regime.is_file():
        saved = pd.read_csv(saved_regime)
        if {"model", "regime_q050"}.issubset(saved.columns):
            ref = sumo_regimes[np.isclose(sumo_regimes["alpha"], REFERENCE_ALPHA)][
                ["model", "regime"]
            ].copy()
            s = saved[["model", "regime_q050"]].copy()
            s["model"] = s["model"].map(normalize_model)
            mm = ref.merge(s, on="model", validate="one_to_one")
            agreement = float(mm["regime"].eq(mm["regime_q050"]).mean())
            sumo_saved_audit = {
                "status": "pass" if agreement == 1.0 else "mismatch",
                "source": str(saved_regime),
                "n_models": len(mm),
                "saved_label_reproduction": agreement,
            }
            if agreement < 1.0:
                raise RuntimeError(
                    "Alpha=.50 SUMO regime reconstruction does not reproduce saved "
                    f"Section 5.4 labels (agreement={agreement:.6f}). "
                    "Stop rather than redefine the primary regime protocol."
                )

    # ---------------- Final table ----------------
    final = (
        pd.DataFrame({"alpha": ALPHAS})
        .merge(
            syn_rank[["alpha", "spearman_rank_agreement"]].rename(
                columns={"spearman_rank_agreement": "synthetic_rank_agreement"}
            ),
            on="alpha", validate="one_to_one",
        )
        .merge(
            h1_df[["alpha", "risk_reduction", "direction", "direction_vs_alpha050"]].rename(
                columns={
                    "risk_reduction": "synthetic_h1_risk_reduction",
                    "direction": "synthetic_h1_direction",
                    "direction_vs_alpha050": "synthetic_h1_direction_vs_reference",
                }
            ),
            on="alpha", validate="one_to_one",
        )
        .merge(
            sumo_rank[["alpha", "spearman_rank_agreement"]].rename(
                columns={"spearman_rank_agreement": "sumo_rank_agreement"}
            ),
            on="alpha", validate="one_to_one",
        )
        .merge(sumo_regime_summary, on="alpha", validate="one_to_one")
    )

    # Reference is by definition 1 for ranking/regime agreement.
    final.loc[np.isclose(final["alpha"], REFERENCE_ALPHA), "synthetic_rank_agreement"] = 1.0
    final.loc[np.isclose(final["alpha"], REFERENCE_ALPHA), "sumo_rank_agreement"] = 1.0
    final.loc[np.isclose(final["alpha"], REFERENCE_ALPHA), "sumo_regime_agreement"] = 1.0

    # ---------------- Save ----------------
    syn_raw.to_csv(inter / "01_synthetic_rows_alpha.csv", index=False)
    syn_models.to_csv(inter / "02_synthetic_model_dmis_by_alpha.csv", index=False)
    syn_rank.to_csv(inter / "03_synthetic_rank_agreement.csv", index=False)
    syn_rank_pairs.to_csv(inter / "04_synthetic_model_rank_pairs.csv", index=False)
    alpha_units_df.to_csv(inter / "05_synthetic_h1_units_by_alpha.csv", index=False)
    h1_df.to_csv(inter / "06_synthetic_h1_point_estimates.csv", index=False)
    h1_fold_df.to_csv(inter / "07_synthetic_h1_family_folds.csv", index=False)

    sumo_raw.to_csv(inter / "10_sumo_rows_alpha.csv", index=False)
    sumo_models.to_csv(inter / "11_sumo_model_dmis_by_alpha.csv", index=False)
    sumo_rank.to_csv(inter / "12_sumo_rank_agreement.csv", index=False)
    sumo_rank_pairs.to_csv(inter / "13_sumo_model_rank_pairs.csv", index=False)
    sumo_regimes.to_csv(inter / "14_sumo_regimes_by_alpha.csv", index=False)
    regime_pairs.to_csv(inter / "15_sumo_regime_pairs.csv", index=False)

    final.to_csv(tables / "Table25_alpha_sensitivity.csv", index=False)
    write_table25_tex(final, tables / "Table25_alpha_sensitivity.tex")

    # Extra detailed table useful for appendix/audit.
    detailed = final.merge(
        syn_rank[["alpha", "kendall_tau_b"]].rename(
            columns={"kendall_tau_b": "synthetic_kendall_tau_b"}
        ),
        on="alpha",
    ).merge(
        sumo_rank[["alpha", "kendall_tau_b"]].rename(
            columns={"kendall_tau_b": "sumo_kendall_tau_b"}
        ),
        on="alpha",
    )
    detailed.to_csv(tables / "TableS_alpha_sensitivity_detailed.csv", index=False)

    audit = {
        "status": "complete",
        "alphas": list(ALPHAS),
        "reference_alpha": REFERENCE_ALPHA,
        "synthetic": {
            "h1_root": str(h1_root),
            "raw_source": str(raw_path),
            "units_source": str(units_path),
            "rows": len(syn_raw),
            "models": int(syn_raw["model"].nunique()),
            "alpha050_dmis_reconstruction": syn_recon_audit,
            "alpha050_unit_reproduction": alpha050_unit_audit,
        },
        "sumo": {
            "root": str(sumo_root),
            "rows": len(sumo_raw),
            "models": int(sumo_raw["model"].nunique()),
            "alpha050_dmis_reconstruction": sumo_recon_audit,
            "alpha050_saved_regime_reproduction": sumo_saved_audit,
            "reference_thresholds": reference_thresholds,
        },
    }
    save_json(diag / "alpha_sensitivity_audit.json", audit)

    # ---------------- Figures ----------------
    plot_rank_agreement(final, figs)
    plot_h1(final, figs)
    plot_rank_trajectories(
        syn_rank_pairs, "Controlled Synthetic", figs, "Fig59_Synthetic_rank_trajectories"
    )
    plot_rank_trajectories(
        sumo_rank_pairs, "Abstract SUMO", figs, "Fig59_SUMO_rank_trajectories"
    )
    plot_sumo_regime_agreement(final, figs)
    for a in (0.25, 0.75):
        plot_regime_confusion(regime_pairs, a, figs)

    log("")
    log("================ Section 5.9 Summary ================")
    print(final.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    log(f"[SAVED] {tables / 'Table25_alpha_sensitivity.csv'}")
    log(f"[SAVED] {tables / 'Table25_alpha_sensitivity.tex'}")
    log(f"[SAVED] {diag / 'alpha_sensitivity_audit.json'}")


if __name__ == "__main__":
    main()
