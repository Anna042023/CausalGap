#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from protocol import ALL_MODELS, DEFAULT_EXPERIMENT_ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    ap.add_argument("--require-sumo", action="store_true")
    args = ap.parse_args()

    from causalgap.models.registry import SPECS
    from causalgap.evaluator import evaluate_intervention
    from causalgap.interventions import make_intervention
    import sklearn
    import scipy
    import torch

    missing_models = [m for m in ALL_MODELS if m not in SPECS]
    if missing_models:
        raise RuntimeError(f"Current unified code is missing H1 models: {missing_models}")
    if len(ALL_MODELS) != 21:
        raise RuntimeError(f"H1 package expects the user's 21 runnable models, got {len(ALL_MODELS)}")

    sig_eval = inspect.signature(evaluate_intervention)
    for name in ("artifact_level", "fail_on_unobservable_trackb"):
        if name not in sig_eval.parameters:
            raise RuntimeError(
                f"Current causalgap.evaluator.evaluate_intervention lacks '{name}'. "
                "Install/use the already-corrected Section 5.1 core before H1."
            )
    sig_inter = inspect.signature(make_intervention)
    for name in ("severity", "weight_gamma"):
        if name not in sig_inter.parameters:
            raise RuntimeError(f"Current make_intervention lacks {name}; H1 weak/strong levels cannot be generated safely.")

    syn = args.experiment_root / "environments" / "synthetic" / "manifest.csv"
    if not syn.exists():
        print(f"[INFO] synthetic environment not prepared yet: {syn}")
    else:
        print(f"[OK] synthetic manifest: {syn}")
    sm = args.experiment_root / "environments" / "sumo_metrla" / "manifest.csv"
    if args.require_sumo and not sm.exists():
        raise FileNotFoundError(f"SUMO manifest required but missing: {sm}")
    if sm.exists(): print(f"[OK] SUMO manifest: {sm}")

    print(f"[OK] torch={torch.__version__} sklearn={sklearn.__version__} scipy={scipy.__version__}")
    print("[OK] evaluator/intervention APIs are compatible with the H1 package")
    print("[OK] 21-model registry check passed")


if __name__ == "__main__":
    main()
