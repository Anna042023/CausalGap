#!/usr/bin/env python3
"""Recompute saved METR-LA graph-input predictions from the released checkpoints.

Uses the archived normalized test windows and graph matrices; no retraining or
raw METR-LA HDF5 is required. The observed targets are never changed.
"""
from pathlib import Path
import json
import numpy as np
import torch
from train_metr_la import make_model, evaluate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    inp = np.load(OUT / 'test_inputs.npz', allow_pickle=False)
    x, a, b = inp['x'], inp['baseline_graph'], inp['changed_graph']
    for seed in (7, 17, 27):
        checkpoint = torch.load(OUT / f'checkpoint_{seed}.pt', weights_only=False, map_location='cpu')
        weights = checkpoint['weights']
        mean, std = checkpoint['plan']['normalization']
        def fresh(fixed=True):
            m = make_model(a, fixed)
            m.load_state_dict(weights)
            m.eval()
            return m
        base = evaluate(fresh(), x, a, mean, std)
        cold = evaluate(fresh(), x, b, mean, std)
        stale = fresh(False)
        evaluate(stale, x[:1], a, mean, std)  # fill original graph cache
        legacy = evaluate(stale, x, b, mean, std)
        repaired = fresh(True)
        evaluate(repaired, x[:1], a, mean, std)
        warm = evaluate(repaired, x, b, mean, std)
        np.savez_compressed(OUT / f'predictions_{seed}.npz', baseline=base,
                            cold_changed=cold, legacy_changed=legacy,
                            repaired_changed=warm)
        print(json.dumps({'seed': seed, 'baseline_MAE_mph': float(np.abs(base-inp['target'])[inp['valid']].mean()),
                          'cold_MAE_mph': float(np.abs(cold-inp['target'])[inp['valid']].mean()),
                          'stale_max_discrepancy_mph': float(np.max(np.abs(legacy-cold))),
                          'repaired_max_discrepancy_mph': float(np.max(np.abs(warm-cold)))}, indent=2), flush=True)

if __name__ == '__main__':
    main()
