#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from protocol import H1_LEVELS, STRUCTURAL_TYPES, safe_name


def main():
    ap = argparse.ArgumentParser(description='Inventory the calibrated SUMO-METR-LA source required by H1.')
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--eval-seeds', nargs='+', type=int, default=[0,1,2,3,4])
    args = ap.parse_args()

    root = args.source_root
    expected = [root / 'baseline.npz']
    for kind in STRUCTURAL_TYPES:
        for spec in H1_LEVELS[kind]:
            for seed in args.eval_seeds:
                expected.append(root / 'interventions' / kind / safe_name(spec['level']) / f'seed_{seed:04d}.npz')

    missing = [p for p in expected if not p.exists()]
    print('='*88)
    print('SUMO-METR-LA H1 source inventory')
    print('source root    :', root)
    print('expected files :', len(expected), '(1 baseline +', len(expected)-1, 'interventions)')
    print('found          :', len(expected)-len(missing))
    print('missing        :', len(missing))
    print('='*88)

    if (root/'baseline.npz').exists():
        with np.load(root/'baseline.npz', allow_pickle=True) as z:
            print('baseline keys  :', z.files)
            if 'x' in z.files: print('baseline x     :', tuple(z['x'].shape))
            if 'y' in z.files: print('baseline y     :', tuple(z['y'].shape))
            if 'adj' in z.files: print('baseline adj   :', tuple(z['adj'].shape))

    if missing:
        print('\nMissing files (first 30):')
        for p in missing[:30]:
            print('  ', p)
        if len(missing) > 30:
            print(f'  ... and {len(missing)-30} more')
        print('\n[STOP] Calibrated SUMO source is incomplete. Do not start model inference.')
        raise SystemExit(2)

    print('\n[PASS] Required SUMO source files are present.')
    print('Next: run prepare_sumo_envs.py; it will perform strict shape/timeline/adjacency checks.')

if __name__ == '__main__':
    main()
