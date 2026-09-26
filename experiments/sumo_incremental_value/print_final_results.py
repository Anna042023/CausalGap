#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--experiment-root', type=Path, required=True)
    args=ap.parse_args()
    root=args.experiment_root/'analysis'/'h1'
    table=root/'tables'/'Table2_H1_incremental_diagnostic_value.csv'
    units=root/'intermediate'/'04_h1_prediction_units.csv'
    if not table.exists(): raise FileNotFoundError(table)
    if not units.exists(): raise FileNotFoundError(units)
    t=pd.read_csv(table)
    u=pd.read_csv(units)
    print('\n===== FINAL H1 TABLE 2 =====')
    print(t.to_string(index=False))
    print('\n===== COVERAGE =====')
    print('prediction units =', len(u))
    print('models           =', u['model'].nunique())
    print('settings         =', sorted(u['setting'].astype(str).unique().tolist()))
    print('families         =', sorted(u['model_family'].astype(int).unique().tolist()))
    print('interventions    =', sorted(u['intervention_type'].astype(str).unique().tolist()))
    print('\nunits by setting/model family:')
    print(u.groupby(['setting','model_family_name']).size().rename('units').reset_index().to_string(index=False))

if __name__=='__main__': main()
