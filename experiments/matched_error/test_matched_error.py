#!/usr/bin/env python3
"""Tiny deterministic end-to-end contract tests; does not require server results."""
import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT=Path(__file__).with_name('run_matched_error.py')


def fixture(root, missing=False, strong_b=0.10):
    path=root/'analysis/h1/intermediate/01_h1_raw_calibrated_rows.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for m, wrd, dm, strong in [('STGCN',0.010,0.10,0.03),('GWNet',0.012,0.20,strong_b),('DLinear',0.25,0.40,0.50)]:
        for seed in range(5):
            for level in ('gamma=0.8','gamma=0.6','gamma=0.4','gamma=0.2'):
                if missing and m=='GWNet' and seed==4 and level=='gamma=0.2':continue
                rows.append(dict(model=m,model_seed=0,intervention_type='weight_modification',
                                 paper_intensity_level=level,intervention_seed=seed,evaluation_track='B',
                                 causal_valid=True,leakage_detected=False,
                                 RD=strong if level=='gamma=0.2' else wrd,
                                 DMIS_cal=dm))
    pd.DataFrame(rows).to_csv(path,index=False)


def run(root, extra=(), expect_ok=True):
    cmd=[sys.executable,str(SCRIPT),'--experiment-root',str(root),'--expected-models','3',*extra]
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if expect_ok and p.returncode:raise AssertionError(p.stdout)
    if not expect_ok and not p.returncode:raise AssertionError('Expected protocol failure')
    return p


def main():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); fixture(root)
        result=run(root)
        output=root/'analysis/matched_error'
        sel=pd.read_csv(output/'05_eligible_pairs_WEAK_ONLY.csv')
        table=pd.read_csv(output/'Table18_matched_error_cohort.csv')
        assert len(sel)==1 and sel.iloc[0].higher_DMIS_model=='GWNet'
        assert len(pd.read_csv(output/'07_matched_strong_seed_rows.csv'))==5
        assert table.iloc[0].Eligible_pairs==1
        assert np.isclose(table.iloc[0].Difference_high_minus_low,.07)
        frozen=(output/'05_eligible_pairs_WEAK_ONLY.csv').read_bytes()
        # Strong endpoint changes must NOT change which weak-matched pairs qualify.
        fixture(root,strong_b=.01)
        run(root)
        assert frozen==(output/'05_eligible_pairs_WEAK_ONLY.csv').read_bytes()
        assert np.isclose(pd.read_csv(output/'Table18_matched_error_cohort.csv').iloc[0].Difference_high_minus_low,-.02)
        # A missing matched strong replicate must fail, never silently average four seeds.
        fixture(root,missing=True)
        run(root,expect_ok=False)
        print('TEST 1: complete weak pairing + same-seed strong evaluation PASS')
        print('TEST 2: weak cohort unchanged when held-out outcomes changed PASS')
        print('TEST 3: missing strong outcome causes explicit failure PASS')

if __name__=='__main__':main()
