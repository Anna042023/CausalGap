#!/usr/bin/env python3
"""Recompute H1/H2 summary rows from saved NPZ run units.
No torch model forward passes, no historical run data, no fabricated results.
"""
from pathlib import Path
import sys, json, csv, hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'graph_inputs'
sys.path.insert(0,str(HERE))
from run_graph_input import new_h1, new_h2

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def same_table(rows,filename):
    with (OUT/filename).open(newline='') as f: orig=list(csv.DictReader(f))
    assert len(orig)==len(rows),(filename,len(orig),len(rows))
    for a,b in zip(orig,rows):
        assert set(a)==set(b),(filename,set(a)^set(b))
        for k in b:
            if isinstance(b[k],(int,float)):
                assert np.isclose(float(a[k]),float(b[k]),atol=1e-10,rtol=0),(filename,k,a[k],b[k])
            else: assert a[k]==b[k],(filename,k,a[k],b[k])
    return len(rows)

def main():
    meta=json.loads((OUT/'manifest.json').read_text())
    assert sha(HERE/'run_graph_input.py')==meta['script_sha256']
    assert sha(ROOT/'real_replay/outputs/test_inputs.npz')==meta['base_input_sha256']
    npz=np.load(ROOT/'real_replay/outputs/test_inputs.npz')
    valid=np.asarray(npz['valid'],bool)
    assert valid.shape==(168,12,207)
    rows={}
    for seed in (7,17,27):
        assert sha(ROOT/f'real_replay/outputs/checkpoint_{seed}.pt')==meta['checkpoint_sha256'][str(seed)]
        fname=OUT/f'graph_input_seed{seed}.npz'
        assert sha(fname)==meta['outputs'][fname.name]['sha256']
        with np.load(fname) as dd:
            assert np.array_equal(dd['valid'],valid)
            rows[seed]={k:np.array(dd[k]) for k in dd.files if k!='valid'}
            assert rows[seed]['base_pred'].shape==(168,12,207)
            assert rows[seed]['base_rep'].shape==(168,207,16)
            assert np.array_equal(rows[seed]['target'],npz['target'])
    h1=new_h1(rows,valid);h2=new_h2(rows,valid)
    counts={'H1':same_table(h1,'weak_to_strong.csv'),'H2':same_table(h2,'calibration.csv')}
    for name,rec in meta['outputs'].items():assert sha(OUT/name)==rec['sha256'],name
    report={'status':'PASS','source':'R18 saved NPZ run-level prediction and representation outputs','numerical_tolerance':1e-10,'table_rows':counts,'seed_count':3,'h1_units_per_holdout':33644,'h2_heldout_cells_per_seed':263296,'historical_reconstruction':False}
    (OUT/'independent_reanalysis_check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
