#!/usr/bin/env python3
"""Deterministic export of existing per-cell METR-LA prediction arrays.
Derived material, not H1/H2 original experimental outputs. Does not train models.
"""
from __future__ import annotations
import csv, gzip, argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--output',type=Path,default=ROOT/'reconstruction/outputs/real_prediction_units.csv.gz')
 ap.add_argument('--manifest',type=Path,default=ROOT/'reconstruction/outputs/real_prediction_units_manifest.json')
 args=ap.parse_args()
 with np.load(ROOT/'real_replay/outputs/test_inputs.npz',allow_pickle=False) as a:
  target=a['target'].copy(); valid=a['valid'].copy(); origins=a['origins_ns'].copy(); sensors=a['sensor_ids'].copy()
 assert target.shape==(168,12,207)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 rows=0;valid_rows=0
 # Fixed gzip header for reproducible gzip binary on repeated builds.
 with args.output.open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0,compresslevel=6) as gz:
   import io
   with io.TextIOWrapper(gz,encoding='utf-8',newline='') as f:
    w=csv.writer(f,lineterminator='\n')
    w.writerow(['seed','origin_ns','horizon','sensor_id','valid','target','baseline','cold_changed','legacy_changed','repaired_changed'])
    for seed in (7,17,27):
     with np.load(ROOT/f'real_replay/outputs/predictions_{seed}.npz',allow_pickle=False) as n:
      b=n['baseline'];cold=n['cold_changed'];legacy=n['legacy_changed'];fixed=n['repaired_changed']
      assert b.shape==cold.shape==legacy.shape==fixed.shape==target.shape
      assert np.array_equal(cold,fixed)
      for o in range(168):
       for h in range(12):
        for s in range(207):
         w.writerow([seed,int(origins[o]),h+1,str(sensors[s]),int(valid[o,h,s]),
                     repr(float(target[o,h,s])),repr(float(b[o,h,s])),repr(float(cold[o,h,s])),
                     repr(float(legacy[o,h,s])),repr(float(fixed[o,h,s]))])
         rows+=1;valid_rows+=int(valid[o,h,s])
 assert rows==1251936 and valid_rows==1195200
 manifest={'source_kind':'DERIVED_FROM_SAVED_METR_LA_PREDICTIONS_NOT_H1_H2','row_key':['seed','origin_ns','horizon','sensor_id'],
           'rows':rows,'valid_rows':valid_rows,'per_checkpoint_rows':417312,'source_test_inputs_sha256':digest(ROOT/'real_replay/outputs/test_inputs.npz'),
           'source_prediction_sha256':{str(s):digest(ROOT/f'real_replay/outputs/predictions_{s}.npz') for s in (7,17,27)},
           'output_sha256':digest(args.output),'output_bytes':args.output.stat().st_size}
 args.manifest.write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'rows':rows,'valid':valid_rows,'compressed_bytes':manifest['output_bytes'],'sha256':manifest['output_sha256']},indent=2))
if __name__=='__main__':main()
