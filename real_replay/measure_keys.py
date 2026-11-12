"""Paired key-construction microbenchmark; not end-to-end model timing."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'cache_replay'))
from run_regression import DSAIWrapper,exact_key
A=np.load(HERE/'outputs/test_inputs.npz')['baseline_graph'];rng=np.random.default_rng(7021)
plan={'blocks':20,'repetitions_per_block':1000,'input_shape':list(A.shape),'input_dtype':str(A.dtype),'time_scope':'key construction only; identical fixed adjacency; function outputs not retained','seed':7021}
(HERE/'key_plan.json').write_text(json.dumps(plan,indent=2));rows=[]
for _ in range(100):DSAIWrapper._adj_key(A);exact_key(A)
for b in range(20):
 for name in rng.permutation(['summary','content']):
  fn=DSAIWrapper._adj_key if name=='summary' else exact_key
  t=time.perf_counter_ns()
  for _ in range(1000):key=fn(A)
  dt=(time.perf_counter_ns()-t)/1000/1000
  rows.append({'block':b,'key':name,'microseconds_per_construction':dt})
f=pd.DataFrame(rows);f.to_csv(HERE/'outputs/key_timings.csv',index=False)
p=f.pivot(index='block',columns='key',values='microseconds_per_construction');logs=np.log(p['content']/p['summary']).to_numpy();bs=np.exp(np.mean(logs[rng.integers(0,20,(20000,20))],axis=1))
r={'summary_median_us':float(p['summary'].median()),'content_median_us':float(p['content'].median()),'paired_content_over_summary':float(np.exp(logs.mean())),'paired_95_interval':np.quantile(bs,[.025,.975]).tolist(),'one_key_payload_bytes':len(exact_key(A)[2]),'two_key_payload_KiB':2*len(exact_key(A)[2])/1024,'blocks':20}
(HERE/'outputs/key_summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
