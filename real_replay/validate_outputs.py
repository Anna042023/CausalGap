"""Independent input, checkpoint-inference, and saved SQL-aggregate validation."""
from pathlib import Path
import os,json,hashlib
import numpy as np
import pandas as pd
import torch
from train_metr_la import make_model,evaluate
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'
def main():
 plan=json.loads((HERE/'plan.json').read_text());rawpath=Path(os.environ.get('METR_LA_PATH',str(HERE/'data/metr-la.h5')))
 assert hashlib.sha256(rawpath.read_bytes()).hexdigest()==plan['raw_sha256']
 raw=pd.read_hdf(rawpath);v=raw.to_numpy(dtype='f4');mask=np.isfinite(v)&(v>0);tr=plan['split_boundaries'][0]
 mean=float(v[:tr][mask[:tr]].astype('f8').mean());std=float(v[:tr][mask[:tr]].astype('f8').std());assert [mean,std]==plan['normalization']
 xall=((pd.DataFrame(np.where(mask,v,np.nan)).ffill().fillna(mean).to_numpy(dtype='f4')-mean)/std).astype('f4')
 inp=np.load(OUT/'test_inputs.npz');starts=np.arange(len(v)-2016,len(v)-11,12)
 assert np.array_equal(np.stack([xall[t-12:t,:,None] for t in starts]),inp['x'])
 assert np.array_equal(np.stack([v[t:t+12] for t in starts]),inp['target'])
 assert np.array_equal(np.stack([mask[t:t+12] for t in starts]),inp['valid'])
 assert np.array_equal(raw.columns.astype(str).to_numpy(dtype='U'),inp['sensor_ids'])
 assert np.array_equal(raw.index[starts-1].asi8,inp['origins_ns'])
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);rows=[]
 groups=pd.read_csv(OUT/'sql_groups.csv');assert len(groups)==21
 for seed in plan['seeds']:
  ck=torch.load(OUT/f'checkpoint_{seed}.pt',weights_only=False,map_location='cpu');p=np.load(OUT/f'predictions_{seed}.npz')
  m=make_model(inp['baseline_graph'],True);m.load_state_dict(ck['weights']);pred=evaluate(m,inp['x'],inp['changed_graph'],mean,std)
  err=float(np.max(np.abs(pred.astype('f8')-p['cold_changed'])));assert err<=1e-4
  assert np.array_equal(p['baseline'],p['legacy_changed']);assert np.array_equal(p['cold_changed'],p['repaired_changed'])
  for day in range(7):
   sl=slice(day*24,(day+1)*24);valid=inp['valid'][sl];y=inp['target'][sl].astype('f8');g=groups[(groups.seed==seed)&(groups.day==day)].iloc[0]
   assert int(valid.sum())==int(g['valid'])
   for key,col in [('baseline','abs_base'),('cold_changed','abs_cold'),('legacy_changed','abs_legacy'),('repaired_changed','abs_repaired')]:
    expected=np.abs(p[key][sl].astype('f8')-y)[valid].sum();assert np.isclose(expected,g[col],rtol=1e-12,atol=1e-7)
  rows.append({'seed':seed,'checkpoint_fresh_max_error':err,'saved_legacy_equals_baseline':True,'saved_repaired_equals_cold':True,'day_groups_pass':7})
 result={'raw_and_preprocessing':True,'models':rows,'total_SQL_groups':21,'tolerance':1e-4}
 (OUT/'independent_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
