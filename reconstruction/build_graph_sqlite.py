#!/usr/bin/env python3
"""Rebuild a keyed SQLite relation from saved graph-input arrays.
Never treats graph-input rewiring as physical counterfactual outcomes.
No model retraining. An intentional one-row fault is rolled back.
"""
from pathlib import Path
import sqlite3,hashlib,csv,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'reconstruction/graph_inputs';OUT=ROOT/'reconstruction/analysis';OUT.mkdir(exist_ok=True)
DB=OUT/'graph_response.sqlite'
if DB.exists(): DB.unlink()
conn=sqlite3.connect(DB)
conn.execute('PRAGMA journal_mode=DELETE')
conn.execute('PRAGMA synchronous=OFF')
conn.executescript('''
CREATE TABLE unit_error (
 seed INTEGER NOT NULL, graph_version TEXT NOT NULL,
 origin INTEGER NOT NULL, sensor INTEGER NOT NULL,
 valid_horizons INTEGER NOT NULL, abs_error_sum REAL NOT NULL,
 relative_rep_response REAL NOT NULL,
 PRIMARY KEY(seed,graph_version,origin,sensor)
);
CREATE INDEX by_graph_seed ON unit_error(graph_version,seed);
''')
levels=['base','weak05','weak10','strong']; rowsource={}
start=time.time()
for seed in [7,17,27]:
 z=np.load(IN/f'graph_input_seed{seed}.npz')
 y=z['target'];m=z['valid'];br=z['base_rep'];n=m.sum(axis=1).astype(np.int16)
 for level in levels:
  pred=z[f'{level}_pred'];r=z[f'{level}_rep']
  e=np.abs((pred-y).astype(np.float64))
  s=(e*m).sum(axis=1)
  q=np.linalg.norm(r-br,axis=-1)/(np.linalg.norm(br,axis=-1)+1e-7)
  rows=[]
  for origin in range(s.shape[0]):
   rows.extend((seed,level,origin,sensor,int(n[origin,sensor]),float(s[origin,sensor]),float(q[origin,sensor])) for sensor in range(s.shape[1]))
  conn.executemany('INSERT INTO unit_error VALUES(?,?,?,?,?,?,?)',rows)
  rowsource[(seed,level)]={'n_rows':len(rows),'valid_rows':int((n>0).sum()),'total_valid_horizons':int(n.sum()),'source_abs_error_mean':float(s.sum()/n.sum()),'source_response_valid_mean':float(q[n>0].mean())}
 conn.commit()
print('ingested',time.time()-start,flush=True)
mae_rows=conn.execute('SELECT seed,graph_version,COUNT(*),SUM(valid_horizons),SUM(abs_error_sum)/SUM(valid_horizons),AVG(relative_rep_response) FROM unit_error GROUP BY seed,graph_version ORDER BY seed,graph_version').fetchall()
assert len(mae_rows)==12
for seed,level,ntot,validcnt,mae,q in mae_rows:
 assert ntot==rowsource[(seed,level)]['n_rows']==34776
 assert validcnt==rowsource[(seed,level)]['total_valid_horizons']
 assert abs(mae-rowsource[(seed,level)]['source_abs_error_mean'])<1e-8
with (OUT/'sql_graph_metrics.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['checkpoint_seed','graph_version','key_rows','valid_horizon_cells','sql_weighted_mae_mph','mean_all_rows_relative_response']);w.writerows(mae_rows)
paired_query='''
SELECT c.seed,c.graph_version,
 COUNT(*) AS all_pairs,
 SUM(CASE WHEN c.valid_horizons>0 AND b.valid_horizons>0 AND c.valid_horizons=b.valid_horizons THEN 1 ELSE 0 END) AS valid_pairs,
 AVG(CASE WHEN c.valid_horizons>0 AND b.valid_horizons>0 AND c.valid_horizons=b.valid_horizons THEN c.relative_rep_response END) AS relative_response,
 AVG(CASE WHEN c.valid_horizons>0 AND b.valid_horizons>0 AND c.valid_horizons=b.valid_horizons THEN (c.abs_error_sum-b.abs_error_sum)/c.valid_horizons END) AS mean_sensor_error_delta
FROM unit_error c INDEXED BY by_graph_seed CROSS JOIN unit_error b INDEXED BY sqlite_autoindex_unit_error_1
ON c.seed=b.seed AND c.origin=b.origin AND c.sensor=b.sensor
AND b.graph_version='base'
WHERE c.graph_version IN ('weak05','weak10','strong')
GROUP BY c.seed,c.graph_version ORDER BY c.seed,c.graph_version;
'''
pairs=conn.execute(paired_query).fetchall()
assert len(pairs)==9
ref=list(csv.DictReader((OUT/'response_unit_summary.csv').open()))
for seed,level,nall,nvalid,q,d in pairs:
 assert nall==34776 and nvalid==33644
 x=next(t for t in ref if int(t['checkpoint_seed'])==seed and t['graph_version']==level)
 assert abs(float(x['mean_relative_rep_response'])-q)<1e-7
 assert abs(float(x['mean_mae_change_mph'])-d)<1e-7
with (OUT/'sql_pair_summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['checkpoint_seed','graph_version','all_origin_sensor_pairs','valid_origin_sensor_pairs','sql_mean_relative_response','sql_mean_mae_change_mph']);w.writerows(pairs)
# Deliberate single-cell deletion inside a savepoint; restore original database.
conn.execute('SAVEPOINT controlled_fault')
before=conn.execute("""SELECT COUNT(*),AVG((c.abs_error_sum-b.abs_error_sum)/c.valid_horizons) FROM unit_error c INDEXED BY by_graph_seed CROSS JOIN unit_error b INDEXED BY sqlite_autoindex_unit_error_1 ON c.seed=b.seed AND c.origin=b.origin AND c.sensor=b.sensor AND b.graph_version='base' WHERE c.graph_version='strong' AND c.seed=7 AND c.valid_horizons>0 AND b.valid_horizons=c.valid_horizons""").fetchone()
source=conn.execute("""SELECT c.seed,c.origin,c.sensor,(c.abs_error_sum-b.abs_error_sum)/c.valid_horizons AS delta
FROM unit_error c INDEXED BY by_graph_seed CROSS JOIN unit_error b INDEXED BY sqlite_autoindex_unit_error_1 ON c.seed=b.seed AND c.origin=b.origin AND c.sensor=b.sensor AND b.graph_version='base'
WHERE c.graph_version='strong' AND c.seed=7 AND c.valid_horizons>0 AND c.valid_horizons=b.valid_horizons ORDER BY delta DESC,c.origin,c.sensor LIMIT 1""").fetchone()
assert source
conn.execute("DELETE FROM unit_error WHERE seed=? AND origin=? AND sensor=? AND graph_version='strong'",tuple(map(int,source[:3])))
after=conn.execute("""SELECT COUNT(*),AVG((c.abs_error_sum-b.abs_error_sum)/c.valid_horizons) FROM unit_error c INDEXED BY by_graph_seed CROSS JOIN unit_error b INDEXED BY sqlite_autoindex_unit_error_1 ON c.seed=b.seed AND c.origin=b.origin AND c.sensor=b.sensor AND b.graph_version='base' WHERE c.graph_version='strong' AND c.seed=7 AND c.valid_horizons>0 AND b.valid_horizons=c.valid_horizons""").fetchone()
missing=conn.execute("SELECT COUNT(*) FROM unit_error b LEFT JOIN unit_error c ON b.seed=c.seed AND b.origin=c.origin AND b.sensor=c.sensor AND c.graph_version='strong' WHERE b.graph_version='base' AND b.seed=7 AND c.seed IS NULL").fetchone()[0]
assert before[0]==after[0]+1 and missing==1 and after[1]<before[1]
conn.execute('ROLLBACK TO controlled_fault');conn.execute('RELEASE controlled_fault')
assert conn.execute("SELECT COUNT(*) FROM unit_error WHERE seed=7 AND graph_version='strong'").fetchone()[0]==34776
conn.commit(); conn.close()
audit={'source':'R18 three saved checkpoint run NPZ','sqlite_rows':417312,'graph_versions_per_seed':4,'keys_per_seed_per_graph':34776,'valid_origin_sensor_pairs_per_seed_graph':33644,'checked_pairs_total':sum(v[3] for v in pairs),'independent_reanalysis':'PASS','reproduction_tolerance':1e-7,'fault_rollback':'PASS','fault_seed7_version':'strong','deleted_seed_origin_sensor':[int(x) for x in source[:3]],'deletion_baseline_delta':float(before[1]),'deletion_naive_delta':float(after[1]),'deletion_delta_mph':float(after[1]-before[1]),'missing_counterpart_detected':int(missing),'records_and_metrics_are_observational_input_only':True,'runtime_seconds':round(time.time()-start,2),'artifacts':{}}
for f in [DB,OUT/'sql_graph_metrics.csv',OUT/'sql_pair_summary.csv']:
 audit['artifacts'][f.name]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
(OUT/'sql_integration_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
