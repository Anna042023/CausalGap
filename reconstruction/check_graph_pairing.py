#!/usr/bin/env python3
"""Transactional fault tests for the saved source-to-SQL relation."""
import sqlite3,csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'reconstruction/analysis'; DB=P/'graph_response.sqlite'
con=sqlite3.connect(DB);con.execute('PRAGMA foreign_keys=ON')
orig=con.execute('SELECT COUNT(*) FROM unit_error').fetchone()[0]
assert orig==417312
seed=7;origin=153;sensor=56
key=(seed,origin,sensor)
rs=[]
def pair_check():
 missing=con.execute("SELECT COUNT(*) FROM unit_error b LEFT JOIN unit_error c ON c.seed=b.seed AND c.origin=b.origin AND c.sensor=b.sensor AND c.graph_version='strong' WHERE b.seed=? AND b.graph_version='base' AND c.seed IS NULL",(seed,)).fetchone()[0]
 bad_support=con.execute("SELECT COUNT(*) FROM unit_error c JOIN unit_error b ON c.seed=b.seed AND c.origin=b.origin AND c.sensor=b.sensor AND b.graph_version='base' WHERE c.seed=? AND c.graph_version='strong' AND c.valid_horizons!=b.valid_horizons",(seed,)).fetchone()[0]
 bad_version=con.execute("SELECT COUNT(*) FROM unit_error WHERE seed=? AND graph_version NOT IN ('base','weak05','weak10','strong')",(seed,)).fetchone()[0]
 return missing,bad_support,bad_version
rs.append(['clean','accepted',*pair_check()])
# Three errors on one record; all rolled back
for name,statement in [('missing counterpart',"DELETE FROM unit_error WHERE seed=? AND origin=? AND sensor=? AND graph_version='strong'"),('wrong graph version',"UPDATE unit_error SET graph_version='strong_wrong' WHERE seed=? AND origin=? AND sensor=? AND graph_version='strong'"),('one-sided horizon mask',"UPDATE unit_error SET valid_horizons=valid_horizons-1 WHERE seed=? AND origin=? AND sensor=? AND graph_version='strong'")]:
 con.execute('SAVEPOINT stress')
 n=con.execute(statement,key).rowcount; assert n==1
 rs.append([name,'rejected',*pair_check()]);con.execute('ROLLBACK TO stress');con.execute('RELEASE stress')
# Enforced uniqueness prevents insertion before execution
con.execute('SAVEPOINT stress')
try:
 con.execute("INSERT INTO unit_error SELECT * FROM unit_error WHERE seed=? AND origin=? AND sensor=? AND graph_version='strong'",key)
 raise AssertionError('duplicate must be rejected')
except sqlite3.IntegrityError:
 rs.append(['duplicate logical key','insertion rejected',*pair_check()])
con.execute('ROLLBACK TO stress');con.execute('RELEASE stress')
assert con.execute('SELECT COUNT(*) FROM unit_error').fetchone()[0]==orig
con.close()
with (P/'sql_integrity_checks.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['fault','status','missing_counterparts','unequal_valid_horizon_counts','unregistered_graph_versions']);w.writerows(rs)
(P/'sql_integrity_audit.json').write_text(json.dumps({'origin_seed_sensor':[seed,origin,sensor],'source_rows_unmodified':orig,'tests':rs,'all_faults_detected':True,'source_db_sha256':hashlib.sha256(DB.read_bytes()).hexdigest()},indent=2))
print(rs)
