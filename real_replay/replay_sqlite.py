"""Reload saved real-data predictions. Rebuildable database and snapshot tests.
Corrections are controlled writes into the NEW replay, not observed traffic events.
Two database connections hold overlapping transactions; no throughput claim.
"""
from pathlib import Path
import json, sqlite3, time, hashlib, csv
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'
DB=OUT/'real_replay.sqlite'
QUERY='''SELECT seed, day, COUNT(*), SUM(ABS(base-target)), SUM(ABS(cold-target)),
 SUM(ABS(legacy-target)), SUM(ABS(repaired-target)), MAX(ABS(repaired-cold)), MAX(ABS(legacy-cold))
 FROM forecast_pairs WHERE valid=1 GROUP BY seed,day ORDER BY seed,day'''
def close(a,b,tol=1e-7):
    return np.allclose(np.asarray(a,dtype='f8'),np.asarray(b,dtype='f8'),rtol=1e-10,atol=tol)
def main():
    if DB.exists():DB.unlink()
    inp=np.load(OUT/'test_inputs.npz');target=inp['target'];valid=inp['valid'];O,H,N=target.shape
    c=sqlite3.connect(DB);c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA synchronous=FULL')
    c.executescript('''CREATE TABLE runs(seed INTEGER PRIMARY KEY, checkpoint_sha TEXT NOT NULL, baseline_graph_sha TEXT NOT NULL, changed_graph_sha TEXT NOT NULL);
    CREATE TABLE forecast_pairs(seed INTEGER, origin INTEGER, day INTEGER, horizon INTEGER, node INTEGER, target REAL, base REAL, cold REAL, legacy REAL, repaired REAL, valid INTEGER CHECK(valid IN(0,1)), PRIMARY KEY(seed,origin,horizon,node), FOREIGN KEY(seed) REFERENCES runs(seed));
    CREATE TABLE version_state(revision INTEGER NOT NULL); INSERT INTO version_state VALUES(0);''')
    oracle=[];tot=0
    for seed in [7,17,27]:
        p=np.load(OUT/f'predictions_{seed}.npz')
        h=lambda x:hashlib.sha256(x).hexdigest()
        c.execute('INSERT INTO runs VALUES(?,?,?,?)',(seed,h((OUT/f'checkpoint_{seed}.pt').read_bytes()),h(inp['baseline_graph'].tobytes()),h(inp['changed_graph'].tobytes())))
        for day in range(7):
            os=np.arange(day*24,(day+1)*24);n=len(os)*H*N
            oi,hi,ni=np.meshgrid(os,np.arange(H),np.arange(N),indexing='ij')
            arrays=[np.full(n,seed),oi.ravel(),np.full(n,day),hi.ravel(),ni.ravel(),target[os].ravel().astype('f8')]
            arrays += [p[k][os].ravel().astype('f8') for k in ['baseline','cold_changed','legacy_changed','repaired_changed']]
            arrays += [valid[os].ravel().astype(int)]
            def rows():
                for row in zip(*arrays):yield tuple(int(x) if i in (0,1,2,3,4,10) else float(x) for i,x in enumerate(row))
            c.executemany('INSERT INTO forecast_pairs VALUES(?,?,?,?,?,?,?,?,?,?,?)',rows());tot+=n
            m=valid[os];y=target[os].astype('f8')
            a=[seed,day,int(m.sum())]+[float(np.abs(p[k][os].astype('f8')-y)[m].sum()) for k in ['baseline','cold_changed','legacy_changed','repaired_changed']]+[float(np.abs(p['repaired_changed'][os].astype('f8')-p['cold_changed'][os])[m].max()),float(np.abs(p['legacy_changed'][os].astype('f8')-p['cold_changed'][os])[m].max())]
            oracle.append(a)
        c.commit();print('loaded',seed,tot,flush=True)
    actual=list(c.execute(QUERY));assert close(actual,oracle)
    pd.DataFrame(actual,columns=['seed','day','valid','abs_base','abs_cold','abs_legacy','abs_repaired','max_repaired_error','max_legacy_error']).to_csv(OUT/'sql_groups.csv',index=False)
    c.execute('CREATE INDEX affected_group ON forecast_pairs(seed,day)');c.commit()
    c.execute('CREATE TABLE summary AS SELECT seed,day,COUNT(*) AS n,SUM(ABS(repaired-target)) AS s FROM forecast_pairs WHERE valid=1 GROUP BY seed,day')
    c.execute('CREATE UNIQUE INDEX summary_key ON summary(seed,day)');c.commit()
    plan=list(c.execute('EXPLAIN QUERY PLAN '+QUERY));(OUT/'query_plan.json').write_text(json.dumps(plan,indent=2))
    # Query timing in paired blocks is descriptive for this one fixed database.
    rng=np.random.default_rng(7038);timings=[]
    for block in range(10):
        for name in rng.permutation(['raw','summary']):
            q='SELECT seed,day,COUNT(*),SUM(ABS(repaired-target)) FROM forecast_pairs WHERE valid=1 GROUP BY seed,day ORDER BY seed,day' if name=='raw' else 'SELECT seed,day,n,s FROM summary ORDER BY seed,day'
            t=time.perf_counter();r=list(c.execute(q));ms=(time.perf_counter()-t)*1000
            assert close(r,[[v[0],v[1],v[2],v[6]] for v in oracle]);timings.append({'block':block,'query':name,'wall_ms':ms})
    pd.DataFrame(timings).to_csv(OUT/'replay_timings.csv',index=False)
    # WAL holds a historical read snapshot while a writer atomically replaces an
    # affected aggregate and revision token. This is a consistency test only.
    reader=sqlite3.connect(DB,isolation_level=None);checks=[]
    for case in range(20):
        seed=[7,17,27][case%3];day=case%7
        reader.execute('BEGIN');oldrev=reader.execute('SELECT revision FROM version_state').fetchone()[0]
        old=reader.execute('SELECT n,s FROM summary WHERE seed=? AND day=?',(seed,day)).fetchone()
        ids=[r[0] for r in c.execute('SELECT rowid FROM forecast_pairs WHERE seed=? AND day=? AND valid=1 ORDER BY origin,horizon,node LIMIT 100',(seed,day))]
        c.execute('BEGIN IMMEDIATE');c.executemany('UPDATE forecast_pairs SET repaired=repaired+0.25 WHERE rowid=?',[(i,) for i in ids])
        cur=c.execute('SELECT COUNT(*),SUM(ABS(repaired-target)) FROM forecast_pairs WHERE seed=? AND day=? AND valid=1',(seed,day)).fetchone()
        c.execute('UPDATE summary SET n=?,s=? WHERE seed=? AND day=?',(*cur,seed,day));c.execute('UPDATE version_state SET revision=revision+1');c.commit()
        still=reader.execute('SELECT n,s FROM summary WHERE seed=? AND day=?',(seed,day)).fetchone();revstill=reader.execute('SELECT revision FROM version_state').fetchone()[0]
        assert old==still and oldrev==revstill
        reader.execute('COMMIT');new=reader.execute('SELECT n,s FROM summary WHERE seed=? AND day=?',(seed,day)).fetchone();newrev=reader.execute('SELECT revision FROM version_state').fetchone()[0]
        full=list(c.execute('SELECT seed,day,COUNT(*),SUM(ABS(repaired-target)) FROM forecast_pairs WHERE valid=1 GROUP BY seed,day ORDER BY seed,day'))
        cached=list(c.execute('SELECT seed,day,n,s FROM summary ORDER BY seed,day'));assert close(full,cached)
        assert close(new,cur) and newrev==oldrev+1
        checks.append({'case':case,'old_revision':oldrev,'new_revision':newrev,'changed_cells':len(ids),'held_reader_unchanged':True,'next_snapshot_matches':True,'incremental_equals_full':True,'stale_token_rejected':oldrev!=newrev})
    reader.close();c.close()
    (OUT/'snapshot_checks.json').write_text(json.dumps(checks,indent=2))
    result={'real_dataset':'METR-LA','paired_keys':tot,'forecast_values_compared':tot*4,'SQL_groups':len(actual),'SQL_groups_equal_numpy':True,'valid_cells_per_seed':int(valid.sum()),'query_timing_blocks':10,'snapshot_cases':20,'snapshot_pass':len(checks),'corrections_per_case':100,'correction_scope':'controlled +0.25 mph to repaired prediction only, not observed target change','sqlite_version':sqlite3.sqlite_version,'cache_tolerance_mph':1e-4,'database_bytes':DB.stat().st_size}
    (OUT/'replay_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
