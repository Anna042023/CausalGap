"""Executed source-level regression, NOT a traffic-accuracy experiment.

The original supplement is never modified. Synthetic input/graph fixtures exercise
its exact DSAIWrapper and STGCN classes. Every variant uses identical fixed random
weights, and cold recomputation supplies the numerical oracle. A byte-key subclass
is a separately labelled repair for future runs, not a reconstruction of history.
"""
from __future__ import annotations
import os, sys, json, hashlib, time, csv, sqlite3, platform, copy
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent
sys.path.insert(0,str(SRC))
from causalgap.models.dsai import DSAIWrapper
from causalgap.models.stgcn import STGCN, cheb_polynomials

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def exact_key(a):
    a=np.ascontiguousarray(a)
    if not np.isfinite(a).all(): raise ValueError('non-finite graph')
    return (a.shape,a.dtype.str,a.tobytes())

class ByteFrontend(DSAIWrapper):
    @staticmethod
    def _adj_key(a): return exact_key(a)

class ByteSTGCN(STGCN):
    def _get_cheb(self,adj,device):
        a=adj.detach().cpu().numpy() if torch.is_tensor(adj) else np.asarray(adj)
        key=exact_key(a)
        if self._cached_key != key or self._cheb.numel()==0:
            self._cheb=torch.from_numpy(cheb_polynomials(a,self.ks)).float().to(device)
            self._cached_key=key
        return self._cheb

def graphs(seed,n=32):
    rng=np.random.default_rng(seed); a=np.zeros((n,n),np.float32)
    for i in range(n): a[i,(i+1)%n]=a[(i+1)%n,i]=1
    poss=[(i,j) for i in range(n) for j in range(i+1,n) if not a[i,j]]
    for i in rng.choice(len(poss),size=n,replace=False):
        u,v=poss[i];a[u,v]=a[v,u]=1
    b=a.copy();removed=[];added=[]
    # Fixed nodes: this is an edge-set change, never a relabelling of observations.
    existing=np.argwhere(np.triu(a,1)>0)
    for idx in rng.choice(len(existing),size=8,replace=False):
        u,v=map(int,existing[idx]); b[u,v]=b[v,u]=0;removed.append([u,v])
    missing=np.argwhere(np.triu(1-a-np.eye(n),1)>0)
    for idx in rng.choice(len(missing),size=8,replace=False):
        u,v=map(int,missing[idx]); b[u,v]=b[v,u]=1;added.append([u,v])
    assert DSAIWrapper._adj_key(a)==DSAIWrapper._adj_key(b)
    assert exact_key(a)!=exact_key(b)
    return a,b,removed,added

def build(a,weights,outer,inner):
    cls=ByteSTGCN if inner else STGCN
    base=cls(a.shape[0],12,12,dropout=0.0)
    wrap=(ByteFrontend if outer else DSAIWrapper)(base,a)
    wrap.load_state_dict(weights);wrap.eval();return wrap

def evaluate(model,x,a):
    with torch.no_grad():
        o=model.inference_forward(x,a,12)
    return o.pred.detach().cpu().numpy(),o.representation.detach().cpu().numpy()

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    plan={'scope':'source-level deterministic regression, synthetic fixtures, no fitted forecast accuracy',
          'n':32,'history':12,'horizon':12,'batch':2,'seeds':list(range(20)),
          'tolerance':1e-5,'variants':['legacy','frontend_only','predictor_only','both'],
          'graph_revision':'8 undirected edges removed, 8 distinct baseline-absent edges added',
          'weights':'fixed random initialization, no training or model selection',
          'sqlite':'saved model outputs reloaded and compared to cold output',
          'source_sha256':{p:sha(SRC/p) for p in ['causalgap/models/dsai.py','causalgap/models/stgcn.py','causalgap/spatial.py']}}
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2))
    db=sqlite3.connect(out/'replay.sqlite');db.executescript('DROP TABLE IF EXISTS predictions; CREATE TABLE predictions (fixture INT, variant TEXT, atom INT, prediction REAL, reference REAL, PRIMARY KEY(fixture,variant,atom));')
    rows=[]; stage=[]; controls=[]
    for seed in plan['seeds']:
        torch.manual_seed(20261107+seed);np.random.seed(20261107+seed)
        a,b,removed,added=graphs(seed)
        x=torch.from_numpy(np.random.default_rng(2000+seed).normal(size=(2,12,32,1)).astype('f4'))
        prototype=DSAIWrapper(STGCN(32,12,12,dropout=0.0),a)
        weights=copy.deepcopy(prototype.state_dict())
        torch.save(weights,out/f'weights_{seed:02d}.pt')
        cold=build(a,weights,False,False)
        t0=time.perf_counter();p0,z0=evaluate(cold,x,a);stage.append((seed,'cold_base',time.perf_counter()-t0))
        cold=build(a,weights,False,False)
        t0=time.perf_counter();gold,zgold=evaluate(cold,x,b);stage.append((seed,'cold_changed',time.perf_counter()-t0))
        arrays={'a':a,'b':b,'x':x.numpy(),'cold_base':p0,'cold_changed':gold,'z_base':z0,'z_changed':zgold,'removed':np.array(removed),'added':np.array(added)}
        for name,outer,inner in [('legacy',False,False),('frontend_only',True,False),('predictor_only',False,True),('both',True,True)]:
            model=build(a,weights,outer,inner);evaluate(model,x,a)
            t0=time.perf_counter();warm,zw=evaluate(model,x,b);duration=time.perf_counter()-t0
            err=float(np.max(np.abs(warm.astype(np.float64)-gold.astype(np.float64))));delta=float(np.max(np.abs(warm-p0)))
            rows.append({'seed':seed,'variant':name,'max_prediction_error':err,'max_change_from_base':delta,'cold_change_from_base':float(np.max(np.abs(gold-p0))),'wrong':int(err>plan['tolerance']),'wall_ms':duration*1000})
            arrays[name]=warm
            db.executemany('INSERT INTO predictions VALUES(?,?,?,?,?)',[(seed,name,i,float(v),float(gold.reshape(-1)[i])) for i,v in enumerate(warm.reshape(-1))])
            if name=='both':
                repeat,_=evaluate(model,x,b)
                back,_=evaluate(model,x,a)
                controls.append({'seed':seed,'identical_repeat_max':float(np.max(np.abs(repeat-warm))),'return_to_A_max':float(np.max(np.abs(back-p0)))})
        np.savez_compressed(out/f'fixture_{seed:02d}.npz',**arrays)
        db.commit()
        print('fixture',seed,'done',flush=True)
    with (out/'cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'controls.json').write_text(json.dumps(controls,indent=2))
    # SQL replay from persisted predictions, exact double values of float32 outputs.
    q='SELECT fixture, variant, MAX(ABS(prediction-reference)) FROM predictions GROUP BY fixture,variant ORDER BY fixture,variant'
    sql={ (s,v):e for s,v,e in db.execute(q)}
    for r in rows: assert abs(sql[r['seed'],r['variant']]-r['max_prediction_error'])<1e-12
    summary={v:{'cases':20,'mismatches':sum(r['wrong'] for r in rows if r['variant']==v),'max_abs_error':max(r['max_prediction_error'] for r in rows if r['variant']==v),'median_max_abs_error':float(np.median([r['max_prediction_error'] for r in rows if r['variant']==v]))} for v in plan['variants']}
    summary['saved_prediction_atoms']=db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    summary['SQL_replay_groups_matched']=len(sql)
    summary['controls_pass']=sum(c['identical_repeat_max']<=1e-5 and c['return_to_A_max']<=1e-5 for c in controls)
    summary['environment']={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'sqlite':sqlite3.sqlite_version,'machine':platform.machine()}
    db.close();(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest_sha256.json'}
    (ROOT/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
