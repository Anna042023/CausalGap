"""New real-observation replay; not a reconstruction of historical CausalGap runs.
Uses the supplied STGCN/DSAI source and separately identified exact-byte caches.
No causal outcomes are inferred: targets are unchanged under topology-input stress.
"""
from __future__ import annotations
import sys, os, json, time, hashlib, copy, csv, sqlite3, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cache_replay'))
from run_regression import ByteFrontend, ByteSTGCN, DSAIWrapper, STGCN, exact_key, SRC
OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def make_model(a, fixed=True):
    cls=ByteSTGCN if fixed else STGCN
    base=cls(a.shape[0],12,12,ks=3,kt=3,dropout=0.0,blocks=((1,8,16),(16,8,16)))
    return (ByteFrontend if fixed else DSAIWrapper)(base,a)
def evaluate(m,x,adj,mean,std):
    m.eval();ys=[]
    with torch.no_grad():
        for i in range(0,len(x),24):
            y=m.inference_forward(torch.from_numpy(x[i:i+24]),adj,12).pred[...,0].numpy()
            ys.append((y.astype('f8')*std+mean).astype('f4'))
    return np.concatenate(ys)
def train():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    raw_path=Path(os.environ.get('METR_LA_PATH',str(HERE/'data/metr-la.h5')))
    # A full local raw copy is retained for traceability and reproducibility.
    raw=pd.read_hdf(raw_path); v=raw.to_numpy(dtype='f4'); valid=np.isfinite(v)&(v>0)
    T,N=v.shape;tr=int(T*.6);ve=int(T*.8)
    mean=float(v[:tr][valid[:tr]].astype('f8').mean());std=float(v[:tr][valid[:tr]].astype('f8').std())
    filled=pd.DataFrame(np.where(valid,v,np.nan)).ffill().fillna(mean).to_numpy(dtype='f4')
    z=((filled-mean)/std).astype('f4')
    corr=np.corrcoef(z[:tr].astype('f8'),rowvar=False);np.fill_diagonal(corr,-np.inf)
    adj=np.zeros((N,N),'f4')
    for i in range(N):adj[i,np.argsort(corr[i],kind='stable')[-5:]]=1
    adj=np.maximum(adj,adj.T);np.fill_diagonal(adj,0)
    # Same number of binary edges, fixed sensor order, 10% replaced.
    rr=np.random.default_rng(7031);edges=np.argwhere(np.triu(adj,1)>0)
    absent=np.argwhere(np.triu(1-adj-np.eye(N),1)>0);k=max(1,len(edges)//10);changed=adj.copy()
    removed=edges[rr.choice(len(edges),k,replace=False)];added=absent[rr.choice(len(absent),k,replace=False)]
    for i,j in removed:changed[i,j]=changed[j,i]=0
    for i,j in added:changed[i,j]=changed[j,i]=1
    assert DSAIWrapper._adj_key(adj)==DSAIWrapper._adj_key(changed)
    assert exact_key(adj)!=exact_key(changed)
    train_starts=np.arange(12,tr-11,12);val_starts=np.arange(tr+12,ve-11,12);test_starts=np.arange(T-7*288,T-11,12)
    plan={'scope':'NEW real-observation input-topology stress replay; unchanged targets; no physical-closure effect; not historical replication',
      'dataset':'METR-LA DCRNN authors release','provider':'https://github.com/liyaguang/DCRNN',
      'download_file_id':'1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC','raw_sha256':sha(raw_path),
      'shape':[T,N],'timestamps':[str(raw.index[0]),str(raw.index[-1])],'timezone':'unspecified in provider HDF5',
      'split_boundaries':[tr,ve,T],'validity_rule':'finite and >0, mask target cells; history forward filled, leading values use training mean',
      'normalization':[mean,std],'graph':'symmetric union of training-only top-5 correlation neighbours, binary, fixed sensor order',
      'graph_edges':len(edges),'rewired_edges':k,'rewire_seed':7031,'history':12,'horizon':12,'stride':12,
      'train_windows':len(train_starts),'validation_windows':len(val_starts),'test_windows':len(test_starts),
      'test_first_target':str(raw.index[test_starts[0]]),'test_last_target':str(raw.index[-1]),
      'seeds':[7,17,27],'epochs':3,'selection':'lowest masked validation MAE among three epochs','batch':32,'lr':0.001,
      'blocks':[[1,8,16],[16,8,16]],'torch_threads':2,'cache_test_tolerance_mph':0.0001,
      'source_hashes':{p:sha(SRC/p) for p in ['causalgap/models/stgcn.py','causalgap/models/dsai.py','causalgap/spatial.py']}}
    (HERE/'plan.json').write_text(json.dumps(plan,indent=2))
    def windows(starts):
        x=np.stack([z[t-12:t,:,None] for t in starts]);y=np.stack([v[t:t+12] for t in starts]);mask=np.stack([valid[t:t+12] for t in starts])
        return x,y,mask
    xtr,ytr,mtr=windows(train_starts);xva,yva,mva=windows(val_starts);xte,yte,mte=windows(test_starts)
    np.savez_compressed(OUT/'test_inputs.npz',x=xte,target=yte,valid=mte,origins_ns=raw.index[test_starts-1].asi8,sensor_ids=raw.columns.astype(str).to_numpy(dtype='U'),baseline_graph=adj,changed_graph=changed,removed=removed,added=added)
    log=[];results=[]
    for seed in ([int(os.environ.get('CAUSALGAP_SEED', os.environ.get('R7_SEED')))] if ('CAUSALGAP_SEED' in os.environ or 'R7_SEED' in os.environ) else plan['seeds']):
        torch.manual_seed(seed);np.random.seed(seed);model=make_model(adj,True)
        opt=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');weights=None
        plan['parameters']=sum(p.numel() for p in model.parameters());plan['frontend_parameters']=model.residual.weight.numel()
        for epoch in range(3):
            model.train();t0=time.perf_counter();order=np.random.default_rng(seed*100+epoch).permutation(len(xtr));ls=[]
            for st in range(0,len(order),32):
                ids=order[st:st+32];xx=torch.from_numpy(xtr[ids]);yy=torch.from_numpy((ytr[ids]-mean)/std);mm=torch.from_numpy(mtr[ids])
                pred=model.training_forward(xx,adj).pred[...,0];loss=torch.abs(pred-yy)[mm].mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();ls.append(float(loss.detach()))
            pv=evaluate(model,xva,adj,mean,std);mae=float(np.abs(pv.astype('f8')-yva)[mva].mean())
            if mae<best:best=mae;weights=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            item={'seed':seed,'epoch':epoch+1,'train_norm_mae':float(np.mean(ls)),'validation_mae_mph':mae,'wall_s':time.perf_counter()-t0};log.append(item);print(item,flush=True)
        ck=OUT/f'checkpoint_{seed}.pt';torch.save({'weights':weights,'plan':plan,'chosen_epoch':best_epoch,'validation_mae':best},ck)
        def loaded(fixed=True):
            m=make_model(adj,fixed);m.load_state_dict(weights);m.eval();return m
        cold=loaded();base=evaluate(cold,xte,adj,mean,std)
        cold=loaded();gold=evaluate(cold,xte,changed,mean,std)
        legacy=loaded(False);evaluate(legacy,xte[:1],adj,mean,std);stale=evaluate(legacy,xte,changed,mean,std)
        repaired=loaded();evaluate(repaired,xte[:1],adj,mean,std);warm=evaluate(repaired,xte,changed,mean,std)
        diff=np.abs(warm.astype('f8')-gold);bad=np.abs(stale.astype('f8')-gold)
        r={'seed':seed,'selected_epoch':best_epoch,'base_mae_mph':float(np.abs(base.astype('f8')-yte)[mte].mean()),'cold_changed_mae_mph':float(np.abs(gold.astype('f8')-yte)[mte].mean()),'legacy_changed_mae_mph':float(np.abs(stale.astype('f8')-yte)[mte].mean()),'legacy_max_abs_discrepancy_mph':float(bad.max()),'legacy_mean_abs_discrepancy_mph':float(bad[mte].mean()),'legacy_affected_valid_cells':int((bad[mte]>1e-4).sum()),'repaired_max_abs_discrepancy_mph':float(diff.max()),'valid_cells':int(mte.sum()),'checkpoint_sha256':sha(ck)}
        results.append(r);np.savez_compressed(OUT/f'predictions_{seed}.npz',baseline=base,cold_changed=gold,legacy_changed=stale,repaired_changed=warm)
        pd.DataFrame([r]).to_csv(OUT/f'prediction_summary_{seed}.csv',index=False)
        pd.DataFrame([q for q in log if q['seed']==seed]).to_csv(OUT/f'training_log_{seed}.csv',index=False)
        print('RESULT',json.dumps(r),flush=True)
    (HERE/'plan.json').write_text(json.dumps(plan,indent=2))
    pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('training_log_*.csv'))],ignore_index=True).to_csv(OUT/'training_log.csv',index=False)
    pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('prediction_summary_*.csv'))],ignore_index=True).to_csv(OUT/'prediction_summary.csv',index=False)
    (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,'sqlite':sqlite3.sqlite_version,'numpy':np.__version__,'cpu':platform.processor(),'threads':torch.get_num_threads()},indent=2))
if __name__=='__main__':train()
