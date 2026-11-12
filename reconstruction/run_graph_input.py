#!/usr/bin/env python3
"""CPU-only graph-input analysis on fixed observed METR-LA targets.
Inputs are the saved METR-LA windows and three checkpoint files.
"""
import json,hashlib,sys,csv,time,platform,warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'real_replay'))
from train_metr_la import make_model
OUT=BASE/'reconstruction'/'graph_inputs';OUT.mkdir(exist_ok=True)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def changed_graph(A,frac,seed=8431):
    rng=np.random.default_rng(seed); A=A.copy();n=len(A)
    edges=np.argwhere(np.triu(A,1)>0)
    absent_original=np.argwhere(np.triu(1-A-np.eye(n),1)>0)
    k=max(1,round(len(edges)*frac))
    take=edges[rng.choice(len(edges),k,replace=False)]
    for i,j in take:A[i,j]=A[j,i]=0
    add=absent_original[rng.choice(len(absent_original),k,replace=False)]
    for i,j in add:A[i,j]=A[j,i]=1
    assert np.count_nonzero(np.triu(A,1))==len(edges)
    return A,k

def eval_one(model,x,adj):
    pp=[];rr=[]
    with torch.no_grad():
        for i in range(0,len(x),24):
            z=model.inference_forward(torch.from_numpy(x[i:i+24]),adj,12)
            pp.append(z.pred[...,0].cpu().numpy().astype(np.float32))
            rr.append(z.representation[:,0].cpu().numpy().astype(np.float32))
    return np.concatenate(pp),np.concatenate(rr)

def new_h1(seed_data,valid):
    # Hold out whole checkpoints (model seeds), no strong outcomes in training folds.
    # Samples are correlated and results are descriptive only.
    dataset=[]
    for seed,d in sorted(seed_data.items()):
        x5=d['weak05_pred'];x10=d['weak10_pred'];b=d['base_pred'];st=d['strong_pred']
        truth=d['target']; mask=valid
        def mae(pred):
            z=np.abs(pred-truth).astype(float)
            z[~mask]=np.nan
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                return np.nanmean(z,axis=1)
        e05=mae(x5);e10=mae(x10);eb=mae(b);est=mae(st)
        z05=np.linalg.norm(d['weak05_rep']-d['base_rep'],axis=-1)/(np.linalg.norm(d['base_rep'],axis=-1)+1e-7)
        z10=np.linalg.norm(d['weak10_rep']-d['base_rep'],axis=-1)/(np.linalg.norm(d['base_rep'],axis=-1)+1e-7)
        # Per-window and sensor weak errors and response. Strong outcomes are targets only.
        f0=np.stack([e05-eb,e10-eb],axis=-1)
        f1=np.stack([e05-eb,e10-eb,z05,z10],axis=-1)
        y=est-eb
        ok=np.isfinite(f1).all(-1)&np.isfinite(y)
        dataset.append((seed,f0[ok],f1[ok],y[ok],int(ok.sum())))
    rows=[]
    for seed,f0,f1,y,n in dataset:
        tr=[z for z in dataset if z[0]!=seed]
        a=np.concatenate([z[1] for z in tr]);c=np.concatenate([z[2] for z in tr]);tar=np.concatenate([z[3] for z in tr]);
        for label,xtest,xtrain in [('weak_errors',f0,a),('weak_errors_plus_response',f1,c)]:
            model=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
            model.fit(xtrain,tar)
            py=model.predict(xtest)
            rows.append({'held_out_checkpoint_seed':seed,'configuration':label,'n':n,'mean_absolute_target_error':float(np.mean(np.abs(py-y))),'mean_squared_target_error':float(np.mean((py-y)**2)),'target_mean':float(np.mean(y))})
    return rows

def new_h2(seed_data,valid):
    # Frozen predictors. Label-dependent heads trained only on first 56 test windows
    # of the archived *observed* records under weak graph changes.
    # Outcome is the SAME observed future under graph-input stress.
    rows=[]
    for seed,d in sorted(seed_data.items()):
        truth=d['target']; b=d['base_pred']; mask=valid
        feat=[]; ctx=[]; trg=[]
        for level in ('weak05','weak10'):
            pr=d[f'{level}_pred']; z=d[f'{level}_rep'];base_rep=d['base_rep']
            resp=np.linalg.norm(z-base_rep,axis=-1)/(np.linalg.norm(base_rep,axis=-1)+1e-7)
            for i in range(56):
                preds=pr[i].T.reshape(-1) # sensor, horizon
                responses=np.repeat(resp[i],12)
                base_context=np.repeat(np.linalg.norm(base_rep[i],axis=-1),12)
                v=mask[i].T.reshape(-1)
                # Same two-feature ridge capacity; baseline uses graph-independent representation norm.
                feat.append(np.stack([preds,responses],axis=-1)[v])
                ctx.append(base_context[v])
                trg.append((truth[i]-pr[i]).T.reshape(-1)[v])
        train=np.concatenate(feat);label=np.concatenate(trg)
        # Keep protocol intentionally simple, equal input dimension and regularization.
        rng=np.random.default_rng(8431+seed)
        m_base=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
        m_resp=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
        m_shuf=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
        basex=train.copy();basex[:,1]=np.concatenate(ctx)
        shufx=train.copy();rng.shuffle(shufx[:,1])
        m_base.fit(basex,label);m_resp.fit(train,label);m_shuf.fit(shufx,label)
        pr=d['strong_pred'][56:];ref=truth[56:];m=mask[56:]
        rep=np.linalg.norm(d['strong_rep'][56:]-d['base_rep'][56:],axis=-1)/(np.linalg.norm(d['base_rep'][56:],axis=-1)+1e-7)
        px=pr.transpose(0,2,1).reshape(-1)
        rx=np.repeat(rep,12,axis=1).reshape(-1)
        mm=m.transpose(0,2,1).reshape(-1)
        true=ref.transpose(0,2,1).reshape(-1)[mm]
        pred=px[mm]; xx=np.stack([pred,rx[mm]],axis=-1)
        xbase=xx.copy();xbase[:,1]=np.repeat(np.linalg.norm(d['base_rep'][56:],axis=-1),12,axis=1).reshape(-1)[mm]
        xshuf=xx.copy();rng.shuffle(xshuf[:,1])
        for tag,est in [('uncorrected',pred),('matched_no_response',pred+m_base.predict(xbase)),('matched_plus_response',pred+m_resp.predict(xx)),('shuffled_response',pred+m_shuf.predict(xshuf))]:
            rows.append({'checkpoint_seed':seed,'head':tag,'n':len(true),'observed_target_mae_mph':float(np.abs(est-true).mean()),'calibration_windows':56,'heldout_windows':112})
    return rows

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    z=np.load(BASE/'real_replay/outputs/test_inputs.npz'); X=np.asarray(z['x'],np.float32);target=np.asarray(z['target'],np.float32);valid=np.asarray(z['valid'],bool);A=np.asarray(z['baseline_graph'],np.float32)
    print('shape',X.shape,target.shape,flush=True)
    graphs={};edits={}
    for level,p in [('weak05',.05),('weak10',.10),('strong',.20)]: graphs[level],edits[level]=changed_graph(A,p)
    data={};hashes={};start=time.time()
    for seed in (7,17,27):
        ck=BASE/f'real_replay/outputs/checkpoint_{seed}.pt'
        payload=torch.load(ck,map_location='cpu',weights_only=False)
        hashes[str(seed)]=sha(ck)
        m=make_model(A,True);m.load_state_dict(payload['weights']);m.eval()
        dd={'target':target}
        for name,adj in [('base',A),*graphs.items()]:
            m=make_model(A,True);m.load_state_dict(payload['weights']);m.eval() # explicitly fresh model to rule out cache bleed
            pred,rep=eval_one(m,X,adj)
            # Model normalized predictions must be restored to units of mph.
            mean,std=payload['plan']['normalization']
            dd[f'{name}_pred']=(pred*std+mean).astype(np.float32)
            dd[f'{name}_rep']=rep.astype(np.float32)
            print('run',seed,name,pred.shape,rep.shape,'elapsed',round(time.time()-start,1),flush=True)
        data[seed]=dd
        np.savez_compressed(OUT/f'graph_input_seed{seed}.npz',**{k:np.asarray(v) for k,v in dd.items()},valid=valid)
    np.savez_compressed(OUT/'graph_versions.npz',baseline=A,**graphs)
    h1=new_h1(data,valid);h2=new_h2(data,valid)
    with (OUT/'weak_to_strong.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,h1[0].keys());w.writeheader();w.writerows(h1)
    with (OUT/'calibration.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,h2[0].keys());w.writeheader();w.writerows(h2)
    manifest={'id':'R18-new-cpu-pilot','source':'R17 saved observed METR-LA test inputs and 3 fitted checkpoints','NOT':'2026 historical H1/H2; does not infer causal road-closure outcomes','status':'NEW_REEVALUATION','checkpoint_sha256':hashes,'script_sha256':sha(Path(__file__)), 'base_input_sha256':sha(BASE/'real_replay/outputs/test_inputs.npz'),'model_count':3,'architecture':'STGCN/DSAI only','dataset':'METR-LA observed unmodified targets','n_original_windows':len(X),'n_sensors':X.shape[2],'n_horizons':12,'graph_rewiring_fractions':{'weak05':.05,'weak10':.1,'strong':.2},'edges_rewired':edits,'graph_random_seed':8431,'h1_protocol':'leave-checkpoint-seed-out ridge on weak graph error and representation response; strong graph errors held out as target; same observed future','h2_protocol':'frozen predictor, two-feature ridge calibration on first 56 observed test windows under weak graph revisions; baseline uses baseline representation norm and response variant uses relative graph response; later 112 windows tested under strong revision','limitations':['model family count 1','three checkpoints not 21 model families','H1 and H2 not original historical protocol','reused held-out period for calibration, explicitly time split','targets observed under unchanged physical world; do not interpret as intervention effects'],'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'cpu':platform.processor()},'outputs':{}}
    for p in (OUT/'weak_to_strong.csv',OUT/'calibration.csv',OUT/'graph_versions.npz',*(OUT.glob('graph_input_seed*.npz'))):
        manifest['outputs'][p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('H1',json.dumps(h1,indent=2),flush=True)
    print('H2',json.dumps(h2,indent=2),flush=True)
    print('DONE elapsed',time.time()-start,flush=True)
if __name__=='__main__':main()
