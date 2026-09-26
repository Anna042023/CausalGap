#!/usr/bin/env python3
"""CausalGap H2: actual frozen-predictor, equal-norm, information-matched steering.

Must be installed at unified/experiments/representation_steering/.
Reuses incremental_value synthetic generator, manifests,
pretrained checkpoints and the model factory. Does NOT retrain forecasting models.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[2]
H1DIR = CODE / 'experiments' / 'incremental_value'
for p in (str(H1DIR), str(CODE)):
    if p not in sys.path: sys.path.insert(0, p)

from evaluate_h1_job import add_model_args, synthetic_bundle  # noqa
from synthetic_env import simulate, windows, split_7_1_2  # noqa
from causalgap.evaluator import load_checkpoint, uniform_sample_indices  # noqa
from causalgap.models.factory import build_model  # noqa
from causalgap.models.registry import get_spec  # noqa
from causalgap.profiles import resolve as resolve_profile  # noqa
from causalgap.interventions import load_external_intervention  # noqa
from causalgap.utils import seed_everything, device_from_arg  # noqa
from core import (READOUT, SteeringHead, SteeringHeadFiLM, construct_features,
                     intervention_response, to_eval, equal_norm, masked_mae, freeze)
from additional_readouts import create_readout, TSLIB
from calibration import calibration_rollout, calibration_topologies
from protocol import MODEL_FAMILIES


def digest(s): return hashlib.sha256(str(s).encode('utf8')).hexdigest()


def deterministic_seed(*pieces): return int(digest('|'.join(map(str, pieces)))[:8], 16)


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def h1_env_path(exp, record):
    from protocol import safe_name
    actual = Path(str(record.path))
    if actual.exists(): return actual
    alt = (exp / 'environments' / 'synthetic' / 'interventions' /
           str(record.intervention_type) / safe_name(str(record.level)) /
           f'seed_{int(record.seed):04d}.npz')
    if not alt.exists(): raise FileNotFoundError(f'{actual} OR {alt}')
    return alt


def independent_calibration(bundle, exp, output, max_windows):
    return calibration_rollout(bundle,exp,output,max_windows)


def intervention_mask(kind, env, raw_y):
    mask = np.isfinite(raw_y)
    side = env.with_suffix('.json')
    meta = json.loads(side.read_text()) if side.exists() else {}
    if kind == 'node_isolation':
        isolated = meta.get('isolated_nodes', [])
        if not isolated: raise RuntimeError(f'{env}: missing isolated_nodes; cannot match IPE support')
        mask[:, :, np.asarray(isolated, np.int64), :] = False
    return mask


def prep_units(args, bundle, model, bridge, device, manifest, cal_state, run_dir):
    burn, x_init, u, eps, cal_x_raw, cal_y_raw, selected, cal_seed = cal_state
    sel_test = np.empty(0,dtype=np.int64) if (args.cal_only or args.head_cal_only) else uniform_sample_indices(len(bundle.test_x), args.max_strong_windows)

    A0 = torch.tensor(bundle.adj, dtype=torch.float32, device=device)
    baseline = {}
    for split, ix in [('cal', selected), ('test', sel_test)]:
        if (args.cal_only or args.head_cal_only) and split=='test': continue
        xsrc = bundle.scaler.transform(cal_x_raw).astype(np.float32) if split=='cal' else bundle.test_x[ix]
        x = torch.tensor(xsrc,dtype=torch.float32,device=device)
        with torch.no_grad():
            p, _ = bridge.forward(x, A0, bundle.pred_len)
        baseline[split] = to_eval(p, bundle.scaler, model).detach().cpu()
    out = []
    readout_identity_checked = False
    for rec in manifest.itertuples(index=False):
        if (args.cal_only or args.head_cal_only) and str(rec.phase)=='strong': continue
        if str(rec.phase) == 'weak' and int(rec.seed) not in args.cal_seeds: continue
        if str(rec.phase) == 'strong' and int(rec.seed) not in args.eval_seeds: continue
        if str(rec.phase) not in ('weak','strong'): continue
        env = h1_env_path(args.experiment_root, rec)
        if str(rec.phase) == 'weak':
            with np.load(env, allow_pickle=False) as z: adjk = np.array(z['adj'], dtype=np.float32)
            # The same calibration environments are reused by every predictor.
            # Cache ONLY selected Dcal windows (not the full simulator trajectory).
            out_root = (args.output_root or args.project_root/'experiment_results'/'representation_steering')
            cache_dir = out_root/'_calibration_cache'
            cache_dir.mkdir(parents=True,exist_ok=True)
            token = digest('|'.join([fingerprint(env),
                     fingerprint(args.experiment_root/'environments'/'synthetic'/'synthetic_metadata.json'),
                     str(cal_seed), str(selected.tolist())]))[:24]
            cache_file=cache_dir/f'{token}.npz'
            if cache_file.exists():
                with np.load(cache_file,allow_pickle=False) as cache:
                    rawx=np.asarray(cache['x'],np.float32)
                    rawy=np.asarray(cache['y'],np.float32)
                print(f'[H2 CACHE] {cache_file.name}',flush=True)
            else:
                traj = simulate(adjk, x_init, u, eps)
                vx, vy = windows(traj[burn:], bundle.history_len, bundle.pred_len)
                rawx, rawy = vx[selected], vy[selected]
                tmp=cache_dir/f'{token}.{os.getpid()}.npz'
                np.savez_compressed(tmp,x=rawx,y=rawy)
                os.replace(tmp,cache_file)
            xk = bundle.scaler.transform(rawx).astype(np.float32)
            support = intervention_mask(str(rec.intervention_type), env, rawy)
            y_eval = rawy
            split = 'cal'; ids = selected
        else:
            # Strong/Dtest: exactly the intervention-consistent outcome from H1.
            inter = load_external_intervention(env, bundle, kind=str(rec.intervention_type))
            if not inter.causal_valid or inter.leakage_detected:
                raise RuntimeError(f'Invalid held-out environment {env}')
            adjk = inter.adj
            xk = np.asarray(inter.x[sel_test], np.float32)
            y_eval = bundle.scaler.inverse(inter.y[sel_test]).astype(np.float32)
            support = np.asarray(inter.y_mask[sel_test], bool)
            split = 'test'; ids = sel_test
        x = torch.tensor(xk, dtype=torch.float32, device=device)
        ak = torch.tensor(adjk, dtype=torch.float32, device=device)
        yt = torch.tensor(y_eval, dtype=torch.float32, device=device)
        mask = torch.tensor(support, dtype=torch.bool, device=device)
        with torch.no_grad():
            z0_pred, z0 = bridge.forward(x, A0, bundle.pred_len)
            before, zt = bridge.forward(x, ak, bundle.pred_len)
            if not readout_identity_checked:
                # Replacing z by exactly z must reconstruct the SAME original prediction.
                # A detached/surrogate or wrong pre-head interface would fail this gate.
                identity, _ = bridge.forward(x, ak, bundle.pred_len, replacement=zt)
                if not torch.allclose(before, identity, rtol=1e-5, atol=1e-6):
                    raise RuntimeError('Frozen readout identity test failed: steering interface is invalid')
                readout_identity_checked = True
                print('[H2 GATE] frozen-readout identity PASS',flush=True)
        before = to_eval(before, bundle.scaler, model)
        b = baseline[split].to(device)
        # matched baseline target is Y0; node isolation uses identical support.
        baseline_y = cal_y_raw if split=='cal' else bundle.scaler.inverse(bundle.test_y[ids]).astype(np.float32)
        y0 = torch.tensor(baseline_y, dtype=torch.float32, device=device)
        baseline_mask = mask if str(rec.intervention_type) == 'node_isolation' else torch.isfinite(y0)
        ipe0 = float(masked_mae(b, y0, baseline_mask).detach().item())
        ipek = float(masked_mae(before, yt, mask).detach().item())
        std_y0 = float(y0[baseline_mask].std(unbiased=False).detach().item())
        rd = (ipek-ipe0) / (ipe0+.05*std_y0)
        dist = float(rec.environment_distance)
        response = intervention_response(z0, zt, dist)
        if not np.isfinite(rd): raise RuntimeError('Non-finite RD in '+str(env))
        # H1 consistency audit, possible only when the same Dtest sample set was used.
        if split == 'test' and len(ids) == len(bundle.test_x):
            csv = run_dir/'causalgap_results.csv'
            if csv.exists():
                d = pd.read_csv(csv)
                match = d[(d['intervention_type'].astype(str) == str(rec.intervention_type)) &
                          (d['paper_intensity_level'].astype(str) == str(rec.level)) &
                          (d['intervention_seed'].astype(int) == int(rec.seed))]
                if len(match) != 1: raise RuntimeError('H1 strong CSV uniqueness mismatch '+str(env))
                want = float(match.iloc[0]['IPEk'])
                if abs(want-ipek) > max(1e-4, abs(want)*5e-3):
                    raise RuntimeError(f'H1 strong IPE mismatch {want} versus reforward {ipek}: {env}')
        item = dict(ipe0=ipe0,split=split, kind=str(rec.intervention_type), level=str(rec.level),
                    seed=int(rec.seed), env=str(env), x=x.cpu(), ak=ak.cpu(), y=yt.cpu(),
                    mask=mask.cpu(), before=float(ipek), rd=rd, zt=zt.cpu(), z0=z0.cpu(),
                    response=response, indices=np.asarray(ids, np.int64).tolist())
        out.append(item)
        print(f'[H2 DATA] {split} {item["kind"]} {item["level"]} seed{item["seed"]} '
              f'N={len(x)} IPE={ipek:.6g} DMIS={response["DMIS"]:.6g}', flush=True)
    return out


def device_item(item, device):
    return tuple(item[k].to(device) for k in ('x','ak','y','mask','zt','z0'))


def to_features(item, device, add_dmis):
    return construct_features(item['zt'].to(device), item['z0'].to(device),
                              item['rd'], item['response'], add_dmis)


def standardize(train, val, test):
    concat = torch.cat([to_features(it, 'cpu', True) for it in train], dim=0)
    mu, sigma = concat.mean(0), concat.std(0, unbiased=False).clamp_min(1e-4)
    # Constant-zero column in matched baseline needs to stay zero after scaling.
    mu[-1] = 0.
    sigma[-1] = max(float(sigma[-1]), 1e-4)
    return mu, sigma


def batches_of(it, size):
    for st in range(0, len(it['x']), size): yield slice(st, min(st+size, len(it['x'])))


def head_loss(head, item, bridge, bundle, model, device, norm, args, add_dmis, shuffle_value=None, train=True):
    x, ak, y, mask, zt, z0 = device_item(item, device)
    f = construct_features(zt, z0, item['rd'], item['response'], add_dmis)
    if shuffle_value is not None: f[:, -1] = float(shuffle_value)
    mu, sigma = norm
    f = (f - mu.to(device)) / sigma.to(device)
    losses, totals = [], []
    for sl in batches_of(item, args.head_batch_size):
        delta = head(f[sl]).reshape_as(zt[sl])
        delta = equal_norm(delta, zt[sl], args.perturb_ratio)
        # This is actual forecast after perturbing the *original* final head input.
        pred, _ = bridge.forward(x[sl], ak, bundle.pred_len, replacement=zt[sl]+delta)
        pred = to_eval(pred, bundle.scaler, model)
        l = masked_mae(pred, y[sl], mask[sl])
        if train and args.teacher_weight > 0:
            if 'teacher' not in item:
                raise RuntimeError('Missing calibration-only teacher direction')
            teacher = item['teacher'][sl].to(device)
            raw_direction = delta.reshape(len(delta), -1)
            teacher = teacher.reshape(len(teacher), -1)
            cosine = torch.nn.functional.cosine_similarity(raw_direction, teacher, dim=1, eps=1e-8)
            l = l + args.teacher_weight * (1.0 - cosine.mean())
        if train:
            l.backward()
        losses.append(float(l.detach().item()) * int(mask[sl].sum().item()))
        totals.append(int(mask[sl].sum().item()))
    return sum(losses)/max(1,sum(totals))


def cache_calibration_teacher(items, bridge, bundle, model, device, args):
    """Compute local forecast-error gradients ONLY on Dcal x Iweak.

    No Dtest / Istrong outcomes enter training, early stopping or teacher.
    This is an exploratory gradient-distilled steering objective, not the
    original frozen H2 protocol. Both matched heads get the same teacher.
    """
    for item in items:
        x, ak, y, mask, zt, _ = device_item(item, device)
        teachers=[]
        for sl in batches_of(item, args.head_batch_size):
            with torch.enable_grad():
                pred, _ = bridge.forward(x[sl], ak, bundle.pred_len, probe=True)
                pred = to_eval(pred, bundle.scaler, model)
                grad = torch.autograd.grad(masked_mae(pred, y[sl], mask[sl]), bridge._probe)[0]
            if not torch.isfinite(grad).all():
                raise RuntimeError('Nonfinite calibration-only gradient teacher')
            teachers.append((-grad.detach()).cpu())
        item['teacher'] = torch.cat(teachers, dim=0)


def train_head(name, train, val, bridge, bundle, model, device, norm, args,
               add_dmis, shuffles):
    input_dim = to_features(train[0], 'cpu', True).shape[-1]
    output_dim = int(np.prod(train[0]['zt'].shape[1:]))
    torch.manual_seed(args.head_seed)
    cls = SteeringHead if args.head_arch == 'mlp' else SteeringHeadFiLM
    head = cls(input_dim, output_dim, hidden=args.hidden).to(device)
    optim = torch.optim.Adam(head.parameters(), lr=args.head_lr, weight_decay=args.head_weight_decay)
    best, best_state, bad = float('inf'), None, 0
    for epoch in range(1,args.head_epochs+1):
        head.train()
        rng = np.random.default_rng(args.head_seed+epoch)
        for i in rng.permutation(len(train)):
            item = train[int(i)]
            optim.zero_grad(set_to_none=True)
            head_loss(head,item,bridge,bundle,model,device,norm,args,add_dmis,
                      shuffle_value=shuffles.get(id(item)) if shuffles else None,train=True)
            optim.step()
        head.eval()
        with torch.no_grad():
            risk = np.mean([head_loss(head,it,bridge,bundle,model,device,norm,args,add_dmis,
                                      shuffle_value=shuffles.get(id(it)) if shuffles else None,
                                      train=False) for it in val])
        print(f'[H2 TRAIN] {name} epoch={epoch} cal-val IPE={risk:.7f}',flush=True)
        if risk < best - args.min_delta:
            best, bad = risk, 0
            best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else: bad += 1
        if bad >= args.head_patience: break
    if best_state is None: raise RuntimeError('No valid steering-head checkpoint')
    head.load_state_dict(best_state)
    head.eval()
    return head, dict(val_ipe=best,epochs=epoch)


def permuted_dmis(items, seed):
    if not items: return {}
    vals = [it['response']['DMIS'] for it in items]
    rng=np.random.default_rng(seed)
    return {id(it):float(v) for it,v in zip(items, rng.permutation(vals))}


def evaluate_head(head, item, bridge, bundle, model, device, norm, args, add_dmis, shuffled=None):
    x, ak, y, mask, zt, z0 = device_item(item,device)
    f=construct_features(zt,z0,item['rd'],item['response'],add_dmis)
    if shuffled is not None: f[:,-1]=float(shuffled)
    mu,sigma=norm
    f=(f-mu.to(device))/sigma.to(device)
    preds=[]
    with torch.no_grad():
        for sl in batches_of(item,args.head_batch_size):
            d=equal_norm(head(f[sl]).reshape_as(zt[sl]),zt[sl],args.perturb_ratio)
            p,_=bridge.forward(x[sl],ak,bundle.pred_len,replacement=zt[sl]+d)
            preds.append(to_eval(p,bundle.scaler,model))
    return float(masked_mae(torch.cat(preds,0),y,mask))


def evaluate_gradient(item,bridge,bundle,model,device,args):
    x,ak,y,mask,zt,z0=device_item(item,device)
    preds=[]
    for sl in batches_of(item,args.head_batch_size):
        # The gradient direction is an explicitly label-oracle auxiliary control.
        with torch.enable_grad():
            p,_=bridge.forward(x[sl],ak,bundle.pred_len,probe=True)
            pred=to_eval(p,bundle.scaler,model)
            loss=masked_mae(pred,y[sl],mask[sl])
            grad=torch.autograd.grad(loss,bridge._probe)[0]
        if not torch.isfinite(grad).all(): raise RuntimeError('Nonfinite gradient steering')
        d=equal_norm(-grad.detach(),zt[sl],args.perturb_ratio)
        with torch.no_grad():
            p,_=bridge.forward(x[sl],ak,bundle.pred_len,replacement=zt[sl]+d)
        preds.append(to_eval(p,bundle.scaler,model).detach())
    return float(masked_mae(torch.cat(preds,0),y,mask))


def evaluate_random(item,bridge,bundle,model,device,args):
    x,ak,y,mask,zt,z0=device_item(item,device)
    rg=torch.Generator(device='cpu')
    rg.manual_seed(deterministic_seed(args.head_seed,item['kind'],item['level'],item['seed']))
    d=torch.randn(zt.shape,generator=rg).to(device)
    d=equal_norm(d,zt,args.perturb_ratio)
    with torch.no_grad():
        p,_=bridge.forward(x,ak,bundle.pred_len,replacement=zt+d)
    return float(masked_mae(to_eval(p,bundle.scaler,model),y,mask))


def main():
    p=argparse.ArgumentParser(description='True H2 frozen-readout steering on controlled Synthetic ONLY')
    add_model_args(p)
    p.add_argument('--cal-seeds',nargs='+',type=int,default=[1000,1001,1002,1003,1004])
    p.add_argument('--max-cal-windows',type=int,default=16)
    p.add_argument('--max-strong-windows',type=int,default=32)
    p.add_argument('--head-batch-size',type=int,default=8)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--head-lr',type=float,default=1e-3)
    p.add_argument('--head-weight-decay',type=float,default=1e-4)
    p.add_argument('--head-seed',type=int,default=2027)
    p.add_argument('--head-epochs',type=int,default=20)
    p.add_argument('--head-patience',type=int,default=5)
    p.add_argument('--min-delta',type=float,default=1e-6)
    p.add_argument('--perturb-ratio',type=float,default=.05)
    p.add_argument('--output-root',type=Path,default=None)
    p.add_argument('--cal-only',action='store_true',help='NO H1 strong/test read; produce calibration-only selection rows')
    p.add_argument('--head-cal-only',action='store_true',help='Train heads on Dcal weak and report held-out Dcal validation only; NEVER load strong test')
    p.add_argument('--head-arch',choices=['mlp','film'],default='film',help='Identical architecture for matched and +DMIS')
    p.add_argument('--teacher-weight',type=float,default=0.01,help='Calibration-only gradient-direction auxiliary loss; 0 disables it')
    p.add_argument('--selection-file',type=Path,default=None,help='freeze selected model before held-out evaluation')

    args=p.parse_args()
    if args.setting != 'synthetic': p.error('H2 is Synthetic ONLY; SUMO is excluded')
    if args.selection_file and args.cal_only: p.error('Selection file is for AFTER selection')
    if args.head_cal_only and args.cal_only: p.error('Choose only one calibration mode')
    if args.teacher_weight < 0: p.error('--teacher-weight must be >= 0')
    if not args.cal_only:
        if not args.selection_file: p.error('Provide --selection-file from select_h2_model.py before H2 test')
        selection=json.loads(args.selection_file.read_text())
        if args.model!=selection.get('selected_model'):
            p.error('Model must equal calibration-selected model in --selection-file')
        if args.model_seed not in selection.get('model_seeds',[]):
            p.error('Model seed not in frozen selection file')
        model_dir=args.experiment_root/'runs'/'synthetic'/args.model/'trackB'/f'seed{args.model_seed}'
        if args.cal_seeds!=selection['cal_topology_seeds'] or args.max_cal_windows!=selection['cal_windows'] or args.profile!=selection['profile']:
            p.error('Calibration settings differ from frozen selection')
        selected_cfg=selection.get('steering_config',{})
        actual_cfg=dict(head_epochs=args.head_epochs,head_patience=args.head_patience,
          head_batch_size=args.head_batch_size,hidden=args.hidden,
          max_strong_windows=args.max_strong_windows,perturb_ratio=args.perturb_ratio)
        if selected_cfg and selected_cfg!=actual_cfg:
            p.error('H2 hyperparameters differ from frozen selection lock')
        if fingerprint(model_dir/'checkpoint.pt')!=selection['checkpoint_sha256_by_seed'][str(args.model_seed)]:
            p.error('Frozen selection checkpoint SHA256 does not match')
    if args.model in {'Chronos','TimesFM','Moirai','Lag-Llama'}:
        p.error(f'{args.model}: uploaded worker exposes pooled detached representation but no verified native frozen-readout steering API; cannot generate paper-valid H2 data.')
    if args.model not in READOUT and args.model not in TSLIB and args.model not in ('DCRNN','STEVE'):
        p.error(f'{args.model} has no verified differentiable frozen-head bridge.')
    if args.train_missing: p.error('H2 may NEVER retrain a forecasting model; omit --train-missing')
    if args.rerun: p.error('H2 does not overwrite H1 runs; omit --rerun')
    if args.head_epochs<1 or args.perturb_ratio<=0: p.error('epochs>0 and perturb-ratio>0 required')
    if set(args.cal_seeds) & set(args.eval_seeds):p.error('Topology calibration seeds must be disjoint from strong test seeds')
    seed_everything(args.head_seed)
    device=device_from_arg(args.device)
    args.seed=int(args.model_seed)
    args.track='B'; args.dataset='SYNTHETIC-H1'
    if args.foundation_cache_dir is None: args.foundation_cache_dir=args.project_root/'cache'/'foundation_inference'
    bundle=synthetic_bundle(args.experiment_root)
    args=resolve_profile(args,args.model,history_len=bundle.history_len,num_nodes=bundle.num_nodes)
    model=build_model(args.model,bundle,args).to(device)
    run_dir=args.experiment_root/'runs'/'synthetic'/args.model/'trackB'/f'seed{args.model_seed}'
    ckpt=run_dir/'checkpoint.pt'
    if not ckpt.exists(): raise FileNotFoundError('Forecast checkpoint missing (do not retrain): '+str(ckpt))
    model=freeze(load_checkpoint(model,ckpt,device,bundle=bundle,allow_legacy=False))
    bridge=create_readout(model)
    h1_manifest=pd.read_csv(args.experiment_root/'environments'/'synthetic'/'manifest.csv')
    h1_manifest=h1_manifest[h1_manifest['phase'].eq('strong')]
    out_root=args.output_root or args.project_root/'experiment_results'/'representation_steering'
    cal_manifest=calibration_topologies(bundle,args.experiment_root,out_root,args.cal_seeds)
    manifest=pd.concat([cal_manifest,h1_manifest],ignore_index=True)
    if args.cal_only or args.head_cal_only:
        manifest=manifest[manifest['phase'].eq('weak')].copy()
    observed=sorted(set(h1_manifest['seed'].astype(int)))
    if not args.cal_only and not args.head_cal_only and not set(args.eval_seeds).issubset(observed):
        raise RuntimeError(f'Strong test seeds missing from H1 manifest; present {observed}')
    missing=[]
    for kind in ['edge_removal','weight_modification','node_isolation']:
        for s in args.cal_seeds:
            if len(manifest[(manifest.intervention_type==kind)&(manifest.phase=='weak')&(manifest.seed==s)])!=3:
                missing.append(('cal',kind,s))
        for s in ([] if (args.cal_only or args.head_cal_only) else args.eval_seeds):
            if len(manifest[(manifest.intervention_type==kind)&(manifest.phase=='strong')&(manifest.seed==s)])!=1:
                missing.append(('test',kind,s))
    if missing: raise RuntimeError('Incomplete H1 intervention design: '+repr(missing))
    out=out_root
    out=out/args.model/f'seed{args.model_seed}'
    out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    cal_state=independent_calibration(bundle,args.experiment_root,out_root,args.max_cal_windows)
    if not args.cal_only and cal_state[-1]!=selection['calibration_rollout_seed']:
        raise RuntimeError('Independent calibration rollout seed differs from frozen model selection')
    units=prep_units(args,bundle,model,bridge,device,manifest,cal_state,run_dir)
    if args.cal_only:
        from build_h1_units import empirical_calibrated_distance
        reference=pd.read_csv(args.experiment_root/'environments'/'synthetic'/'calibration_distance_reference.csv')
        rows=[]
        for item in units:
            kind=item['kind']
            ref=reference[reference.intervention_type.eq(kind)]['environment_distance'].to_numpy(float)
            calibrated_distance=empirical_calibrated_distance(item['response']['dT'],ref)
            dcal=item['response']['DZ']/max(calibrated_distance,1e-6)
            rows.append(dict(model=args.model,model_seed=args.model_seed,
                             model_family=MODEL_FAMILIES[args.model],
                             intervention_type=kind,level=item['level'],
                             calibration_topology_seed=item['seed'],
                             calibration_rollout_seed=cal_state[-1],
                             n_windows=len(item['x']),
                             IPE0=item['ipe0'],IPEk=item['before'],
                             RD=item['rd'],DMISraw=item['response']['DMIS'],
                             DMIScal=dcal,DZ=item['response']['DZ'],
                             dT=item['response']['dT'],
                             distance_calibrated=calibrated_distance,
                             source_environment=item['env'],
                             checkpoint_sha256=fingerprint(ckpt)))
        target=out/'calibration_units.csv'
        pd.DataFrame(rows).to_csv(target,index=False)
        (out/'calibration_manifest.json').write_text(json.dumps(
            dict(calibration_only=True,independent_rollout_seed=cal_state[-1],
                 temporal_source='independent synthetic rollout; no Dval or Dtest',
                 topology_seeds=args.cal_seeds,checkpoint_sha256=fingerprint(ckpt),
                 max_windows=args.max_cal_windows,model=args.model,model_seed=args.model_seed),indent=2))
        print(f'[H2 CAL ONLY] {target}; no held-out strong outcomes accessed',flush=True)
        return
    weak=[v for v in units if v['split']=='cal']
    test=[v for v in units if v['split']=='test']
    if len(weak)<6 or (not args.head_cal_only and not test): raise RuntimeError('Insufficient Dcal/Iweak or Dtest/Istrong units')
    # Group-disjoint calibration validation by intervention seed, not by adjacent windows.
    distinct=sorted(set(it['seed'] for it in weak))
    if len(distinct)<2: raise RuntimeError('Need >=2 weak intervention seeds for group-disjoint early stopping')
    val_seed=distinct[-1]
    train=[it for it in weak if it['seed']!=val_seed]
    val=[it for it in weak if it['seed']==val_seed]
    norm=standardize(train,val,test)
    if args.teacher_weight > 0:
        cache_calibration_teacher(train,bridge,bundle,model,device,args)
    shuffle_train=permuted_dmis(train,args.head_seed+100)
    shuffle_val=permuted_dmis(val,args.head_seed+101)
    shuffle_test=permuted_dmis(test,args.head_seed+102)
    trained={}; fit_meta={}
    for name,added in [('Matched_without_DMIS',False),('Matched+DMIS',True),('Shuffled_DMIS',True)]:
        shuffles = ({**shuffle_train,**shuffle_val} if name=='Shuffled_DMIS' else {})
        head, meta=train_head(name,train,val,bridge,bundle,model,device,norm,args,added,shuffles)
        trained[name]=(head,added)
        fit_meta[name]=meta
        torch.save({'model':head.cpu().state_dict(),'head_input_dim':int(norm[0].numel()),
                    'head_output_dim':int(np.prod(train[0]['zt'].shape[1:])),
                    'feature_mean':norm[0], 'feature_std':norm[1],
                    'run_metadata':meta},out/(name+'.pt'))
        head.to(device)
    if args.head_cal_only:
        report = dict(model=args.model,model_seed=args.model_seed,
             setting='controlled_synthetic_calibration_weak_ONLY',
             architecture=args.head_arch,teacher_weight=args.teacher_weight,
             calibration_validation_seed=val_seed,
             n_train_interventions=len(train),n_validation_interventions=len(val),
             checkpoint_sha256=fingerprint(ckpt),
             validation_ipe={k:float(v['val_ipe']) for k,v in fit_meta.items()},
             validation_extra_dmis_ipe=float(
                 fit_meta['Matched_without_DMIS']['val_ipe'] - fit_meta['Matched+DMIS']['val_ipe']),
             strong_test_outcomes_accessed=False)
        (out/'h2_head_calibration_only.json').write_text(json.dumps(report,indent=2))
        print('[H2 CAL-ONLY HEADS] '+json.dumps(report,sort_keys=True),flush=True)
        return
    rows=[]
    for item in test:
        baseline=item['before']; values={}
        for name,(head,added) in trained.items():
            values[name]=evaluate_head(head,item,bridge,bundle,model,device,norm,args,added,
                                       shuffle_test[id(item)] if name=='Shuffled_DMIS' else None)
        values['Gradient']=evaluate_gradient(item,bridge,bundle,model,device,args)
        values['Random']=evaluate_random(item,bridge,bundle,model,device,args)
        for name,after in values.items():
            rows.append(dict(model=args.model,model_seed=args.model_seed,setting='controlled_synthetic',
                             intervention_type=item['kind'],level=item['level'],intervention_seed=item['seed'],
                             condition=name,n_windows=len(item['x']),IPE_before=baseline,IPE_after=after,
                             error_reduction=baseline-after,
                             error_reduction_pct=100*(baseline-after)/max(baseline,1e-12),
                             raw_DMIS=item['response']['DMIS'],RD=item['rd'],
                             direction_norm_ratio=args.perturb_ratio,
                             source_environment=item['env'],
                             oracle_posthoc=True))
        print(f'[H2 TEST] {item["kind"]} seed{item["seed"]} before={baseline:.6f} '
              f'matched={values["Matched_without_DMIS"]:.6f} dm={values["Matched+DMIS"]:.6f}', flush=True)
    pd.DataFrame(rows).to_csv(out/'h2_paired_steering.csv',index=False)
    metadata=dict(model=args.model,model_seed=args.model_seed,setting='controlled_synthetic',
                  checkpoint=str(ckpt.resolve()),checkpoint_sha256=fingerprint(ckpt),
                  h1_experiment_root=str(args.experiment_root),support=list(READOUT)+list(TSLIB)+['DCRNN','STEVE'],
                  calibrated_on='independent_Dcal x Iweak only',tested_on='Dtest x Istrong only',
                  selected_model_lock_file=str(args.selection_file),
                  independent_calibration_rollout_seed=cal_state[-1],
                  cal_seeds=args.cal_seeds,eval_seeds=args.eval_seeds,cal_val_seed=val_seed,
                  max_cal_windows=args.max_cal_windows,max_strong_windows=args.max_strong_windows,
                  perturb_ratio=args.perturb_ratio,fit_meta=fit_meta,seconds=time.time()-start,
                  steering_config=dict(profile=args.profile,head_epochs=args.head_epochs,
                                       head_patience=args.head_patience,hidden=args.hidden,
                                       head_lr=args.head_lr,head_weight_decay=args.head_weight_decay,
                                       head_seed=args.head_seed,head_batch_size=args.head_batch_size),
                  verified_original_readout=True,readout_identity_checked=True,
                  posthoc_label_oracle=True,forecast_model_frozen=True,
                  exploratory_variant=True,head_arch=args.head_arch,
                  teacher_weight=args.teacher_weight)
    (out/'h2_manifest.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    print(f'[H2 PASS] {out / "h2_paired_steering.csv"}',flush=True)

if __name__=='__main__': main()
