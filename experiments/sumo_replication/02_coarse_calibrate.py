#!/usr/bin/env python3
"""Coarse STATISTICS only: no PeMS, station/trajectory matching, SPSA, or real OD."""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from config import ROOT, CODE, BIN, END, MPH
sys.path.insert(0,str(CODE))


def stats(speed_mph, freeflow):
    """Mean/std, congestion fraction and temporal peak/valley SPEED ratio."""
    a=np.asarray(speed_mph,float)
    if a.ndim==4: a=a[:,0,:,0]  # first step of each overlapping training window
    elif a.ndim==3:
        a=a[...,0] if a.shape[-1]==1 else a[:,0,:]
    if a.ndim!=2:raise ValueError(f'Expected a time-by-node array, got {a.shape}')
    v=np.where(np.isfinite(a)&(a>0),a,np.nan)
    allv=v[np.isfinite(v)]
    if len(allv)<20:raise ValueError('Too few valid speed measurements')
    # Avoid np.nanmean(empty row) warnings for bins with no valid sensor speeds.
    counts=np.isfinite(v).sum(axis=1)
    keep=counts>0
    network_speed=np.nansum(v[keep],axis=1)/counts[keep]
    if len(network_speed)<2:
        raise ValueError('Too few nonempty time bins for peak/valley statistics')
    return {'mean_speed_mph':float(allv.mean()), 'speed_std_mph':float(allv.std()),
      'congestion_fraction':float((allv<.6*freeflow).mean()),
      'peak_valley_speed_ratio':float(np.quantile(network_speed,.9)/
                                    max(np.quantile(network_speed,.1),1e-4))}


def target_stats(root=ROOT):
    from causalgap.data import load_dataset
    g=np.load(root/'subgraph.npz',allow_pickle=False)
    idx=g['indices'].astype(int);free=float(g['freeflow_mph'])
    bundle=load_dataset('METR-LA',history_len=12,pred_len=12)
    # No validation or test series used for fitting coarse statistics.
    arr=bundle.scaler.inverse(bundle.train_x[...,0])[:,:,idx]
    return stats(arr,free)


def sumo_env():
    import os, shutil
    home=os.getenv('SUMO_HOME')
    if home: sys.path.insert(0,str(Path(home)/'tools'))
    if shutil.which('sumo') is None or shutil.which('duarouter') is None:
        raise RuntimeError('Missing SUMO/duarouter binaries in PATH')
    import sumolib
    return sumolib


def make_routes(root, seed, factor, peak_amp):
    sumo_env();import subprocess,shutil,xml.etree.ElementTree as ET
    a=np.load(root/'subgraph.npz',allow_pickle=False)['adj']
    n=len(a); rng=np.random.default_rng(seed)
    # Synthetic homogeneous demands (veh/sensor/hour); departures ramp near midpoint.
    # Identical traffic for ALL topology interventions for a fixed sim seed.
    rate=38*float(factor)*n/3600
    num=int(round(rate*END))
    if num<50:raise RuntimeError('Synthetic demand too low')
    # Inverse-transform the peak profile to preserve nonnegative deterministic departures.
    cand=rng.uniform(0,END-1,num*3)
    accept=rng.uniform(0,1,len(cand))<(1+float(peak_amp)*np.sin(2*np.pi*cand/max(END,1))**2)/(1+float(peak_amp))
    depart=np.sort(rng.choice(cand[accept],size=min(num,int(accept.sum())),replace=False))
    if len(depart)<num:depart=np.sort(np.concatenate((depart,rng.uniform(0,END-1,num-len(depart)))))
    graph_tag=hashlib.sha256((root/'subgraph.npz').read_bytes()).hexdigest()[:10]
    trip=root/'routes'/f'trips_g{graph_tag}_seed{seed}_factor{factor:g}_peak{peak_amp:g}.xml'
    route=trip.with_name('routes'+trip.name[5:]);trip.parent.mkdir(parents=True,exist_ok=True)
    rootx=ET.Element('routes')
    ET.SubElement(rootx,'vType',id='passenger',vClass='passenger',accel='2.6',decel='4.5',length='5',minGap='2.5',sigma='0.5')
    for j,t in enumerate(depart):
        i,d=rng.choice(n,2,replace=False)
        ET.SubElement(rootx,'trip',id=f'veh{j}',type='passenger',depart=f'{t:.2f}',
                      **{'from':f'sensor_{i}','to':f'sensor_{d}'})
    ET.ElementTree(rootx).write(trip,encoding='utf-8',xml_declaration=True)
    if route.exists():return route
    cmd=[shutil.which('duarouter'),'-n',str(root/'abstract.net.xml'),'-r',str(trip),'-o',str(route),
       '--ignore-errors','true','--seed',str(seed),'--routing-algorithm','dijkstra','--no-warnings','true']
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode or not route.exists():raise RuntimeError(f'duarouter failed: {p.stderr[-3000:]}')
    v=len(ET.parse(route).getroot().findall('vehicle'))
    if v < num*.90: raise RuntimeError(f'Synthetic trips routed {v}/{num}; topology unusable')
    print(f'[ROUTE] seed={seed}: {v}/{num} routed',flush=True)
    return route


def closure_safe_route(route, root):
    """Reuse identical trips; give ALL vehicles (including post-t0 arrivals) routing mode 8.

    SUMO 1.18 may not expose --device.rerouting.mode as a command-line option.
    A vType generic param is understood at vehicle loading and does not change
    any departure, destination, route, seed, or traffic demand.
    """
    import xml.etree.ElementTree as ET
    route=Path(route)
    route_bytes=route.read_bytes()
    digest=hashlib.sha256(route_bytes).hexdigest()[:16]
    safe_dir=Path(root)/'routes'/'closure_safe'
    safe_dir.mkdir(parents=True,exist_ok=True)
    safe=safe_dir/f'{route.stem}_{digest}_mode8.xml'
    if safe.exists():return safe
    tree=ET.parse(route)
    types=tree.getroot().findall('vType')
    if not types:
        raise RuntimeError(f'Closure route file has no vType: {route}')
    for typ in types:
        for prm in list(typ.findall('param')):
            if prm.attrib.get('key')=='device.rerouting.mode':typ.remove(prm)
        ET.SubElement(typ,'param',key='device.rerouting.mode',value='8')
    tmp=safe.with_suffix('.tmp')
    tree.write(tmp,encoding='utf-8',xml_declaration=True)
    tmp.replace(safe)
    return safe


def sumo_command(root, route, seed, hard_closure=False):
    """SUMO 1.18 compatible: no unsupported --device.rerouting.mode CLI flag.

    --ignore-route-errors is restricted to hard-closure counterfactuals;
    baseline retains strict route validation. Mode 8 is provided by vType
    generic parameters for trips loaded after t0, not by a CLI option.
    """
    import shutil
    executable=shutil.which('sumo') or 'sumo'
    chosen=closure_safe_route(route,root) if hard_closure else Path(route)
    cmd=[executable,'-n',str(root/'abstract.net.xml'),'-r',str(chosen),'--seed',str(seed),
         '--begin','0','--end',str(END),'--step-length','1','--no-step-log','true',
         '--time-to-teleport','180','--device.rerouting.probability','0',
         '--no-warnings','true']
    if hard_closure:
        cmd.extend(['--ignore-route-errors','true'])
    return cmd


def _reroute_vehicles(traci, vehicle_ids, blocked_edges=()):
    """Replan active/newly departed vehicles; never silently swallow failure.

    For hard closures routing mode 0 tries an actual detour (honors permissions).
    Mode 8 is a fall-back for unreachable destinations: the vehicle may wait or
    teleport rather than crashing SUMO. All such cases are recorded for audit.
    """
    blocked=set(blocked_edges)
    diag={'attempts':0,'failures':0,'unchanged_blocked_routes':0}
    for vid in vehicle_ids:
        diag['attempts']+=1
        try:
            if blocked:
                traci.vehicle.setRoutingMode(vid,0)
            traci.vehicle.rerouteTraveltime(vid,currentTravelTimes=True)
            if blocked:
                route=traci.vehicle.getRoute(vid)
                idx=traci.vehicle.getRouteIndex(vid)
                future=route[max(0,int(idx)+1):]
                if blocked.intersection(future):
                    diag['unchanged_blocked_routes']+=1
        except traci.TraCIException:
            diag['failures']+=1
        finally:
            if blocked:
                # Transient lane-permission changes can make future trips
                # unreachable even when their original route was valid.
                try:traci.vehicle.setRoutingMode(vid,8)
                except traci.TraCIException:pass
    return diag


def simulate(root, route, seed, kind=None, affected=(),gamma=None, start_t=None):
    sumo_env();import traci
    a=np.load(root/'subgraph.npz',allow_pickle=False)
    n=len(a['adj']);ff=float(a['freeflow_mph'])
    hard=kind in ('edge_removal','node_isolation')
    cmd=sumo_command(root,route,seed,hard_closure=hard)
    traci.start(cmd)
    record=[];acc=[[] for _ in range(n)]
    teleport=departures=0;applied=False
    route_audit={'attempts':0,'failures':0,'unchanged_blocked_routes':0}
    start_t=START_T if start_t is None else int(start_t)
    blocked=set(affected) if hard else set()
    touched=set()
    try:
        while traci.simulation.getTime()<END:
            t=traci.simulation.getTime()
            if kind and not applied and t>=start_t:
                # Mode 8 BEFORE closing protects pre-existing fixed routes.
                # Physical closure still occurs strictly after the context.
                if hard:
                    for vid in traci.vehicle.getIDList():
                        traci.vehicle.setRoutingMode(vid,8)
                for e in affected:
                    nlanes=traci.edge.getLaneNumber(e)
                    for j in range(nlanes):
                        lane=f'{e}_{j}'
                        if hard:
                            traci.lane.setDisallowed(lane,['passenger'])
                        elif kind=='weight_modification':
                            if gamma is None or not (0<float(gamma)<1):
                                raise ValueError('weight_modification requires 0 < gamma < 1')
                            traci.lane.setMaxSpeed(lane,max(.5,traci.lane.getMaxSpeed(lane)*float(gamma)))
                        else:
                            raise ValueError(f'Unknown intervention type: {kind}')
                    touched.add(e)
                d=_reroute_vehicles(traci,list(traci.vehicle.getIDList()),blocked)
                for k,v in d.items():route_audit[k]+=v
                applied=True
            try:
                traci.simulationStep()
            except traci.FatalTraCIError as exc:
                raise RuntimeError('SUMO closed its TraCI connection during intervention; '
                   f'kind={kind}, seed={seed}, affected={len(affected)}. '
                   'Do NOT use partial output. Inspect SUMO stderr for blocked or unreachable trips.') from exc
            teleport+=traci.simulation.getStartingTeleportNumber()
            departures+=traci.simulation.getDepartedNumber()
            # Vehicles that depart after t0 were NOT present at actuation time;
            # replan them as well instead of leaving a stale fixed route.
            if applied and kind:
                departed_ids=traci.simulation.getDepartedIDList()
                if departed_ids:
                    d=_reroute_vehicles(traci,departed_ids,blocked)
                    for k,v in d.items():route_audit[k]+=v
            for i in range(n):
                v=traci.edge.getLastStepMeanSpeed(f'sensor_{i}')
                if v>=0:acc[i].append(float(v)*MPH)
            tt=traci.simulation.getTime()
            if abs(tt/BIN-round(tt/BIN))<1e-7:
                record.append([float(np.mean(acc[i])) if acc[i] else ff for i in range(n)])
                acc=[[] for _ in range(n)]
    finally:
        try:traci.close(False)
        except Exception:pass
    arr=np.asarray(record,np.float32)
    expected=END//BIN
    if len(arr)!=expected:raise RuntimeError(f'Expected {expected} bins, got {len(arr)}')
    b=int(start_t//BIN);x=arr[b-12:b];y=arr[b:b+12]
    if x.shape!=(12,n) or y.shape!=(12,n):raise RuntimeError('Invalid paired time windows')
    diag={'teleports':int(teleport),'departures':int(departures),
          'applied':bool(applied) if kind else None,'affected_edges':len(affected),
          'touched_edges':len(touched),'hard_closure':bool(hard),**route_audit}
    if kind:
        print('[ACTUATION AUDIT]',kind,'seed=',seed,diag,flush=True)
    return x[...,None],y[...,None],diag

from config import T0 as START_T


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--factors',type=float,nargs='+',default=[1.,2.,4.])
    ap.add_argument('--peak-amps',type=float,nargs='+',default=[.2,.8])
    ap.add_argument('--seed',type=int,default=1901);args=ap.parse_args()
    target=target_stats(args.root);a=np.load(args.root/'subgraph.npz',allow_pickle=False)
    free=float(a['freeflow_mph']);records=[];best=None
    for factor in args.factors:
        for peak in args.peak_amps:
            route=make_routes(args.root,args.seed,factor,peak)
            x,y,diag=simulate(args.root,route,args.seed)
            observed=stats(np.concatenate([x,y],axis=0),free)
            errs=[abs(observed[k]-target[k])/max(abs(target[k]),1e-3) for k in target]
            loss=float(np.mean(errs))
            row=dict(factor=factor,peak_amp=peak,loss=loss,**{'sim_'+k:v for k,v in observed.items()},
                     **{'real_'+k:v for k,v in target.items()})
            records.append(row);print('[COARSE]',row,flush=True)
            if best is None or loss<best['loss']:best=dict(row,route=str(route),seed=args.seed)
    pd.DataFrame(records).to_csv(args.root/'coarse_grid.csv',index=False)
    (args.root/'coarse_choice.json').write_text(json.dumps(best,indent=2))
    print('[DONE] coarse candidate chosen:',best,'\nNOTE: report actual mismatch; this is not real-OD or trajectory calibration.')
if __name__=='__main__':main()
