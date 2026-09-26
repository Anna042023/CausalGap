#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, sys, time, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from config import *
from common import *


def physical_targets(kind, inter, sensor_ids, mapping, net):
    sid_to_edge = dict(zip(mapping.sensor_id.astype(str), mapping.sumo_edge.astype(str)))
    edge_by_id = {e.getID(): e for e in net.getEdges() if not e.isSpecial()}
    targets = []
    if kind in ('edge_removal', 'weight_modification'):
        key = 'removed_edges' if kind == 'edge_removal' else 'modified_edges'
        changed = inter.meta.get(key, [])
        for item in changed:
            i, j = int(item[0]), int(item[1])
            if i >= len(sensor_ids) or j >= len(sensor_ids): continue
            e0 = sid_to_edge.get(str(sensor_ids[i])); e1 = sid_to_edge.get(str(sensor_ids[j]))
            if not e0 or not e1 or e0 not in edge_by_id or e1 not in edge_by_id: continue
            try:
                path, cost = net.getOptimalPath(edge_by_id[e0], edge_by_id[e1], vClass='passenger')
            except Exception:
                path = None
            if path:
                # One representative physical road segment per intervened sensor-graph edge.
                cand = [e for e in path if e.allows('passenger') and not e.isSpecial()]
                if cand:
                    targets.append(cand[len(cand)//2].getID())
                else:
                    targets.append(e1)
            else:
                targets.append(e1)
    elif kind == 'node_isolation':
        nodes = [int(x) for x in inter.meta.get('isolated_nodes', [])]
        for i in nodes:
            if i >= len(sensor_ids): continue
            eid = sid_to_edge.get(str(sensor_ids[i]))
            if not eid or eid not in edge_by_id: continue
            e = edge_by_id[eid]
            targets.append(eid)
            for n in [e.getFromNode(), e.getToNode()]:
                for ee in list(n.getIncoming()) + list(n.getOutgoing()):
                    if not ee.isSpecial() and ee.allows('passenger'):
                        targets.append(ee.getID())
    # stable dedupe
    return list(dict.fromkeys(targets))


def parse_trip_pairs(route_file: Path):
    # duarouter output has vehicles with route edges; record first/last.
    pairs = []
    root = ET.parse(route_file).getroot()
    routes_by_id = {r.attrib.get('id'): r.attrib.get('edges','').split() for r in root.findall('route')}
    for v in root.findall('vehicle'):
        edges = []
        r = v.find('route')
        if r is not None: edges = r.attrib.get('edges','').split()
        elif 'route' in v.attrib: edges = routes_by_id.get(v.attrib['route'], [])
        if len(edges) >= 2: pairs.append((edges[0], edges[-1]))
    return list(dict.fromkeys(pairs))


def run_one(net_file, route_file, mapping, sim_seed, intervention=None, gamma=None, physical_edges=None):
    import traci
    import traci.constants as tc
    cmd = [str(sumo_home()/'bin'/'sumo'), '-n', str(net_file), '-r', str(route_file), '--step-length', str(STEP_SECONDS),
           '--seed', str(int(sim_seed)), '--no-step-log', 'true', '--time-to-teleport', '300',
           '--device.rerouting.probability', '1.0', '--device.rerouting.period', '60',
           '--begin', '0', '--end', str(END_SECONDS)]
    traci.start(cmd, label=f'h1_{os.getpid()}_{sim_seed}_{time.time_ns()}')
    c = traci.getConnection(f'h1_{os.getpid()}_{sim_seed}_{time.time_ns()}') if False else traci
    # traci.start without switch returns the default connection in this worker process.
    sensor_edges = mapping.sumo_edge.astype(str).tolist()
    edge_speed_limit = dict(zip(mapping.sumo_edge.astype(str), mapping.speed_limit_mps.astype(float)))
    for e in set(sensor_edges):
        try: traci.edge.subscribe(e, (tc.VAR_LAST_STEP_MEAN_SPEED,))
        except Exception: pass
    bins = []
    acc = {e: [] for e in sensor_edges}
    last = {e: max(edge_speed_limit.get(e, 13.9), 0.1) for e in sensor_edges}
    teleport_count = 0
    loaded = 0; departed = 0
    applied = False
    disconnected_ratio = 0.0
    pairs = parse_trip_pairs(route_file)
    current_bin_end = BIN_SECONDS
    try:
        while traci.simulation.getTime() < END_SECONDS:
            traci.simulationStep()
            t = traci.simulation.getTime()
            loaded += traci.simulation.getLoadedNumber()
            departed += traci.simulation.getDepartedNumber()
            teleport_count += traci.simulation.getStartingTeleportNumber()
            if intervention is not None and (not applied) and t >= T0_SECONDS:
                physical_edges = physical_edges or []
                if intervention in ('edge_removal', 'node_isolation'):
                    for e in physical_edges:
                        try: traci.edge.setDisallowed(e, ['passenger'])
                        except Exception: pass
                elif intervention == 'weight_modification':
                    for e in physical_edges:
                        try:
                            base = traci.edge.getMaxSpeed(e)
                            traci.edge.setMaxSpeed(e, max(0.1, float(base) * float(gamma)))
                        except Exception: pass
                for vid in traci.vehicle.getIDList():
                    try: traci.vehicle.rerouteTraveltime(vid, currentTravelTimes=True)
                    except Exception: pass
                # Route feasibility diagnostic after dynamic change.
                bad = 0
                for fr, to in pairs[:1000]:
                    try:
                        rr = traci.simulation.findRoute(fr, to, vType='passenger', routingMode=0)
                        if len(rr.edges) == 0: bad += 1
                    except Exception:
                        bad += 1
                disconnected_ratio = bad / max(min(len(pairs), 1000), 1)
                applied = True
            subs = traci.edge.getAllSubscriptionResults()
            for e in sensor_edges:
                v = subs.get(e, {}).get(tc.VAR_LAST_STEP_MEAN_SPEED, -1.0)
                if v is not None and float(v) >= 0:
                    last[e] = float(v)
                    acc[e].append(float(v))
            if t + 1e-9 >= current_bin_end:
                vals = []
                for e in sensor_edges:
                    vals.append(float(np.mean(acc[e])) if acc[e] else float(last[e]))
                    acc[e].clear()
                bins.append(np.array(vals, dtype=np.float32) * MPS_TO_MPH)
                current_bin_end += BIN_SECONDS
    finally:
        try: traci.close(False)
        except Exception: pass
    arr = np.stack(bins, axis=0)
    expected = int(END_SECONDS // BIN_SECONDS)
    if len(arr) < expected:
        raise RuntimeError(f'Expected {expected} 5-minute bins, got {len(arr)}')
    start = int((T0_SECONDS - HISTORY_STEPS*BIN_SECONDS)//BIN_SECONDS)
    t0bin = int(T0_SECONDS//BIN_SECONDS)
    x = arr[start:t0bin]
    y = arr[t0bin:t0bin+FUTURE_STEPS]
    diag = dict(teleport_count=int(teleport_count), loaded=int(loaded), departed=int(departed),
                failed_insertion_rate=float(max(loaded-departed,0)/max(loaded,1)), disconnected_od_ratio=float(disconnected_ratio))
    return x[:, :, None], y[:, :, None], diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-seeds', nargs='+', type=int, default=SIMULATION_SEEDS_FINAL)
    ap.add_argument('--intervention-seeds', nargs='+', type=int, default=EVAL_INTERVENTION_SEEDS)
    ap.add_argument('--only', choices=['baseline','edge_removal','weight_modification','node_isolation','all'], default='all')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    ensure_dirs()
    if not CALIBRATED_OD_JSON.exists():
        raise FileNotFoundError(f'{CALIBRATED_OD_JSON}; complete paper-mode SPSA calibration first')
    calib = json.loads(CALIBRATED_OD_JSON.read_text())
    od = np.asarray(calib['od_vph'], dtype=float)
    mapping = load_sensor_mapping()
    mapping = mapping[mapping.matched == True].copy()  # noqa: E712
    # Preserve all 207 output nodes. Missing SUMO mappings would violate the source contract.
    if len(mapping) != 207:
        raise RuntimeError(f'Need 207 sensor mappings for H1 source arrays; got {len(mapping)}. Fix mapping before generation.')
    net_file = locate_net_file()
    import sumolib
    net = sumolib.net.readNet(str(net_file), withInternal=False)
    a0, sensor_ids = load_model_adj_and_ids()
    # Reorder mapping to model sensor order.
    m = mapping.set_index(mapping.sensor_id.astype(str)).loc[list(map(str,sensor_ids))].reset_index(drop=True)

    # One baseline route file per simulator seed; paired intervention uses the same file and seed.
    baseline_x=[]; baseline_y=[]; baseline_diag=[]
    for s in args.sim_seeds:
        route = ROUTE_ROOT / f'h1_seed{s}.rou.xml'
        if args.overwrite or not route.exists():
            write_routes_from_od(net_file, m, od, END_SECONDS, s, route)
        x,y,d = run_one(net_file, route, m, s)
        baseline_x.append(x); baseline_y.append(y); baseline_diag.append(d)
    bx=np.stack(baseline_x,axis=0).astype(np.float32)
    by=np.stack(baseline_y,axis=0).astype(np.float32)
    # run_one returns [12,207,1], stack -> [B,12,207,1]
    SOURCE_ROOT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(SOURCE_ROOT/'baseline.npz', x=bx, y=by, adj=a0.astype(np.float32))
    json_dump(SOURCE_ROOT/'baseline.json', {'simulator_seeds':args.sim_seeds,'diagnostics':baseline_diag,'units':'mph',
        'history_steps':12,'future_steps':12,'t0_seconds':T0_SECONDS,'sumo_version':'1.18.0'})
    print('[BASELINE]', bx.shape, by.shape)
    if args.only=='baseline': return

    kinds = list(H1_LEVELS) if args.only=='all' else [args.only]
    for kind in kinds:
        for spec in H1_LEVELS[kind]:
            for iseed in args.intervention_seeds:
                outdir=SOURCE_ROOT/'interventions'/kind/safe_name(spec['level'])
                outdir.mkdir(parents=True,exist_ok=True)
                out=outdir/f'seed_{iseed:04d}.npz'
                if out.exists() and not args.overwrite:
                    print('[REUSE]',out); continue
                inter, a0_check, sensor_ids2 = make_graph_intervention(kind,spec,iseed)
                phys=physical_targets(kind,inter,sensor_ids,m,net)
                ys=[]; diags=[]
                for bi,s in enumerate(args.sim_seeds):
                    route=ROUTE_ROOT/f'h1_seed{s}.rou.xml'
                    xk,yk,d=run_one(net_file,route,m,s,intervention=kind,gamma=spec.get('gamma'),physical_edges=phys)
                    # Deterministic common-random-number check: pre-t0 X must equal baseline X.
                    maxdx=float(np.max(np.abs(xk-bx[bi])))
                    if maxdx>1e-4:
                        raise RuntimeError(f'Pre-intervention history mismatch for {kind}/{spec["level"]}/seed{iseed}/sim{s}: maxdx={maxdx}')
                    ys.append(yk); d['max_pre_t0_x_diff']=maxdx; diags.append(d)
                yk=np.stack(ys,axis=0).astype(np.float32)
                np.savez_compressed(out,x=bx,y=yk,adj=np.asarray(inter.adj,dtype=np.float32))
                meta=dict(inter.meta)
                meta.update({'generator':'SUMO_1.18_METRLA_PAIRED','causal_valid':True,'leakage_detected':False,
                             'paper_intensity_level':spec['level'],'h1_phase':spec['phase'],'intervention_sampling_seed':iseed,
                             'simulator_seeds':args.sim_seeds,'physical_sumo_edges':phys,'physical_mapping_rule':'representative midpoint road segment on shortest path between mapped sensor edges; node isolation closes incident junction roads',
                             'common_random_numbers':True,'routes_recomputed_after_t0':True,'diagnostics':diags})
                json_dump(out.with_suffix('.json'),meta)
                print('[DONE]',out,'physical_edges=',len(phys),'y=',yk.shape)

if __name__=='__main__':
    main()
