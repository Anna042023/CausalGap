#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from config import *
from common import *


def paper_mapping(mapping: pd.DataFrame, target: pd.DataFrame | None = None) -> pd.DataFrame:
    """Use only physically matched sensors within the manuscript's 500 m rule.

    IMPORTANT: this function does NOT force the draft's reported 187/207 count. It applies the
    stated protocol and reports the count obtained from the current metadata/network. If it differs
    from 187, the manuscript number must be updated rather than silently dropping sensors.
    """
    m = mapping.copy()
    if 'matched' not in m.columns:
        m['matched'] = True
    keep = m['matched'].astype(bool)
    if 'distance_m' in m.columns:
        keep &= pd.to_numeric(m['distance_m'], errors='coerce').fillna(np.inf) <= float(MAX_SENSOR_EDGE_DISTANCE_M)
    m = m.loc[keep].copy()
    m['sensor_id'] = m['sensor_id'].astype(str)
    m['sumo_edge'] = m['sumo_edge'].astype(str)
    if target is not None:
        allowed = set(target['sensor_id'].astype(str))
        m = m[m.sensor_id.isin(allowed)].copy()
    return m


def expand_edge_data_to_sensors(ed: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Expand one SUMO edge measurement to every aligned sensor on that edge.

    This fixes the previous drop_duplicates('sumo_edge') behaviour that silently removed sensors
    when several METR-LA detectors mapped to the same SUMO edge.
    """
    mm = mapping[['sumo_edge', 'sensor_id']].drop_duplicates().rename(columns={'sumo_edge': 'edge'})
    mm['edge'] = mm['edge'].astype(str)
    mm['sensor_id'] = mm['sensor_id'].astype(str)
    out = ed.copy()
    out['edge'] = out['edge'].astype(str)
    return out.merge(mm, on='edge', how='inner')


def simulate_candidate(
    od: np.ndarray,
    seed: int,
    target: pd.DataFrame,
    mapping: pd.DataFrame,
    net_file: Path,
    tag: str,
    duration_per_hour: int = 3600,
    step_length: float = 1.0,
):
    """Evaluate one OD candidate on the representative TRAIN-period calibration hours.

    Calibration uses a precomputed route file and disables dynamic rerouting. This is both faster
    and appropriate for the unperturbed baseline OD calibration. Rerouting remains enabled later in
    the actual paired structural-intervention runs, exactly where the paper requires it.
    """
    work = CALIB_ROOT / 'spsa_paper_runs' / tag
    work.mkdir(parents=True, exist_ok=True)
    hours = sorted(int(x) for x in target.hour.unique().tolist())
    duration = int(len(hours) * duration_per_hour)

    routes = work / 'routes.rou.xml'
    write_routes_from_od(net_file, mapping, od, duration, seed, routes)

    edgedata = work / 'edgedata.xml'
    add = work / 'sensors.add.xml'
    create_edge_data_additional(mapping, add, edgedata, period=duration_per_hour)

    sumo = sumo_home() / 'bin' / 'sumo'
    cmd = [
        str(sumo), '-n', str(net_file), '-r', str(routes), '-a', str(add),
        '--begin', '0', '--end', str(duration),
        '--step-length', str(float(step_length)),
        '--seed', str(int(seed)),
        '--no-step-log', 'true',
        '--duration-log.disable', 'true',
        '--time-to-teleport', '300',
    ]
    run(cmd)

    ed = parse_edge_data(edgedata)
    if ed.empty:
        raise RuntimeError('SUMO edgeData is empty')
    ed = expand_edge_data_to_sensors(ed, mapping)
    ed['hour_index'] = (ed.begin // duration_per_hour).astype(int)
    ed['hour'] = ed.hour_index.map({i: h for i, h in enumerate(hours)})
    ed['speed_mph'] = ed.speed_mps.clip(lower=0) * MPS_TO_MPH

    sim = ed.groupby(['hour', 'sensor_id'], as_index=False).agg(
        speed_mph=('speed_mph', 'mean'),
        flow_vph=('entered', 'sum'),
    )
    z = target.merge(sim, on=['hour', 'sensor_id'], suffixes=('_obs', '_sim'))
    if len(z) < 30:
        raise RuntimeError(f'Too few simulated sensor/hour matches: {len(z)}')

    sp_scale = max(float(np.nanmedian(z.speed_mph_obs)), 1.0)
    fl_scale = max(float(np.nanmedian(z.flow_vph_obs)), 1.0)
    sp_loss = float(np.nanmean(np.abs(z.speed_mph_sim - z.speed_mph_obs)) / sp_scale)
    fl_loss = float(np.nanmean(np.abs(z.flow_vph_sim - z.flow_vph_obs)) / fl_scale)
    loss = 0.5 * sp_loss + 0.5 * fl_loss

    z.to_csv(work / 'detail.csv', index=False)
    return loss, sp_loss, fl_loss, z


def main():
    ap = argparse.ArgumentParser(description='Paper-aligned fast SPSA OD calibration for SUMO–METR-LA.')
    ap.add_argument('--iterations', type=int, default=6,
                    help='Use 4-6 for the first pass; resume to 10-20 only if validation remains poor.')
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--a', type=float, default=0.18)
    ap.add_argument('--c', type=float, default=0.12)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--step-length', type=float, default=1.0)
    ap.add_argument('--parallel-pair', action='store_true', default=True)
    ap.add_argument('--no-parallel-pair', dest='parallel_pair', action='store_false')
    args = ap.parse_args()

    ensure_dirs()
    targets_path = CALIB_ROOT / 'calibration_targets.csv'
    if not targets_path.exists():
        raise FileNotFoundError(f'{targets_path}; run Step 3 first')

    target = pd.read_csv(targets_path, dtype={'sensor_id': str})
    target['sensor_id'] = target['sensor_id'].astype(str)
    raw_mapping = load_sensor_mapping()
    mapping = paper_mapping(raw_mapping, target)
    common_sensors = sorted(set(target.sensor_id) & set(mapping.sensor_id))
    target = target[target.sensor_id.isin(common_sensors)].copy()

    print('===== PAPER ALIGNMENT =====', flush=True)
    print('500m-matched mapping sensors =', mapping.sensor_id.nunique(), flush=True)
    print('train calibration sensors    =', target.sensor_id.nunique(), flush=True)
    print('shared SUMO edges            =', mapping.sumo_edge.nunique(), flush=True)
    print('NOTE: do not force the draft count 187; report the actual protocol result.', flush=True)

    if target.sensor_id.nunique() < 50:
        raise RuntimeError('Too few aligned calibration sensors.')

    net = locate_net_file()
    init = make_initial_od(target, mapping)
    mask = ~np.eye(init.shape[0], dtype=bool)
    theta0 = np.log(np.maximum(init[mask], 1.0))

    state_p = CALIB_ROOT / 'spsa_paper_state.json'
    hist_p = CALIB_ROOT / 'spsa_paper_history.csv'

    if args.resume and state_p.exists():
        st = json.loads(state_p.read_text())
        theta = np.array(st['theta'], dtype=float)
        start = int(st['iteration']) + 1
        best = (float(st['best_loss']), np.array(st['best_theta'], dtype=float))
        hist = pd.read_csv(hist_p).to_dict('records') if hist_p.exists() else []
    else:
        theta = theta0.copy()
        start = 1
        best = (float('inf'), theta.copy())
        hist = []
        for p in (state_p, hist_p):
            if p.exists():
                p.unlink()

    def unpack(th):
        od = np.zeros_like(init)
        od[mask] = np.exp(np.clip(th, np.log(1.0), np.log(12000.0)))
        return od

    for k in range(start, args.iterations + 1):
        ak = args.a / ((k + 10) ** 0.602)
        ck = args.c / (k ** 0.101)
        # deterministic per-iteration perturbation; resume produces exactly the same sequence
        rng_k = np.random.default_rng(args.seed + 10007 * k)
        delta = rng_k.choice([-1.0, 1.0], size=theta.shape)
        od_plus = unpack(theta + ck * delta)
        od_minus = unpack(theta - ck * delta)

        def plus_job():
            return simulate_candidate(od_plus, args.seed, target, mapping, net,
                                      f'iter{k:03d}_plus', step_length=args.step_length)

        def minus_job():
            return simulate_candidate(od_minus, args.seed, target, mapping, net,
                                      f'iter{k:03d}_minus', step_length=args.step_length)

        if args.parallel_pair:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fp = ex.submit(plus_job)
                fm = ex.submit(minus_job)
                lp, sp_p, fl_p, _ = fp.result()
                lm, sp_m, fl_m, _ = fm.result()
        else:
            lp, sp_p, fl_p, _ = plus_job()
            lm, sp_m, fl_m, _ = minus_job()

        g = ((lp - lm) / (2.0 * ck)) * delta
        theta = theta - ak * g
        current = 0.5 * (lp + lm)
        candidate_theta = theta.copy()
        if current < best[0]:
            best = (current, candidate_theta)

        row = dict(
            iteration=k, loss_plus=lp, loss_minus=lm, loss_mean=current,
            speed_loss_plus=sp_p, speed_loss_minus=sp_m,
            flow_loss_plus=fl_p, flow_loss_minus=fl_m,
            a_k=ak, c_k=ck,
        )
        hist.append(row)
        pd.DataFrame(hist).to_csv(hist_p, index=False)
        json_dump(state_p, {
            'iteration': k,
            'theta': theta.tolist(),
            'best_loss': best[0],
            'best_theta': best[1].tolist(),
        })
        print('[SPSA]', row, flush=True)

    st = json.loads(state_p.read_text())
    theta_best = np.array(st['best_theta'], dtype=float)
    od_best = unpack(theta_best)
    loss, sp_loss, fl_loss, detail = simulate_candidate(
        od_best, args.seed, target, mapping, net, 'final_best', step_length=args.step_length
    )
    detail.to_csv(CALIB_ROOT / 'calibration_fit_detail_paper.csv', index=False)

    out_json = CALIB_ROOT / 'calibrated_od_paper.json'
    payload = {
        'protocol': 'paper_train_only_SPSA_compact_zone_OD',
        'implementation_note': (
            'The manuscript specifies SPSA OD calibration but not the OD-zone parameterization. '
            'This implementation retains the deterministic 2x3 geographic zone grid. '
            'Calibration uses TRAIN-period targets only; final validity is assessed separately on Dval.'
        ),
        'zone_grid': [2, 3],
        'od_vph': od_best.tolist(),
        'objective_train': float(loss),
        'speed_normalized_mae_train': float(sp_loss),
        'flow_normalized_mae_train': float(fl_loss),
        'representative_hours': sorted(target.hour.unique().astype(int).tolist()),
        'aligned_sensor_count': int(target.sensor_id.nunique()),
        'unique_sumo_edges': int(mapping.sumo_edge.nunique()),
        'max_alignment_distance_m': float(MAX_SENSOR_EDGE_DISTANCE_M),
        'spsa_seed': int(args.seed),
        'iterations': int(args.iterations),
        'step_length_s': float(args.step_length),
        'dynamic_rerouting_during_calibration': False,
    }
    json_dump(out_json, payload)

    # Existing Step 5 reads CALIBRATED_OD_JSON. Back it up and install the paper-calibrated OD.
    if CALIBRATED_OD_JSON.exists():
        backup = CALIB_ROOT / 'calibrated_od_before_paper_fix.json'
        if not backup.exists():
            shutil.copy2(CALIBRATED_OD_JSON, backup)
    shutil.copy2(out_json, CALIBRATED_OD_JSON)

    print('[DONE] paper-calibrated OD ->', out_json, flush=True)
    print('[INSTALLED FOR STEP 5] ->', CALIBRATED_OD_JSON, flush=True)
    print('train objective=', loss, 'speed=', sp_loss, 'flow=', fl_loss, flush=True)
    print('NEXT: run 04b_validate_calibration_paper.py on disjoint Dval.', flush=True)


if __name__ == '__main__':
    main()
