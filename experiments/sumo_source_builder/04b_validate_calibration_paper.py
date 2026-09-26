#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from config import *
from common import *


def paper_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    m = mapping.copy()
    if 'matched' not in m.columns:
        m['matched'] = True
    keep = m['matched'].astype(bool)
    if 'distance_m' in m.columns:
        keep &= pd.to_numeric(m['distance_m'], errors='coerce').fillna(np.inf) <= float(MAX_SENSOR_EDGE_DISTANCE_M)
    m = m.loc[keep].copy()
    m['sensor_id'] = m['sensor_id'].astype(str)
    m['sumo_edge'] = m['sumo_edge'].astype(str)
    return m


def expand_edge_data_to_sensors(ed: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    mm = mapping[['sumo_edge', 'sensor_id']].drop_duplicates().rename(columns={'sumo_edge': 'edge'})
    mm['edge'] = mm['edge'].astype(str)
    mm['sensor_id'] = mm['sensor_id'].astype(str)
    out = ed.copy()
    out['edge'] = out['edge'].astype(str)
    return out.merge(mm, on='edge', how='inner')


def js_divergence(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float('nan')
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    if not np.isfinite(lo + hi) or hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    px, _ = np.histogram(x, bins=edges)
    py, _ = np.histogram(y, bins=edges)
    px = px.astype(float) + 1e-12
    py = py.astype(float) + 1e-12
    px /= px.sum()
    py /= py.sum()
    m = 0.5 * (px + py)
    kl_x = np.sum(px * np.log(px / m))
    kl_y = np.sum(py * np.log(py / m))
    return float(0.5 * (kl_x + kl_y))


def bh_reject(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    reject = np.zeros(len(p), dtype=bool)
    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0:
        return reject
    pv = p[valid_idx]
    order = np.argsort(pv)
    ranked = pv[order]
    thresholds = q * np.arange(1, len(ranked) + 1) / len(ranked)
    ok = np.where(ranked <= thresholds)[0]
    if len(ok):
        k = ok.max()
        reject_valid = np.zeros(len(pv), dtype=bool)
        reject_valid[order[:k + 1]] = True
        reject[valid_idx] = reject_valid
    return reject


def load_dval_observed(hours: list[int], mapping: pd.DataFrame) -> pd.DataFrame:
    """Build the disjoint Dval observations from the manuscript's 7:1:1:1 chronology."""
    sp = load_metr_speed()
    n = len(sp)
    i_train = int(n * 0.7)
    i_val = int(n * 0.8)
    val = sp.iloc[i_train:i_val].copy()

    flow = read_pems_flow(Pems_FLOW_CSV)
    flow = flow[(flow.timestamp >= val.index.min()) & (flow.timestamp <= val.index.max())]
    flow = flow[flow.timestamp.dt.hour.isin(hours)].copy()

    aligned = sorted(set(mapping.sensor_id.astype(str)) & set(val.columns.astype(str)) & set(flow.sensor_id.astype(str)))
    if len(aligned) < 50:
        raise RuntimeError(f'Only {len(aligned)} aligned validation sensors.')

    speed = val.loc[val.index.hour.isin(hours), aligned].copy()
    speed.index.name = 'timestamp'
    speed_long = speed.reset_index().melt(id_vars='timestamp', var_name='sensor_id', value_name='speed_mph_obs')
    speed_long['sensor_id'] = speed_long.sensor_id.astype(str)
    speed_long['hour'] = speed_long.timestamp.dt.hour.astype(int)

    flow2 = flow[flow.sensor_id.isin(aligned)].copy()
    # PeMS Station Hour can contain duplicate records in raw exports; aggregate safely.
    flow2 = flow2.groupby(['timestamp', 'sensor_id'], as_index=False).flow.sum()
    flow2['hour'] = flow2.timestamp.dt.hour.astype(int)
    flow2 = flow2.rename(columns={'flow': 'flow_vph_obs'})

    # Align at exact hour timestamp when possible. For JSD/KS we only need sensor/hour samples;
    # an inner merge prevents inventing observations when either source is missing.
    obs = speed_long.merge(flow2[['timestamp', 'sensor_id', 'hour', 'flow_vph_obs']],
                           on=['timestamp', 'sensor_id', 'hour'], how='inner')
    obs = obs.dropna(subset=['speed_mph_obs', 'flow_vph_obs'])
    return obs


def simulate_validation_once(od: np.ndarray, mapping: pd.DataFrame, net_file: Path,
                             hours: list[int], seed: int, step_length: float) -> pd.DataFrame:
    work = CALIB_ROOT / 'paper_validation_runs' / f'seed_{seed}'
    work.mkdir(parents=True, exist_ok=True)
    duration_per_hour = 3600
    duration = len(hours) * duration_per_hour
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
        '--seed', str(int(seed)), '--no-step-log', 'true', '--duration-log.disable', 'true',
        '--time-to-teleport', '300',
    ]
    run(cmd)
    ed = parse_edge_data(edgedata)
    if ed.empty:
        raise RuntimeError(f'empty edgeData for validation seed {seed}')
    ed = expand_edge_data_to_sensors(ed, mapping)
    ed['hour_index'] = (ed.begin // duration_per_hour).astype(int)
    ed['hour'] = ed.hour_index.map({i: h for i, h in enumerate(hours)})
    ed['speed_mph_sim'] = ed.speed_mps.clip(lower=0) * MPS_TO_MPH
    sim = ed.groupby(['hour', 'sensor_id'], as_index=False).agg(
        speed_mph_sim=('speed_mph_sim', 'mean'),
        flow_vph_sim=('entered', 'sum'),
    )
    sim['sim_seed'] = int(seed)
    return sim


def metric_by_sensor(obs: pd.DataFrame, sim: pd.DataFrame, obs_col: str, sim_col: str) -> pd.DataFrame:
    rows = []
    for sid in sorted(set(obs.sensor_id) & set(sim.sensor_id)):
        xo = obs.loc[obs.sensor_id == sid, obs_col].to_numpy(float)
        xs = sim.loc[sim.sensor_id == sid, sim_col].to_numpy(float)
        xo = xo[np.isfinite(xo)]
        xs = xs[np.isfinite(xs)]
        if len(xo) < 2 or len(xs) < 2:
            continue
        ks = ks_2samp(xo, xs, alternative='two-sided', mode='auto')
        rows.append({
            'sensor_id': sid,
            'jsd': js_divergence(xo, xs),
            'wasserstein': float(wasserstein_distance(xo, xs)),
            'ks_stat': float(ks.statistic),
            'ks_p': float(ks.pvalue),
            'n_obs': int(len(xo)),
            'n_sim': int(len(xs)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Disjoint Dval calibration diagnostics for SUMO–METR-LA.')
    ap.add_argument('--hours', nargs='+', type=int, default=[7, 12, 17])
    ap.add_argument('--seeds', nargs='+', type=int, default=[3101, 3102, 3103, 3104, 3105])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--step-length', type=float, default=1.0)
    ap.add_argument('--fdr-q', type=float, default=0.05)
    args = ap.parse_args()

    ensure_dirs()
    if not CALIBRATED_OD_JSON.exists():
        raise FileNotFoundError(f'{CALIBRATED_OD_JSON}; run 04_calibrate_demand_spsa_paper.py first')

    od_obj = json.loads(CALIBRATED_OD_JSON.read_text())
    od = np.asarray(od_obj['od_vph'], dtype=float)
    mapping = paper_mapping(load_sensor_mapping())
    obs = load_dval_observed(sorted(args.hours), mapping)
    val_sids = set(obs.sensor_id.astype(str))
    mapping = mapping[mapping.sensor_id.isin(val_sids)].copy()

    print('===== DVAL OBSERVED =====', flush=True)
    print('rows=', len(obs), 'sensors=', obs.sensor_id.nunique(),
          'time=', obs.timestamp.min(), '->', obs.timestamp.max(), flush=True)

    net = locate_net_file()
    sims = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {
            ex.submit(simulate_validation_once, od, mapping, net, sorted(args.hours), int(seed), float(args.step_length)): seed
            for seed in args.seeds
        }
        for fut in as_completed(futs):
            seed = futs[fut]
            sim = fut.result()
            sims.append(sim)
            print(f'[VALIDATION] seed {seed} done rows={len(sim)}', flush=True)
    sim = pd.concat(sims, ignore_index=True)

    flow_m = metric_by_sensor(obs, sim, 'flow_vph_obs', 'flow_vph_sim')
    speed_m = metric_by_sensor(obs, sim, 'speed_mph_obs', 'speed_mph_sim')
    flow_m['bh_reject'] = bh_reject(flow_m.ks_p.to_numpy(float), q=float(args.fdr_q))
    speed_m['bh_reject'] = bh_reject(speed_m.ks_p.to_numpy(float), q=float(args.fdr_q))

    out_dir = CALIB_ROOT / 'paper_validation'
    out_dir.mkdir(parents=True, exist_ok=True)
    obs.to_csv(out_dir / 'dval_observed.csv', index=False)
    sim.to_csv(out_dir / 'dval_simulated.csv', index=False)
    flow_m.to_csv(out_dir / 'flow_sensor_metrics.csv', index=False)
    speed_m.to_csv(out_dir / 'speed_sensor_metrics.csv', index=False)

    report = {
        'protocol': 'paper_disjoint_Dval_7_1_1_1',
        'validation_hours': sorted(args.hours),
        'simulation_seeds': [int(x) for x in args.seeds],
        'fdr_q': float(args.fdr_q),
        'aligned_validation_sensors': int(obs.sensor_id.nunique()),
        'flow_median_jsd': float(np.nanmedian(flow_m.jsd)),
        'speed_median_jsd': float(np.nanmedian(speed_m.jsd)),
        'flow_median_wasserstein': float(np.nanmedian(flow_m.wasserstein)),
        'speed_median_wasserstein': float(np.nanmedian(speed_m.wasserstein)),
        'flow_ks_non_rejection_rate': float(1.0 - flow_m.bh_reject.mean()),
        'speed_ks_non_rejection_rate': float(1.0 - speed_m.bh_reject.mean()),
        'flow_ks_rejection_rate': float(flow_m.bh_reject.mean()),
        'speed_ks_rejection_rate': float(speed_m.bh_reject.mean()),
        'draft_reference_only': {
            'flow_median_jsd': 0.048,
            'speed_median_jsd': 0.053,
            'flow_ks_non_rejection_rate': 0.92,
        },
        'important_note': (
            'Draft values are references from the manuscript, not acceptance values injected by code. '
            'Report the actually measured values. A mismatch must be resolved experimentally or in the text; '
            'do not overwrite measured diagnostics.'
        ),
    }
    json_dump(out_dir / 'calibration_validation_report.json', report)

    print('\n===== PAPER CALIBRATION VALIDATION =====', flush=True)
    for k, v in report.items():
        if k not in {'draft_reference_only', 'important_note'}:
            print(f'{k}: {v}', flush=True)
    print('draft reference:', report['draft_reference_only'], flush=True)
    print('[DONE]', out_dir / 'calibration_validation_report.json', flush=True)
    print('NEXT: if diagnostics are acceptable, run run_05_generate_final.sh then run_06_validate.sh.', flush=True)


if __name__ == '__main__':
    main()
