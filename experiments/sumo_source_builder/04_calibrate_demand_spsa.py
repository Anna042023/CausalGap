#!/usr/bin/env python3
"""METR-LA SUMO demand calibration, v2.

Changes from the uploaded v1:
  * never drop duplicate sensor->edge mappings;
  * preserve all target rows (no silent inner-join sample selection);
  * record genuine missing speeds as NaN, not 0 mph;
  * pilot OD multipliers rather than hard-code an unsupported rescale;
  * keep pilot/runs isolated from paper-mode canonical artifacts;
  * select the best ACTUALLY EVALUATED theta, not post-update theta;
  * resume with saved RNG state and consistent protocol/configuration.

The 6x6 matrix is a single constant demand for all 3 consecutive simulated
hours, labelled 07/12/17 solely to compare with representative observations.
It does NOT reproduce an actual non-contiguous 24-hour traffic cycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import *
from common import (
    ensure_dirs, load_sensor_mapping, locate_net_file, make_initial_od,
    write_routes_from_od, create_edge_data_additional, parse_edge_data,
    sumo_home, json_dump,
)

PROTOCOL = 'metrla-sumo-spsa-v2-all-sensors-fixed-rows-best-candidate'
RUN_ROOT = CALIB_ROOT / 'spsa_calibration'


def prepare_tables(target: pd.DataFrame, mapping: pd.DataFrame):
    """Validate alignment and attach SUMO edge to every target sensor/hour."""
    needed = {'hour', 'sensor_id', 'speed_mph', 'flow_vph'}
    if not needed.issubset(target):
        raise ValueError(f'Target missing columns {needed - set(target.columns)}')
    required_mapping = {'sensor_id', 'sumo_edge', 'matched'}
    if not required_mapping.issubset(mapping):
        raise ValueError(f'Mapping missing columns {required_mapping - set(mapping.columns)}')
    target = target.copy()
    target['sensor_id'] = target.sensor_id.astype(str)
    target['hour'] = target.hour.astype(int)
    target['speed_mph'] = pd.to_numeric(target.speed_mph, errors='coerce')
    target['flow_vph'] = pd.to_numeric(target.flow_vph, errors='coerce')
    if target[['speed_mph', 'flow_vph']].isna().any().any():
        raise ValueError('Target has non-finite speed/flow observations')
    if target.duplicated(['hour', 'sensor_id']).any():
        raise ValueError('Duplicate (hour, sensor_id) in targets')
    if len(target) == 0:
        raise ValueError('Empty target file')
    matched = mapping.loc[mapping.matched == True, ['sensor_id', 'sumo_edge']].copy()  # noqa: E712
    matched['sensor_id'] = matched.sensor_id.astype(str)
    matched['sumo_edge'] = matched.sumo_edge.astype(str)
    if matched.duplicated('sensor_id').any():
        raise ValueError('One sensor has several matched edges: resolve mapping explicitly')
    out = target.merge(matched, on='sensor_id', how='left', validate='many_to_one')
    missing = out.loc[out.sumo_edge.isna(), 'sensor_id'].unique().tolist()
    if missing:
        raise ValueError(f'Target sensors have no matched edge: {missing}')
    n_sensors = out.sensor_id.nunique()
    n_edges = out.sumo_edge.nunique()
    print(f'[MAPPING] target_rows={len(out)} target_sensors={n_sensors} '
          f'unique_edges={n_edges} shared-edge extra sensors={n_sensors - n_edges}', flush=True)
    if n_sensors > n_edges:
        print('[NOTE] Multiple real sensors share the SAME simulated edge measurement. '
              'Do not claim independent sensor-level SUMO validation for these sensors.', flush=True)
    return out


def edge_data_to_detail(target_with_edge: pd.DataFrame, ed: pd.DataFrame,
                        duration_per_hour: int = 3600):
    """Pure conversion, testable without SUMO: one row per ORIGINAL target row."""
    hours = sorted(target_with_edge.hour.unique().tolist())
    if len(ed):
        ed = ed.copy()
        ed['edge'] = ed.edge.astype(str)
        ed['hour_index'] = (ed.begin // duration_per_hour).astype(int)
        ed['hour'] = ed.hour_index.map(dict(enumerate(hours)))
        ed = ed.loc[ed.hour.notna()].copy()
        ed['hour'] = ed.hour.astype(int)
        # SUMO uses speed=-1 on edge/time bins without valid speed samples.
        ed['speed_mph_sim'] = ed.speed_mps.where(ed.speed_mps >= 0, np.nan) * MPS_TO_MPH
        ed['flow_vph_sim'] = ed.entered.clip(lower=0) * (3600.0 / duration_per_hour)
        sim = ed.groupby(['hour', 'edge'], as_index=False).agg(
            speed_mph_sim=('speed_mph_sim', 'mean'),
            flow_vph_sim=('flow_vph_sim', 'sum'),
        )
    else:
        sim = pd.DataFrame(columns=['hour', 'edge', 'speed_mph_sim', 'flow_vph_sim'])
    detail = target_with_edge.merge(
        sim, left_on=['hour', 'sumo_edge'], right_on=['hour', 'edge'],
        how='left', validate='many_to_one', sort=False,
    )
    # No edgeData row means zero entering vehicles, NOT absence of a target.
    detail['flow_vph_sim'] = detail.flow_vph_sim.fillna(0.0)
    detail['speed_mph_obs'] = detail.pop('speed_mph')
    detail['flow_vph_obs'] = detail.pop('flow_vph')
    detail = detail.drop(columns='edge')
    if len(detail) != len(target_with_edge):
        raise RuntimeError('Unexpected loss/duplication of target rows')
    if not np.isfinite(detail.flow_vph_sim).all():
        raise RuntimeError('Non-finite simulated flow')
    return detail


def evaluate_detail(detail: pd.DataFrame):
    """Same scale-normalized MAE family as v1; explicitly report coverage.

    The sensor-level objective retains all shared-edge observations; it is
    *not* 197 independent edge measurements. Missing speeds are omitted from
    speed MAE and counted separately, never converted to invented 0 mph.
    """
    valid = np.isfinite(detail.speed_mph_sim.to_numpy(dtype=float))
    speed_coverage = float(valid.mean())
    if speed_coverage < 0.90:
        raise RuntimeError(f'Only {speed_coverage:.1%} target rows have valid simulated '
                           'speed. Fix traffic coverage before comparing SPSA losses.')
    sp_scale = max(float(np.nanmedian(detail.speed_mph_obs)), 1.0)
    fl_scale = max(float(np.nanmedian(detail.flow_vph_obs)), 1.0)
    speed_loss = float(np.mean(np.abs(
        detail.loc[valid, 'speed_mph_sim'] - detail.loc[valid, 'speed_mph_obs']
    )) / sp_scale)
    flow_loss = float(np.mean(np.abs(detail.flow_vph_sim - detail.flow_vph_obs)) / fl_scale)
    ratio = float(detail.flow_vph_sim.sum() / max(detail.flow_vph_obs.sum(), 1e-8))
    loss = 0.5 * speed_loss + 0.5 * flow_loss
    return dict(loss=loss, speed_loss=speed_loss, flow_loss=flow_loss,
                flow_ratio=ratio, speed_coverage=speed_coverage,
                rows=len(detail), sensors=int(detail.sensor_id.nunique()),
                unique_edges=int(detail.sumo_edge.nunique()))


def simulate_candidate(od, seed, target_with_edge, mapping, net_file, tag,
                       work_root, duration_per_hour=3600):
    """Run a candidate without touching v1 results or the canonical OD."""
    work = Path(work_root) / tag
    work.mkdir(parents=True, exist_ok=True)
    hours = sorted(target_with_edge.hour.unique().tolist())
    duration = int(len(hours) * duration_per_hour)
    routes = work / 'routes.rou.xml'
    write_routes_from_od(net_file, mapping, od, duration, seed, routes)
    edgedata = work / 'edgedata.xml'
    add = work / 'sensors.add.xml'
    create_edge_data_additional(mapping, add, edgedata, period=duration_per_hour)
    cmd = [str(sumo_home() / 'bin' / 'sumo'), '-n', str(net_file),
           '-r', str(routes), '-a', str(add), '--begin', '0', '--end', str(duration),
           '--step-length', '1', '--seed', str(int(seed)), '--no-step-log', 'true',
           '--time-to-teleport', '300', '--device.rerouting.probability', '1.0',
           '--device.rerouting.period', '120']
    log = work / 'sumo.log'
    print(f'[SUMO] {tag}: log={log}', flush=True)
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'SUMO exit={result.returncode}; inspect {log}')
    detail = edge_data_to_detail(target_with_edge, parse_edge_data(edgedata), duration_per_hour)
    met = evaluate_detail(detail)
    print(f'[METRIC] {tag}: {met}', flush=True)
    return met, detail


def init_od(target, mapping, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('--od-scale must be positive and finite')
    # IMPORTANT: this is a controlled pilot factor, NOT an assumed correction.
    return make_initial_od(target, mapping) * float(scale)


def od_to_theta(od):
    mask = ~np.eye(len(od), dtype=bool)
    return np.log(np.maximum(od[mask], 1.0))


def theta_to_od(theta, shape, max_pair_vph):
    mask = ~np.eye(shape[0], dtype=bool)
    od = np.zeros(shape, dtype=float)
    od[mask] = np.exp(np.clip(theta, 0.0, np.log(max_pair_vph)))
    return od


def signature(target_with_edge, net, args):
    payload = dict(protocol=PROTOCOL, seed=args.seed, od_scale=args.od_scale,
                   max_pair_vph=args.max_pair_vph, a=args.a, c=args.c,
                   network=str(net.resolve()), network_size=net.stat().st_size,
                   network_mtime_ns=net.stat().st_mtime_ns,
                   observations=target_with_edge[['hour','sensor_id','sumo_edge',
                       'speed_mph','flow_vph']].sort_values(['hour','sensor_id']).to_json(orient='records'))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def show_audit(target_with_edge, initial):
    print('[AUDIT] target rows=', len(target_with_edge),
          'sensors=', target_with_edge.sensor_id.nunique(),
          'unique SUMO edges=', target_with_edge.sumo_edge.nunique(), flush=True)
    print('[AUDIT] OD veh/h=', round(float(initial.sum()), 3),
          'max OD cell=', round(float(initial.max()), 3), flush=True)
    print('[AUDIT] observed station flow sum is NOT a trip/OD total.', flush=True)
    print('[AUDIT] This 2D OD is constant over three consecutive simulated hours.', flush=True)


def publish_outputs(run_dir: Path, met: dict):
    """Explicit publication; keep a timestamped backup of existing canonical files."""
    backup = CALIB_ROOT / 'spsa_calibration' / 'canonical_backups' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    srcs = [('calibrated_od.json', CALIBRATED_OD_JSON),
            ('calibration_fit_detail.csv', CALIB_ROOT / 'calibration_fit_detail.csv')]
    for name, dst in srcs:
        if dst.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, backup / name)
    for name, dst in srcs:
        shutil.copy2(run_dir / name, dst)
    print(f'[PUBLISH] Canonical outputs updated. Original backups: {backup if backup.exists() else "none"}', flush=True)
    if not 0.5 <= met['flow_ratio'] <= 1.5:
        print('[WARNING] Flow ratio is still far from 1: publishing does NOT certify calibration.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', action='store_true', help='No SUMO; inspect alignment and OD scale')
    ap.add_argument('--pilot-scales', nargs='+', type=float,
                    help='Run measured alternatives, e.g. --pilot-scales 1 2 (no SPSA)')
    ap.add_argument('--iterations', type=int, default=20)
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--od-scale', type=float, default=1.0)
    ap.add_argument('--max-pair-vph', type=float, default=5000.0)
    ap.add_argument('--a', type=float, default=0.18)
    ap.add_argument('--c', type=float, default=0.12)
    ap.add_argument('--run-name', type=str, default='calibration_run')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--publish', action='store_true',
                    help='Explicitly overwrite canonical OD/detail with timestamped backups')
    args = ap.parse_args()
    if args.iterations < 1 or args.max_pair_vph <= 1 or args.a <= 0 or args.c <= 0:
        ap.error('iterations>=1, max-pair-vph>1, a>0, c>0 are required')
    if args.pilot_scales and (args.resume or args.publish):
        ap.error('--pilot-scales cannot be combined with --resume/--publish')
    ensure_dirs()
    target_path = CALIB_ROOT / 'calibration_targets.csv'
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    target = pd.read_csv(target_path, dtype={'sensor_id': str})
    mapping = load_sensor_mapping()
    aligned = prepare_tables(target, mapping)
    net = locate_net_file()
    init = init_od(target, mapping, args.od_scale)
    show_audit(aligned, init)
    if args.audit:
        return

    if args.pilot_scales:
        records = []
        for factor in args.pilot_scales:
            if factor <= 0 or not np.isfinite(factor):
                raise ValueError('All pilot factors must be positive and finite')
            initial = init_od(target, mapping, factor)
            if np.any(initial > args.max_pair_vph):
                print(f'[WARNING] Pilot factor={factor:g} exceeds --max-pair-vph in OD '
                      '(pilot uses requested OD without clipping).', flush=True)
            label = f'factor_{factor:g}'.replace('.', 'p')
            met, detail = simulate_candidate(initial, args.seed, aligned, mapping, net,
                                              label, RUN_ROOT / 'pilots')
            row = dict(od_scale=factor, od_total_vph=float(initial.sum()), **met)
            records.append(row)
            detail.to_csv(RUN_ROOT / 'pilots' / label / 'detail.csv', index=False)
            pd.DataFrame(records).to_csv(RUN_ROOT / 'pilot_summary.csv', index=False)
        print('[PILOT] Summary ->', RUN_ROOT / 'pilot_summary.csv', flush=True)
        print(pd.DataFrame(records).to_string(index=False), flush=True)
        print('[STOP] Inspect flow_ratio, speed_loss and simulation logs before SPSA.', flush=True)
        return

    # Restrict a run-name to one path component; avoid overwriting old v1 state.
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        ap.error('run-name must be a simple file name')
    session = RUN_ROOT / 'runs' / args.run_name
    state_file = session / 'spsa_state.json'
    hist_file = session / 'spsa_history.csv'
    ident = signature(aligned, net, args)
    theta_init = od_to_theta(init)
    if args.resume:
        if not state_file.exists():
            raise FileNotFoundError(f'No v2 state at {state_file}; cannot resume old v1 state')
        st = json.loads(state_file.read_text())
        if st.get('signature') != ident or st.get('protocol') != PROTOCOL:
            raise RuntimeError('Resume signature mismatch (seed/scales/targets/network/code). '
                               'Use a new --run-name and start fresh.')
        theta = np.asarray(st['theta'], dtype=float)
        best_theta = np.asarray(st['best_theta'], dtype=float)
        best_loss = float(st['best_loss'])
        best_tag = str(st['best_tag'])
        rng = np.random.default_rng()
        rng.bit_generator.state = st['rng_state']
        start = int(st['iteration']) + 1
        history = pd.read_csv(hist_file).to_dict('records')
        print(f'[RESUME] iteration={start} best={best_loss:.6f}', flush=True)
    else:
        if state_file.exists() or hist_file.exists():
            raise FileExistsError(f'{session} has a previous run. Use --resume with same '
                                  'parameters, or change --run-name. No files overwritten.')
        session.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        theta = theta_init.copy()
        start = 1
        first, _ = simulate_candidate(init, args.seed, aligned, mapping, net,
                                      'initial', session / 'simulations')
        best_loss, best_theta, best_tag = first['loss'], theta.copy(), 'initial'
        history = []
        json_dump(state_file, dict(protocol=PROTOCOL, signature=ident,
                   iteration=0, theta=theta.tolist(), best_loss=best_loss,
                   best_theta=best_theta.tolist(), best_tag=best_tag,
                   rng_state=rng.bit_generator.state))
        print('[INITIAL] actual measured loss=', best_loss, flush=True)
    if start > args.iterations:
        print('[RESUME] No iterations left; evaluating previous best.', flush=True)
    for k in range(start, args.iterations + 1):
        ak = args.a / ((k + 10) ** 0.602)
        ck = args.c / (k ** 0.101)
        delta = rng.choice([-1., 1.], size=theta.shape)
        th_plus, th_minus = theta + ck * delta, theta - ck * delta
        met_plus, _ = simulate_candidate(theta_to_od(th_plus, init.shape, args.max_pair_vph),
                                          args.seed, aligned, mapping, net, f'iter{k:03d}_plus',
                                          session / 'simulations')
        met_minus, _ = simulate_candidate(theta_to_od(th_minus, init.shape, args.max_pair_vph),
                                           args.seed, aligned, mapping, net, f'iter{k:03d}_minus',
                                           session / 'simulations')
        for label, test_theta, met in [('plus', th_plus, met_plus), ('minus', th_minus, met_minus)]:
            if met['loss'] < best_loss:
                best_loss, best_theta, best_tag = met['loss'], test_theta.copy(), f'iter{k:03d}_{label}'
        g = ((met_plus['loss'] - met_minus['loss']) / (2.0 * ck)) * delta
        theta = np.clip(theta - ak * g, 0.0, np.log(args.max_pair_vph))
        row = dict(iteration=k, loss_plus=met_plus['loss'], loss_minus=met_minus['loss'],
                   loss_mean=0.5*(met_plus['loss']+met_minus['loss']),
                   flow_ratio_plus=met_plus['flow_ratio'], flow_ratio_minus=met_minus['flow_ratio'],
                   speed_loss_plus=met_plus['speed_loss'], speed_loss_minus=met_minus['speed_loss'],
                   flow_loss_plus=met_plus['flow_loss'], flow_loss_minus=met_minus['flow_loss'],
                   best_measured_loss=best_loss, best_tag=best_tag, a_k=ak, c_k=ck)
        history.append(row)
        pd.DataFrame(history).to_csv(hist_file, index=False)
        json_dump(state_file, dict(protocol=PROTOCOL, signature=ident, iteration=k,
                  theta=theta.tolist(), best_theta=best_theta.tolist(),
                  best_loss=best_loss, best_tag=best_tag,
                  rng_state=rng.bit_generator.state))
        print('[SPSA]', row, flush=True)
    od_best = theta_to_od(best_theta, init.shape, args.max_pair_vph)
    final, detail = simulate_candidate(od_best, args.seed, aligned, mapping, net,
                                       'final_verified', session / 'simulations')
    detail.to_csv(session / 'calibration_fit_detail.csv', index=False)
    record = dict(protocol=PROTOCOL, implementation_note=(
        'Deterministic 2x3 geographical zones; same OD in all 3 consecutive simulation '
        'hours used as proxies for PeMS 7/12/17 hour targets. Some sensors share a '
        'single simulated edge measurement. This is a controlled implementation '
        'choice, not a demonstrated realistic METR-LA 24-hour simulation.'),
        zone_grid=[2, 3], od_vph=od_best.tolist(), objective=final['loss'],
        speed_normalized_mae=final['speed_loss'], flow_normalized_mae=final['flow_loss'],
        flow_ratio=final['flow_ratio'], speed_coverage=final['speed_coverage'],
        calibrated_sensor_rows=final['rows'], calibrated_sensor_count=final['sensors'],
        unique_simulated_edges=final['unique_edges'],
        representative_hours=sorted(aligned.hour.unique().astype(int).tolist()),
        spsa_seed=args.seed, iterations=args.iterations, od_scale=args.od_scale,
        best_evaluated_tag=best_tag, best_measured_loss=best_loss)
    json_dump(session / 'calibrated_od.json', record)
    print('[DONE] Isolated candidate ->', session / 'calibrated_od.json', flush=True)
    print('[RESULT]', final, 'OD total=', float(od_best.sum()), flush=True)
    if args.publish:
        publish_outputs(session, final)
    else:
        print('[NOT PUBLISHED] Canonical calibrated_od.json remains unchanged. '
              'Use --resume --publish with same parameters after reviewing output.', flush=True)


if __name__ == '__main__':
    main()
