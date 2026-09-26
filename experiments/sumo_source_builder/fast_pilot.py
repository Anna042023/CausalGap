#!/usr/bin/env python3
"""Isolated, reproducible SHORT SUMO demand screen. Not a paper calibration.

Uses unchanged v2 mapping and metric functions, but simulates only the first
representative hour and omits dynamic rerouting by default. Therefore outputs
MUST NOT be compared numerically with the old three-hour, rerouting pilot.
To compare OD multipliers, run *both* scales with this same protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (ensure_dirs, load_sensor_mapping, locate_net_file, sumo_home,
                    write_routes_from_od, create_edge_data_additional, parse_edge_data)
from config import CALIB_ROOT

spec = importlib.util.spec_from_file_location('spsa_v2_core', HERE / '04_calibrate_demand_spsa.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

SCREEN_VERSION = 'isolated-fast-screen-v1-no-paper-publish'
OUTPUT_ROOT = CALIB_ROOT / 'spsa_calibration' / 'fast_screen'


def atomic_json(path: Path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def hash_file(p: Path):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def config_fingerprint(net, target, mapping, od, args):
    # Every change affecting the interpretation of a completed run invalidates reuse.
    metadata = dict(version=SCREEN_VERSION, factor=float(args.factor),
                    hours=int(args.screen_hours), rerouting=bool(args.official_rerouting),
                    seed=int(args.seed), sumo_step_length=1, period=3600,
                    net=str(net.resolve()), net_size=net.stat().st_size,
                    net_mtime_ns=net.stat().st_mtime_ns,
                    od_sha256=hashlib.sha256(np.ascontiguousarray(od, dtype='<f8').tobytes()).hexdigest(),
                    target_sha256=hash_file(target), mapping_sha256=hash_file(mapping),
                    code_sha256=hash_file(HERE / 'fast_pilot.py'),
                    core_sha256=hash_file(HERE / '04_calibrate_demand_spsa.py'),
                    common_sha256=hash_file(HERE / 'common.py'),
                    config_sha256=hash_file(HERE / 'config.py'))
    metadata['fingerprint'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return metadata


def candidate_command(net, routes, add, duration, seed, official_rerouting=False):
    cmd = [str(sumo_home() / 'bin' / 'sumo'), '-n', str(net), '-r', str(routes),
           '-a', str(add), '--begin', '0', '--end', str(duration),
           '--step-length', '1', '--seed', str(int(seed)), '--no-step-log', 'true',
           '--time-to-teleport', '300']
    if official_rerouting:
        cmd.extend(['--device.rerouting.probability', '1.0',
                    '--device.rerouting.period', '120'])
    return cmd


def run_sumo(cmd, log, heartbeat=30):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('[SUMO] Starting; terminal updates show WALL time, not simulated time.', flush=True)
    print('[SUMO] Log:', log, flush=True)
    started = time.monotonic()
    with log.open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    rc = proc.wait(timeout=heartbeat)
                    break
                except subprocess.TimeoutExpired:
                    print(f'[HEARTBEAT] SUMO alive; elapsed={int(time.monotonic()-started)}s '
                          f'pid={proc.pid}; log={log}', flush=True)
        except KeyboardInterrupt:
            print('[INTERRUPTED] Stopping only this SUMO child. Existing results retained.', flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if rc != 0:
        raise RuntimeError(f'SUMO exit code={rc}; inspect {log}')
    print(f'[SUMO DONE] wall={int(time.monotonic()-started)}s', flush=True)


def evaluate_factor(args, aligned, target, mapping, net):
    factor = args.factor
    label = f'factor_{factor:g}'.replace('.', 'p')
    folder = OUTPUT_ROOT / f'first_{args.screen_hours}h' / (
        'official_reroute120' if args.official_rerouting else 'no_rerouting') / label
    folder.mkdir(parents=True, exist_ok=True)
    od = core.init_od(target, mapping, factor)
    source_target = CALIB_ROOT / 'calibration_targets.csv'
    source_map = CALIB_ROOT.parent / 'mapping' / 'sensor_to_sumo_edge.csv'
    manifest = config_fingerprint(net, source_target, source_map, od, args)
    manifest_file = folder / 'manifest.json'
    result_file = folder / 'result.json'
    detail_file = folder / 'detail.csv'
    routes_file = folder / 'routes.rou.xml'
    existing = json.loads(manifest_file.read_text()) if manifest_file.exists() else None
    same = existing is not None and existing.get('fingerprint') == manifest['fingerprint']
    if result_file.exists() and detail_file.exists() and same:
        result = json.loads(result_file.read_text())
        print(f'[REUSE COMPLETE] {label}: {result}', flush=True)
        return result
    if existing and not same:
        raise RuntimeError(f'Protocol/input change at {folder}; do not reuse old route/data. '
                           'Move that directory away or use the original matching inputs.')
    if not manifest_file.exists():
        atomic_json(manifest_file, manifest)
    duration = 3600 * args.screen_hours
    if not routes_file.exists() or routes_file.stat().st_size == 0:
        print(f'[ROUTES] {label} OD={od.sum():.1f} veh/h (once)', flush=True)
        write_routes_from_od(net, mapping, od, duration, args.seed, routes_file)
    else:
        print(f'[REUSE ROUTES] {routes_file}', flush=True)
    add = folder / 'sensors.add.xml'
    edgedata = folder / 'edgedata.xml'
    create_edge_data_additional(mapping, add, edgedata, period=3600)
    if edgedata.exists():
        edgedata.unlink()  # Never interpret a partially-written previous SUMO result.
    cmd = candidate_command(net, routes_file, add, duration, args.seed, args.official_rerouting)
    atomic_json(folder / 'sumo_command.json', cmd)
    run_sumo(cmd, folder / 'sumo.log', heartbeat=args.heartbeat)
    if not edgedata.exists():
        raise RuntimeError(f'SUMO produced no edgeData at {edgedata}')
    only_hours = sorted(aligned.hour.unique().tolist())[:args.screen_hours]
    subset = aligned.loc[aligned.hour.isin(only_hours)].copy()
    detail = core.edge_data_to_detail(subset, parse_edge_data(edgedata), duration_per_hour=3600)
    met = core.evaluate_detail(detail)
    result = dict(od_scale=factor, od_total_vph=float(od.sum()), **met,
                  hours_screened=only_hours, rerouting='official_120s' if args.official_rerouting else 'off',
                  use_for_paper=False, protocol=SCREEN_VERSION)
    atomic_csv(detail, detail_file)
    atomic_json(result_file, result)  # Completion marker written LAST.
    print('[RESULT]', result, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scales', nargs='+', type=float, default=[1, 2])
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--screen-hours', type=int, choices=[1, 2, 3], default=1)
    ap.add_argument('--official-rerouting', action='store_true',
                    help='Use original costly rerouting, for protocol-matched short screens.')
    ap.add_argument('--heartbeat', type=int, default=30)
    ap.add_argument('--audit-only', action='store_true')
    args = ap.parse_args()
    if not args.scales or any(x <= 0 or not np.isfinite(x) for x in args.scales):
        ap.error('All OD scales must be finite and >0')
    if args.heartbeat < 1:
        ap.error('--heartbeat must be >= 1')
    ensure_dirs()
    source_target = CALIB_ROOT / 'calibration_targets.csv'
    target = pd.read_csv(source_target, dtype={'sensor_id': str})
    mapping = load_sensor_mapping()
    aligned = core.prepare_tables(target, mapping)
    net = locate_net_file()
    print('[SCREEN] one 1-second-step simulation per factor; first representative hours only.', flush=True)
    print('[SCREEN] Protocol:', 'original rerouting 120s' if args.official_rerouting else 'NO dynamic rerouting', flush=True)
    print('[SCREEN] Existing factor_1 3-hour pilot is NOT an apples-to-apples comparison.', flush=True)
    print('[SCREEN] Output isolated:', OUTPUT_ROOT, flush=True)
    if args.audit_only:
        for f in args.scales:
            print('[AUDIT] factor=', f, 'OD veh/h=', core.init_od(target, mapping, f).sum(), flush=True)
        return
    results = []
    for f in args.scales:
        args.factor = f
        results.append(evaluate_factor(args, aligned, target, mapping, net))
        summary = (OUTPUT_ROOT / f'first_{args.screen_hours}h' /
                   ('official_reroute120' if args.official_rerouting else 'no_rerouting') /
                   'screen_summary.csv')
        atomic_csv(pd.DataFrame(results), summary)
        print('[SUMMARY SAVED]', summary, flush=True)
    print('[SCREEN COMPLETE] Diagnostic only; official full-duration validation is still required.', flush=True)


if __name__ == '__main__':
    main()
