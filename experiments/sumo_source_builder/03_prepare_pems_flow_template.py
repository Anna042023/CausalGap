#!/usr/bin/env python3
import argparse
from config import *
from common import ensure_dirs, write_pems_template, read_pems_flow, load_sensor_mapping, build_target_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default=str(Pems_FLOW_CSV))
    ap.add_argument('--hours', nargs='+', type=int, default=[7, 12, 17])
    args = ap.parse_args()
    ensure_dirs()
    p = Path(args.input)
    if not p.exists():
        write_pems_template(p)
        print('[STOP] Paper-mode calibration requires real Caltrans PeMS hourly flow.')
        print('Template written:', p)
        print('Station list     :', p.parent / 'pems_station_ids.txt')
        print('Required columns : sensor_id,timestamp,flow')
        print('Export the hourly PeMS flow for these station IDs over the METR-LA period, then rerun this script.')
        raise SystemExit(2)
    flow = read_pems_flow(p)
    mapping = load_sensor_mapping()
    target = build_target_table(p, mapping, hours=tuple(args.hours))
    out = CALIB_ROOT / 'calibration_targets.csv'
    target.to_csv(out, index=False)
    print('flow rows               :', len(flow))
    print('aligned calibration rows:', len(target))
    print('aligned sensors         :', target.sensor_id.nunique())
    print('hours                   :', sorted(target.hour.unique().tolist()))
    print('[DONE]', out)

if __name__ == '__main__':
    from pathlib import Path
    main()
