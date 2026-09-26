#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from config import *
from common import ensure_dirs, load_locations, locate_net_file, json_dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--radius', type=float, default=MAX_SENSOR_EDGE_DISTANCE_M)
    args = ap.parse_args()
    ensure_dirs()
    import sumolib
    net_file = locate_net_file()
    net = sumolib.net.readNet(str(net_file), withInternal=False)
    loc = load_locations()
    rows = []
    for r in loc.itertuples(index=False):
        x, y = net.convertLonLat2XY(float(r.longitude), float(r.latitude))
        cand = net.getNeighboringEdges(x, y, float(args.radius), includeJunctions=False)
        filtered = []
        for edge, dist in cand:
            if edge.isSpecial():
                continue
            allows = edge.allows('passenger')
            if allows:
                filtered.append((edge, float(dist)))
        if filtered:
            edge, dist = min(filtered, key=lambda t: t[1])
            rows.append(dict(sensor_id=str(r.sensor_id), latitude=float(r.latitude), longitude=float(r.longitude), matched=True,
                             sumo_edge=edge.getID(), distance_m=dist,
                             from_node=edge.getFromNode().getID(), to_node=edge.getToNode().getID(),
                             speed_limit_mps=float(edge.getSpeed())))
        else:
            rows.append(dict(sensor_id=str(r.sensor_id), latitude=float(r.latitude), longitude=float(r.longitude), matched=False,
                             sumo_edge='', distance_m=np.nan, from_node='', to_node='', speed_limit_mps=np.nan))
    out = pd.DataFrame(rows)
    p = MAPPING_ROOT / 'sensor_to_sumo_edge.csv'
    out.to_csv(p, index=False)
    matched = out[out.matched == True]  # noqa: E712
    print('network             :', net_file)
    print('sensors             :', len(out))
    print('matched <= radius   :', len(matched))
    print('unmatched           :', int((~out.matched).sum()))
    if len(matched):
        print('median distance (m) :', float(matched.distance_m.median()))
        print('max distance (m)    :', float(matched.distance_m.max()))
    print('output              :', p)
    if len(matched) < 180:
        raise RuntimeError('Too few METR-LA sensors mapped to the SUMO road network. Increase OSM bbox or inspect network import before continuing.')

if __name__ == '__main__':
    main()
