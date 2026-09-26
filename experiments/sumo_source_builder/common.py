from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import *


def ensure_dirs():
    for p in [WORK_ROOT, NETWORK_ROOT, MAPPING_ROOT, CALIB_ROOT, ROUTE_ROOT, LOG_ROOT, SOURCE_ROOT]:
        p.mkdir(parents=True, exist_ok=True)


def safe_name(s: str) -> str:
    return str(s).replace('/', '_').replace(' ', '_')


def json_dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding='utf-8')


def _json_default(x):
    if isinstance(x, Path): return str(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None, check=True):
    print('[CMD]', ' '.join(map(str, cmd)), flush=True)
    p = subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None, env=env, check=False)
    if check and p.returncode != 0:
        raise RuntimeError(f'Command failed with code {p.returncode}: {cmd}')
    return p.returncode


def sumo_home() -> Path:
    x = os.environ.get('SUMO_HOME')
    if not x:
        try:
            import sumo
            x = sumo.SUMO_HOME
        except Exception as e:
            raise RuntimeError('SUMO_HOME is not set and python package sumo could not provide it') from e
    p = Path(x)
    if not (p / 'bin' / 'sumo').exists():
        raise RuntimeError(f'Invalid SUMO_HOME={p}')
    return p


def load_locations() -> pd.DataFrame:
    df = pd.read_csv(LOCATIONS_CSV)
    need = {'sensor_id', 'latitude', 'longitude'}
    if not need.issubset(df.columns):
        raise ValueError(f'{LOCATIONS_CSV} missing columns {need - set(df.columns)}')
    df['sensor_id'] = df['sensor_id'].astype(str)
    return df


def load_metr_speed() -> pd.DataFrame:
    df = pd.read_hdf(H5_PATH)
    df.columns = [str(x) for x in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def locate_net_file() -> Path:
    candidates = [
        NETWORK_ROOT / 'metrla.net.xml.gz',
        NETWORK_ROOT / 'metrla.net.xml',
        NETWORK_ROOT / 'osm.net.xml.gz',
        NETWORK_ROOT / 'osm.net.xml',
    ]
    candidates += sorted(NETWORK_ROOT.glob('*.net.xml.gz')) + sorted(NETWORK_ROOT.glob('*.net.xml'))
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f'No SUMO net file under {NETWORK_ROOT}')


def load_sensor_mapping() -> pd.DataFrame:
    p = MAPPING_ROOT / 'sensor_to_sumo_edge.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, dtype={'sensor_id': str, 'sumo_edge': str})
    return df


def load_model_adj_and_ids():
    sys.path.insert(0, str(UNIFIED_ROOT))
    from causalgap.data import load_dataset
    b = load_dataset('METR-LA', history_len=12, pred_len=12)
    loc = load_locations()
    # Prefer official graph order file for sensor IDs; it must match the 207 model nodes.
    ids_path = SENSOR_GRAPH_ROOT / 'graph_sensor_ids.txt'
    txt = ids_path.read_text().strip()
    ids = [x.strip() for line in txt.splitlines() for x in line.split(',') if x.strip()]
    if len(ids) != int(b.num_nodes):
        raise RuntimeError(f'graph_sensor_ids has {len(ids)} ids but model bundle has {b.num_nodes} nodes')
    return np.asarray(b.adj, dtype=np.float32), ids


def make_graph_intervention(kind: str, spec: dict, seed: int):
    sys.path.insert(0, str(UNIFIED_ROOT))
    from causalgap.interventions import make_intervention
    a0, ids = load_model_adj_and_ids()
    n = a0.shape[0]
    x = np.zeros((1, 12, n, 1), dtype=np.float32)
    y = np.zeros_like(x)
    inter = make_intervention(
        kind, x, y, a0, seed=int(seed), y_mask=np.ones_like(y, dtype=bool),
        severity=spec.get('severity'), weight_gamma=spec.get('gamma'),
        edge_removal_mode='paper_top40_uniform', edge_pool_ratio=0.40,
    )
    return inter, a0, ids


def zones_from_mapping(mapping: pd.DataFrame, rows=2, cols=3):
    df = mapping[mapping['matched'] == True].copy()  # noqa: E712
    # Deterministic geographic grid; avoids introducing an extra learned clustering dependency.
    lat_bins = np.linspace(df.latitude.min() - 1e-9, df.latitude.max() + 1e-9, rows + 1)
    lon_bins = np.linspace(df.longitude.min() - 1e-9, df.longitude.max() + 1e-9, cols + 1)
    df['r'] = np.clip(np.digitize(df.latitude, lat_bins) - 1, 0, rows - 1)
    df['c'] = np.clip(np.digitize(df.longitude, lon_bins) - 1, 0, cols - 1)
    df['zone'] = (df['r'] * cols + df['c']).astype(int)
    return df


def write_routes_from_od(net_file: Path, mapping: pd.DataFrame, od: np.ndarray,
                         duration_s: int, seed: int, out_routes: Path,
                         begin_s: int = 0, type_id='passenger'):
    """Create OD trips then let duarouter compute valid routes on the baseline network.

    Counts are deterministic under a fixed seed. The same route file is reused for paired
    baseline/intervention runs; all active vehicles are rerouted after t0 under the intervention.
    """
    import random
    import re
    import xml.etree.ElementTree as ET
    rng = np.random.default_rng(int(seed))
    zdf = zones_from_mapping(mapping)
    zones = sorted(zdf.zone.unique().tolist())
    edges_by_zone = {z: zdf.loc[zdf.zone == z, 'sumo_edge'].drop_duplicates().tolist() for z in zones}
    if not zones:
        raise RuntimeError('No mapped zones')

    # CONNECTED_OD_PAIRS_FIX_V2
    # Preserve zone-level OD demand.
    # Select passenger-reachable directed road pairs.

    import sumolib
    import json
    import hashlib

    edge_ids = sorted({
        str(e)
        for z in zones
        for e in edges_by_zone[z]
    })

    stat = net_file.stat()

    cache_signature = {
        "version": 'CONNECTED_OD_PAIRS_FIX_V2',
        "network": str(net_file.resolve()),
        "size": stat.st_size,
        "mtime": stat.st_mtime_ns,
        "zones": {
            str(z): sorted(
                str(e) for e in edges_by_zone[z]
            )
            for z in zones
        },
    }

    signature = hashlib.sha256(
        json.dumps(
            cache_signature,
            sort_keys=True
        ).encode()
    ).hexdigest()[:20]

    cache_file = (
        net_file.parent
        / f"connected_od_{signature}.json"
    )

    if cache_file.exists():

        print(
            "[OD] Loading connectivity cache",
            flush=True
        )

        legal_pairs = json.loads(
            cache_file.read_text()
        )

    else:

        print(
            "[OD] Reading network and checking connectivity",
            flush=True
        )

        net = sumolib.net.readNet(
            str(net_file)
        )

        allowed = {
            eid
            for eid in edge_ids
            if net.getEdge(eid).allows("passenger")
        }

        reachable_by_source = {}

        for idx, eid in enumerate(
            edge_ids,
            start=1
        ):

            if eid not in allowed:

                reachable_by_source[eid] = set()

                continue

            reachable = net.getReachable(
                net.getEdge(eid),
                vclass="passenger"
            )

            reachable_by_source[eid] = {
                e.getID()
                for e in reachable
                if e.getID() in allowed
            }

            if idx % 20 == 0:

                print(
                    f"[OD] Checked {idx}/{len(edge_ids)} edges",
                    flush=True
                )

        legal_pairs = {}

        for oz in zones:

            for dz in zones:

                key = f"{oz}|{dz}"

                pairs = []

                for fr in edges_by_zone[oz]:

                    fr = str(fr)

                    reachable = (
                        reachable_by_source[fr]
                    )

                    for to in edges_by_zone[dz]:

                        to = str(to)

                        if fr == to:
                            continue

                        if to in reachable:

                            pairs.append(
                                [fr, to]
                            )

                legal_pairs[key] = pairs

        cache_file.write_text(
            json.dumps(legal_pairs)
        )

        print(
            "[OD] Connectivity cache saved",
            flush=True
        )

    for oi, oz in enumerate(zones):

        for di, dz in enumerate(zones):

            if (
                oi >= od.shape[0]
                or di >= od.shape[1]
            ):
                continue

            rate = float(od[oi, di])

            if rate <= 0:
                continue

            key = f"{oz}|{dz}"

            if not legal_pairs.get(key):

                raise RuntimeError(
                    f"Positive OD {key} has no "
                    "passenger-reachable road pair. "
                    "Demand must not be silently dropped."
                )

    print(
        "[OD] Connectivity preparation complete",
        flush=True
    )

    trips = out_routes.with_suffix('.trips.xml')
    trips.parent.mkdir(parents=True, exist_ok=True)
    rows_xml = ['<routes>', f'  <vType id="{type_id}" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5"/>']
    vid = 0
    hours = max(duration_s / 3600.0, 1e-9)
    for oi, oz in enumerate(zones):
        for di, dz in enumerate(zones):
            if oi == di or oi >= od.shape[0] or di >= od.shape[1]:
                continue
            rate = max(0.0, float(od[oi, di]))  # veh/hour
            count = int(round(rate * hours))
            if count <= 0 or not edges_by_zone[oz] or not edges_by_zone[dz]:
                continue
            departs = np.sort(rng.uniform(begin_s, begin_s + duration_s, size=count))
            for depart in departs:
                pairs = legal_pairs[f"{oz}|{dz}"]
                fr, to = pairs[
                    int(rng.integers(len(pairs)))
                ]
                rows_xml.append(
                    f'  <trip id="veh{vid}" type="{type_id}" depart="{depart:.2f}" from="{fr}" to="{to}" '
                    'departLane="best" departSpeed="max"/>'
                )
                vid += 1
    # GLOBAL_DEPARTURE_SORT_FIX_V1
    # Sort all trips globally, not separately within each OD pair.

    def _trip_depart(row):
        element = ET.fromstring(row.strip())

        if element.tag != "trip":
            raise RuntimeError(
                f"Expected trip element, got {element.tag}"
            )

        depart = element.get("depart")

        if depart is None:
            raise RuntimeError(
                f"Trip has no departure time: {row}"
            )

        return float(depart)

    header = rows_xml[:2]
    trip_rows = rows_xml[2:]

    trip_rows = sorted(
        trip_rows,
        key=_trip_depart
    )

    rows_xml = header + trip_rows

    rows_xml.append('</routes>')
    trips.write_text('\n'.join(rows_xml), encoding='utf-8')
    duarouter = sumo_home() / 'bin' / 'duarouter'
    run([
        str(duarouter), '-n', str(net_file), '-r', str(trips), '-o', str(out_routes),
        '--ignore-errors', 'true', '--routing-algorithm', 'dijkstra', '--seed', str(int(seed)),
    ])
    # Sort routed vehicles globally as well.

    tree = ET.parse(out_routes)
    root = tree.getroot()

    vehicles = [
        child
        for child in root
        if child.tag == 'vehicle'
    ]

    other_elements = [
        child
        for child in root
        if child.tag != 'vehicle'
    ]

    vehicles = sorted(
        vehicles,
        key=lambda v: float(
            v.attrib.get('depart', 'inf')
        )
    )

    root[:] = other_elements + vehicles

    tree.write(
        out_routes,
        encoding='utf-8',
        xml_declaration=True
    )

    generated = len(trip_rows)
    routed = len(vehicles)
    unrouted = generated - routed

    success = (
        100.0 * routed / generated
        if generated > 0
        else 0.0
    )

    print(
        f"[ROUTE AUDIT] "
        f"generated={generated} "
        f"routed={routed} "
        f"unrouted={unrouted} "
        f"success={success:.2f}%",
        flush=True
    )

    if unrouted > 0:
        print(
            "[ROUTE WARNING] "
            "Some OD trips have no valid route. "
            "Do not treat this as a completed "
            "calibration.",
            flush=True
        )


    if generated != routed:

        raise RuntimeError(
            f"Routing incomplete: "
            f"{generated} generated, "
            f"{routed} routed, "
            f"{unrouted} unrouted."
        )

    return trips, out_routes


def create_edge_data_additional(mapping: pd.DataFrame, path: Path, output_file: Path, period=3600):
    edges = mapping.loc[mapping.matched == True, 'sumo_edge'].drop_duplicates().tolist()  # noqa: E712
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = [
        '<additional>',
        f'  <edgeData id="metrlaSensors" file="{output_file}" period="{int(period)}" excludeEmpty="false" edges="{" ".join(edges)}"/>',
        '</additional>'
    ]
    path.write_text('\n'.join(txt), encoding='utf-8')


def parse_edge_data(path: Path):
    tree = ET.parse(path)
    rows = []
    for interval in tree.getroot().iter('interval'):
        begin = float(interval.attrib.get('begin', 0))
        end = float(interval.attrib.get('end', 0))
        for e in interval.findall('edge'):
            rows.append({
                'begin': begin, 'end': end, 'edge': e.attrib.get('id'),
                'speed_mps': float(e.attrib.get('speed', '-1')),
                'entered': float(e.attrib.get('entered', '0')),
                'departed': float(e.attrib.get('departed', '0')),
                'arrived': float(e.attrib.get('arrived', '0')),
            })
    return pd.DataFrame(rows)


def write_pems_template(path: Path):
    loc = load_locations()
    pd.DataFrame(columns=['sensor_id', 'timestamp', 'flow']).to_csv(path, index=False)
    (path.parent / 'pems_station_ids.txt').write_text('\n'.join(loc.sensor_id.astype(str)), encoding='utf-8')


def read_pems_flow(path: Path):
    df = pd.read_csv(path, dtype={'sensor_id': str})
    need = {'sensor_id', 'timestamp', 'flow'}
    if not need.issubset(df.columns):
        raise ValueError(f'{path} must contain {sorted(need)}')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['flow'] = pd.to_numeric(df['flow'], errors='coerce')
    return df.dropna(subset=['flow'])


def build_target_table(pems_csv: Path, mapping: pd.DataFrame, hours=(7, 12, 17)):
    sp = load_metr_speed()
    split = int(len(sp) * 0.7)
    train = sp.iloc[:split]
    flow = read_pems_flow(pems_csv)
    train_end = train.index.max()
    train_start = train.index.min()
    flow = flow[(flow.timestamp >= train_start) & (flow.timestamp <= train_end)]
    common = set(train.columns) & set(flow.sensor_id.astype(str)) & set(mapping.loc[mapping.matched == True, 'sensor_id'].astype(str))  # noqa: E712
    if len(common) < 50:
        raise RuntimeError(f'Only {len(common)} sensors have speed+flow+SUMO mapping; paper mode requires a substantial aligned set.')
    rows = []
    for h in hours:
        s = train[train.index.hour == int(h)]
        f = flow[flow.timestamp.dt.hour == int(h)]
        fs = f.groupby('sensor_id')['flow'].median()
        sm = s[list(sorted(common))].median(axis=0)
        for sid in sorted(common):
            if sid in fs.index and np.isfinite(fs[sid]) and np.isfinite(sm[sid]):
                rows.append({'hour': int(h), 'sensor_id': sid, 'speed_mph': float(sm[sid]), 'flow_vph': float(fs[sid])})
    return pd.DataFrame(rows)


def make_initial_od(target: pd.DataFrame, mapping: pd.DataFrame, rows=2, cols=3,
                    demand_fraction=0.12):
    """Initial, NOT calibrated, zone OD in vehicles/hour.

    demand_fraction reproduces the original 0.12 assumption by default.
    Use a measured pilot to examine alternative scales; PeMS station flows
    are not an OD trip count and must not be summed as such.
    """
    if not np.isfinite(demand_fraction) or demand_fraction <= 0:
        raise ValueError('demand_fraction must be positive and finite')
    zdf = zones_from_mapping(mapping, rows=rows, cols=cols)
    sid2zone = dict(zip(zdf.sensor_id.astype(str), zdf.zone.astype(int)))
    t = target.copy()
    t['zone'] = t.sensor_id.map(sid2zone)
    zflow = t.dropna(subset=['zone']).groupby('zone').flow_vph.median()
    zones = sorted(zdf.zone.unique().tolist())
    K = len(zones)
    base = np.zeros((K, K), dtype=np.float64)
    for i, zi in enumerate(zones):
        out = max(float(zflow.get(zi, zflow.median() if len(zflow) else 1000.0)), 50.0)
        for j in range(K):
            if i != j:
                base[i, j] = out / max(K - 1, 1) * float(demand_fraction)
    return base
