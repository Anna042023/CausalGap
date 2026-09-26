#!/usr/bin/env python3
"""One-pass, NO-SUMO check of existing SUMO–METR-LA H1 source readiness.

Does not alter simulator data, calibrated OD, publication files or model results.
It checks an alternative vehicle-counting definition (entered + departed) without
silently changing the research metric or interpreting it as proven correct.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path('/data/wanganna/CausalGap/sumo_metrla_builder')


def read_edge_counts(xml_path: Path, interval_start: float = 0.):
    """Read just the first SUMO hour. Reject duplicate interval/edge records."""
    result = {}
    for _, elem in ET.iterparse(str(xml_path), events=('end',)):
        if elem.tag == 'interval':
            start = float(elem.attrib.get('begin', 'nan'))
            if abs(start - interval_start) < 1e-6:
                for edge in elem.findall('edge'):
                    edge_id = edge.attrib.get('id')
                    if edge_id in result:
                        raise RuntimeError(f'Duplicate edge record in interval: {edge_id}')
                    result[edge_id] = (float(edge.attrib.get('entered', 0)),
                                       float(edge.attrib.get('departed', 0)))
            elem.clear()
    if not result:
        raise RuntimeError(f'No first-hour edge records in {xml_path}')
    return result


def audit_factor(folder: Path, factor: int):
    d = folder / f'factor_{factor}'
    detail_path = d / 'detail.csv'
    if not detail_path.exists():
        return None
    df = pd.read_csv(detail_path, dtype={'sensor_id':str, 'sumo_edge':str})
    needed = {'sensor_id', 'sumo_edge', 'flow_vph_sim', 'flow_vph_obs'}
    if not needed.issubset(df):
        raise RuntimeError(f'{detail_path}: missing {needed - set(df.columns)}')
    observed = float(df.flow_vph_obs.sum())
    measured = float(df.flow_vph_sim.sum())
    row = dict(factor=factor, sensors=int(df.sensor_id.nunique()),
               unique_edges=int(df.sumo_edge.nunique()),
               observed_sum_vph=observed, baseline_sim_sum_vph=measured,
               baseline_ratio=measured/observed if observed > 0 else None,
               missing_sumo_edges=None, departed_on_mapped_edges=None,
               entered_plus_departed_ratio=None)
    xml_path = d/'edgedata.xml'
    if xml_path.exists():
        counts = read_edge_counts(xml_path)
        row['missing_sumo_edges'] = int((~df.sumo_edge.isin(counts)).sum())
        departed = df.sumo_edge.map(lambda e: counts.get(e, (0., 0.))[1])
        # Shared edges are intentionally counted as repeated sensor-level proxies,
        # matching the existing 197-row objective; not independent observations.
        row['departed_on_mapped_edges'] = float(departed.sum())
        row['entered_plus_departed_ratio'] = float((measured+departed.sum())/observed)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--min-screen-flow-ratio', type=float, default=0.5,
                    help='A pragmatic STOP threshold, not a validated scientific calibration criterion')
    args = ap.parse_args()
    if not 0 < args.min_screen_flow_ratio <= 1:
        ap.error('--min-screen-flow-ratio must be in (0,1]')
    p = args.root/'calibration'/'spsa_calibration'/'fast_screen'/'first_1h'/'no_rerouting'
    rows = [audit_factor(p, f) for f in (1,2,4)]
    rows = [r for r in rows if r is not None]
    if not rows:
        raise RuntimeError(f'No existing factor detail.csv files under {p}')
    output = args.root/'calibration'/'spsa_calibration'/'finish52_readiness'
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'flow_accounting.csv', index=False)

    mapping_path = args.root/'mapping'/'sensor_to_sumo_edge.csv'
    matched = None
    if mapping_path.exists():
        m = pd.read_csv(mapping_path)
        if 'matched' in m:
            matched = int(m.matched.astype(str).str.lower().eq('true').sum())
    best = min(rows, key=lambda r: abs(1-r['baseline_ratio']))
    max_screen_ratio = max(r['baseline_ratio'] for r in rows)
    max_counting_ratio = max((r['entered_plus_departed_ratio'] for r in rows
                              if r['entered_plus_departed_ratio'] is not None), default=None)
    passed_screen = bool(max_screen_ratio >= args.min_screen_flow_ratio)
    full_nodes_ready = bool(matched == 207) if matched is not None else None
    report = {
        'protocol':'first-hour fast screen, NO rerouting, NOT PAPER CALIBRATION',
        'factors':rows,
        'best_screen_factor':best['factor'],
        'max_measured_flow_ratio':max_screen_ratio,
        'max_entered_plus_departed_ratio_exploratory':max_counting_ratio,
        'matched_model_sensors_of_207':matched,
        'all_207_mapped':full_nodes_ready,
        'screen_flow_threshold':args.min_screen_flow_ratio,
        'screen_flow_gate_pass':passed_screen,
        'ready_to_claim_paper_calibrated':False,
        'ready_to_run_full_61_file_pipeline':False,
        'reason':'Existing factors use short non-rerouted diagnostics, not 3-hour official calibration; full 207-node mapping and a properly validated OD remain necessary.'
    }
    (output/'readiness.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('===== EXISTING RUNS ONLY: NO SUMO =====')
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'\nMatched of 207 model sensors: {matched if matched is not None else "mapping file unavailable"}')
    print('Flow screen gate:', 'PASS' if passed_screen else 'FAIL', '(diagnostic only)')
    print('Ready for a paper-calibrated SUMO claim: NO')
    print('Do NOT run 20 SPSA iterations or 61-file generation on the current OD.')
    print('Saved:', output/'readiness.json')


if __name__ == '__main__':
    main()
