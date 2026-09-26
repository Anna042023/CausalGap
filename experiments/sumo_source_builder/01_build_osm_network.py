#!/usr/bin/env python3
from pathlib import Path
import argparse, os, shutil, subprocess, sys
import pandas as pd
from config import *
from common import ensure_dirs, load_locations, run, sumo_home


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pad-deg', type=float, default=BBOX_PAD_DEG)
    ap.add_argument('--tiles', type=int, default=OSM_TILES)
    ap.add_argument('--force-download', action='store_true')
    args = ap.parse_args()
    ensure_dirs()
    loc = load_locations()
    west = float(loc.longitude.min() - args.pad_deg)
    east = float(loc.longitude.max() + args.pad_deg)
    south = float(loc.latitude.min() - args.pad_deg)
    north = float(loc.latitude.max() + args.pad_deg)
    bbox = f'{west:.6f},{south:.6f},{east:.6f},{north:.6f}'
    print('METR-LA sensor bbox:', bbox)
    print('sensor ranges:', loc.latitude.min(), loc.latitude.max(), loc.longitude.min(), loc.longitude.max())
    sh = sumo_home()
    osmget = sh / 'tools' / 'osmGet.py'
    osmbuild = sh / 'tools' / 'osmBuild.py'
    osm = NETWORK_ROOT / 'metrla_bbox.osm.xml.gz'
    net = NETWORK_ROOT / 'metrla.net.xml.gz'

    if args.force_download:
        for p in NETWORK_ROOT.glob('metrla*osm*'):
            p.unlink(missing_ok=True)
        net.unlink(missing_ok=True)

    # SUMO 1.18 osmGet supports the WebWizard-compatible -b/-p/-d/-z syntax.
    if not osm.exists():
        run([sys.executable, str(osmget), f'--bbox={bbox}', '-p', 'metrla', '-d', str(NETWORK_ROOT), '-z'])
        generated = list(NETWORK_ROOT.glob('metrla*bbox*.osm.xml.gz')) + list(NETWORK_ROOT.glob('metrla*.osm.xml.gz'))
        if not generated:
            raise RuntimeError('osmGet finished but no OSM file appeared. Check network access / Overpass service.')
        src = generated[0]
        if src != osm:
            shutil.copy2(src, osm)
    else:
        print('[REUSE]', osm)

    if not net.exists():
        # Passenger-only keeps the simulation network manageable and relevant to road traffic.
        run([
            sys.executable, str(osmbuild), '-f', str(osm), '-p', 'metrla', '-d', str(NETWORK_ROOT), '-z',
            '--vehicle-classes', 'passenger',
            '--netconvert-options', 'geometry.remove,roundabouts.guess,ramps.guess,junctions.join,tls.guess-signals,tls.discard-simple,tls.join',
        ])
        generated = list(NETWORK_ROOT.glob('metrla*.net.xml.gz')) + list(NETWORK_ROOT.glob('metrla*.net.xml'))
        if not generated:
            raise RuntimeError('osmBuild finished but no net file appeared')
        src = generated[0]
        if src != net:
            shutil.copy2(src, net)
    else:
        print('[REUSE]', net)

    print('[DONE] network:', net)

if __name__ == '__main__':
    main()
