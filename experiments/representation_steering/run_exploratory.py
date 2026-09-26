#!/usr/bin/env python3
"""H2 v3: calibration-only audit and SEPARATE exploratory follow-up.

This NEVER edits the original v2 H2 folder, selected model, or seed0--4 outputs.
Do not call a new strong-test run 'confirmatory': v2 strong outcomes were seen.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DEFAULT_PROJECT=Path('/data/wanganna/CausalGap')

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=DEFAULT_PROJECT)
    ap.add_argument('--experiment-root',type=Path)
    ap.add_argument('--original-h2-root',type=Path)
    ap.add_argument('--output-root',type=Path)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--mode',required=True,choices=['calibration-only','exploratory-strong'])
    ap.add_argument('--arch',choices=['mlp','film'],default='film')
    ap.add_argument('--teacher-weight',type=float,default=0.01)
    ap.add_argument('--head-epochs',type=int,default=20)
    ap.add_argument('--head-patience',type=int,default=5)
    ap.add_argument('--head-batch-size',type=int,default=8)
    ap.add_argument('--hidden',type=int,default=32)
    ap.add_argument('--max-strong-windows',type=int,default=32)
    ap.add_argument('--perturb-ratio',type=float,default=.05)
    args=ap.parse_args()
    if args.teacher_weight<0:ap.error('--teacher-weight must be >=0')
    project=args.project_root
    original=args.original_h2_root or project/'experiment_results'/'representation_steering'
    exp=args.experiment_root or project/'experiment_results'/'incremental_value'
    lock=original/'h2_selected_model.json'
    if not lock.exists(): raise FileNotFoundError(lock)
    fixed=json.loads(lock.read_text())
    if fixed['selected_model']!='DLinear':
        raise RuntimeError('This archived experiment selects DLinear; do not reselect on strong outcomes.')
    if fixed['model_seeds'] != [0,1,2,3,4]:
        raise RuntimeError('Expected 5 already-frozen model seeds')
    if fixed.get('steering_config')!=dict(head_epochs=args.head_epochs,head_patience=args.head_patience,
       head_batch_size=args.head_batch_size,hidden=args.hidden,max_strong_windows=args.max_strong_windows,
       perturb_ratio=args.perturb_ratio):
        raise RuntimeError('Old frozen hyperparameters changed. Keep fixed for this matched-capacity audit.')
    variant=f'{args.arch}_teacher{args.teacher_weight:g}'
    output=args.output_root or project/'experiment_results'/'representation_steering'/variant
    if output.resolve()==original.resolve() or original.resolve() in output.resolve().parents:
        raise RuntimeError('Refusing to modify original frozen H2 directory')
    output.mkdir(parents=True,exist_ok=True)
    (output/'logs').mkdir(exist_ok=True)
    print('MODE:',args.mode,'VARIANT:',variant,'OUTPUT:',output,flush=True)
    for seed in fixed['model_seeds']:
        ck=exp/'runs'/'synthetic'/'DLinear'/'trackB'/f'seed{seed}'/'checkpoint.pt'
        if sha256(ck)!=fixed['checkpoint_sha256_by_seed'][str(seed)]:
            raise RuntimeError(f'Checkpoint mismatch seed{seed}')
    for seed in fixed['model_seeds']:
        run=output/'DLinear'/f'seed{seed}'
        wanted=run/('h2_head_calibration_only.json' if args.mode=='calibration-only' else 'h2_paired_steering.csv')
        if wanted.exists():
            raise RuntimeError(f'Refusing silent overwrite; use a new --output-root: {wanted}')
        cmd=[sys.executable,'-u',str(HERE/'run_steering.py'),
           '--model','DLinear','--setting','synthetic','--model-seed',str(seed),
           '--device',args.device,'--project-root',str(project),'--experiment-root',str(exp),
           '--output-root',str(output),'--selection-file',str(lock),
           '--profile',fixed['profile'],
           '--cal-seeds',*map(str,fixed['cal_topology_seeds']),
           '--eval-seeds','0','1','2','3','4',
           '--max-cal-windows',str(fixed['cal_windows']),
           '--max-strong-windows',str(args.max_strong_windows),
           '--head-epochs',str(args.head_epochs),
           '--head-patience',str(args.head_patience),
           '--head-batch-size',str(args.head_batch_size),
           '--hidden',str(args.hidden),
           '--perturb-ratio',str(args.perturb_ratio),
           '--head-arch',args.arch,'--teacher-weight',str(args.teacher_weight)]
        if args.mode=='calibration-only':cmd+=['--head-cal-only']
        logfile=output/'logs'/f'DLinear_seed{seed}_{args.mode}.log'
        print('[START]',seed,flush=True)
        with logfile.open('w') as f:
            f.write('COMMAND: '+' '.join(cmd)+'\n');f.flush()
            returncode=subprocess.run(cmd,cwd=HERE.parents[1],stdout=f,stderr=subprocess.STDOUT,
                                      env=os.environ.copy()).returncode
        if returncode!=0 or not wanted.exists():
            raise RuntimeError(f'[FAILED] seed{seed}; inspect {logfile}')
        print('[PASS]',seed,flush=True)
    if args.mode=='calibration-only':
        rows=[]
        for seed in fixed['model_seeds']:
            d=json.loads((output/'DLinear'/f'seed{seed}'/'h2_head_calibration_only.json').read_text())
            assert not d['strong_test_outcomes_accessed']
            rows.append(dict(model_seed=seed,arch=args.arch,teacher_weight=args.teacher_weight,
                matched_val_ipe=d['validation_ipe']['Matched_without_DMIS'],
                dmis_val_ipe=d['validation_ipe']['Matched+DMIS'],
                shuffled_val_ipe=d['validation_ipe']['Shuffled_DMIS'],
                improvement_in_cal_ipe=d['validation_extra_dmis_ipe']))
        df=pd.DataFrame(rows)
        df.to_csv(output/'calibration_comparison.csv',index=False)
        print(df.to_string(index=False))
        print('[Dcal ONLY]',output/'calibration_comparison.csv')
    else:
        cmd=[sys.executable,str(HERE/'analyze_steering.py'),
             '--output-root',str(output),'--selection-file',str(lock)]
        if subprocess.run(cmd,cwd=HERE.parents[1]).returncode!=0:
            raise RuntimeError('Exploratory aggregation failed; original H2 unaffected')
        paper=output/'Table21_H2_steering.csv'
        paper.rename(output/'Table21_H2_EXPLORATORY_NOT_CONFIRMATORY.csv')
        (output/'EXPLORATORY_RESULT_NOTICE.txt').write_text(
            'The original five-seed H2 strong-test outcomes were inspected before this '
            'variant was designed. This is exploratory follow-up, not a new confirmatory '
            'test. The frozen v2 Table21 must remain unchanged.\n')
        print('[EXPLORATORY ONLY]',output/'Table21_H2_EXPLORATORY_NOT_CONFIRMATORY.csv')

if __name__=='__main__':main()
