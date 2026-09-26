#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from protocol import FOUNDATION_MODELS, resolve_foundation_parameter_count


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root', type=Path, default=Path('/data/wanganna/CausalGap'))
    ap.add_argument('--output', type=Path, default=None)
    args=ap.parse_args()
    rows=[]
    for m in sorted(FOUNDATION_MODELS):
        n, src = resolve_foundation_parameter_count(m, args.project_root)
        rows.append({'model':m,'parameter_count':n,'source':src,'resolved':n is not None})
    df=pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output,index=False)
    if not df['resolved'].all():
        raise SystemExit(2)

if __name__=='__main__':
    main()
