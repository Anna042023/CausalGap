
#!/usr/bin/env python3
"""
CausalGap Section 5.4
Operational Regime Stability (Synthetic + SUMO final)

Inputs
------
Synthetic:
Search recursively for:
    causalgap_results.csv

Example:
runs/synthetic/TimesNet/trackB/seed0/causalgap_results.csv

SUMO:
Directly read:
    sumo_model_response_error.csv


Outputs
-------
Table3_regime_stability_final.csv

Columns:
setting
num_samples
GMM_ARI
GMM_NMI
agreement_q0.4
agreement_q0.6

Also outputs regime assignments and distributions.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


REGIMES = [
    "Stable Invariance",
    "Adaptive Response",
    "Over-Reactive Instability",
    "Pathological Rigidity"
]


def encode(labels):
    mapper = {x:i for i,x in enumerate(REGIMES)}
    return np.array([mapper[x] for x in labels])


def load_synthetic(root):

    files = list(Path(root).rglob("causalgap_results.csv"))

    if len(files) == 0:
        raise FileNotFoundError(
            "No causalgap_results.csv found"
        )

    dfs=[]

    for f in files:
        df=pd.read_csv(f)

        # keep only valid Track B samples
        if "evaluation_track" in df.columns:
            df=df[
                df["evaluation_track"]
                .astype(str)
                .str.upper()=="B"
            ]

        if "leakage_detected" in df.columns:
            df=df[
                ~df["leakage_detected"]
                .astype(str)
                .str.lower()
                .isin(["true","1","yes"])
            ]

        dfs.append(df)


    out=pd.concat(
        dfs,
        ignore_index=True
    )

    return out



def load_sumo(path):

    df=pd.read_csv(path)

    required=[
        "model",
        "DMIS",
        "RD"
    ]

    for c in required:
        if c not in df.columns:
            raise KeyError(
                f"SUMO missing column {c}"
            )

    return df



def get_dmis_synthetic(df):

    if "DMIS" in df.columns:
        return df["DMIS"].values

    if (
        "representation_response_raw" in df.columns
        and
        "environment_distance" in df.columns
    ):
        return (
            df["representation_response_raw"].values /
            np.maximum(
                df["environment_distance"].values,
                1e-6
            )
        )

    raise ValueError(
        "Cannot calculate Synthetic DMIS"
    )


def assign_regime(dmis, rd, q):

    d_threshold=np.quantile(dmis,q)
    r_threshold=np.quantile(rd,q)

    labels=[]

    for d,r in zip(dmis,rd):

        if d < d_threshold and r < r_threshold:
            labels.append(
                "Stable Invariance"
            )

        elif d >= d_threshold and r < r_threshold:
            labels.append(
                "Adaptive Response"
            )

        elif d >= d_threshold and r >= r_threshold:
            labels.append(
                "Over-Reactive Instability"
            )

        else:
            labels.append(
                "Pathological Rigidity"
            )

    return np.array(labels)



def evaluate(df, setting, output):

    if setting=="Controlled Synthetic":
        dmis=get_dmis_synthetic(df)
    else:
        dmis=df["DMIS"].values


    rd=df["RD"].values


    primary=assign_regime(
        dmis,
        rd,
        0.5
    )


    df=df.copy()
    df["regime_q050"]=primary


    result={
        "setting":setting,
        "num_samples":len(df)
    }


    for q in [0.4,0.6]:

        r=assign_regime(
            dmis,
            rd,
            q
        )

        result[
            f"agreement_q{q}"
        ]=float(
            np.mean(r==primary)
        )


    X=np.stack(
        [
            dmis,
            rd
        ],
        axis=1
    )


    gmm=GaussianMixture(
        n_components=4,
        random_state=0
    )

    gmm.fit(X)

    pred=gmm.predict(X)


    result["GMM_ARI"]=(
        adjusted_rand_score(
            encode(primary),
            pred
        )
    )

    result["GMM_NMI"]=(
        normalized_mutual_info_score(
            encode(primary),
            pred
        )
    )


    df.to_csv(
        output /
        f"{setting.replace(' ','_')}_regime_assignments.csv",
        index=False
    )


    dist=(
        df["regime_q050"]
        .value_counts(normalize=True)
        .reset_index()
    )

    dist.columns=[
        "regime",
        "ratio"
    ]

    dist["setting"]=setting

    return result, dist



def main():

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--synthetic-root",
        required=True
    )

    parser.add_argument(
        "--sumo-csv",
        required=True
    )

    parser.add_argument(
        "--output",
        default="regime_stability_results"
    )

    args=parser.parse_args()


    output=Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True
    )


    synthetic=load_synthetic(
        args.synthetic_root
    )

    sumo=load_sumo(
        args.sumo_csv
    )


    results=[]
    distributions=[]


    r,d=evaluate(
        synthetic,
        "Controlled Synthetic",
        output
    )

    results.append(r)
    distributions.append(d)



    r,d=evaluate(
        sumo,
        "Abstract SUMO",
        output
    )

    results.append(r)
    distributions.append(d)



    table=pd.DataFrame(results)

    table=table[
        [
            "setting",
            "num_samples",
            "GMM_ARI",
            "GMM_NMI",
            "agreement_q0.4",
            "agreement_q0.6"
        ]
    ]

    table.to_csv(
        output/
        "Table3_regime_stability_final.csv",
        index=False
    )


    pd.concat(
        distributions,
        ignore_index=True
    ).to_csv(
        output/
        "regime_distribution.csv",
        index=False
    )


    print(table)


if __name__=="__main__":
    main()
