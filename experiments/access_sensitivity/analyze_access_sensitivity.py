#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CausalGap Section 5.8 close-out: DSAI + observational stress tests (V3).

V3 includes all V2 fixes and additionally fixes the Track A/B regime-distribution normalization bug by normalizing the numeric count Series per track.

V2 fixes the first version by:
  * using a canonical frozen Track-B cohort rather than recursively pooling historical runs;
  * reading the newly generated isolated Synthetic Track-A cohort;
  * converting NaN/Inf to JSON null instead of crashing;
  * refusing many-to-many A/B matching;
  * keeping observational topology-input stress separate from intervention-consistent regimes;
  * keeping I-405 RD-only.

Plots intentionally leave Matplotlib's font family at its default. Only font sizes are enlarged.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OBS_DATASETS=["NYCTaxi","NYCBike1","NYCBike2","BJTaxi","PEMS-BAY","PEMS04","PEMS08"]
STRUCTURAL_TYPES={"edge_removal","weight_modification","node_isolation"}
REGIME_ORDER=["Stable Invariance","Adaptive Response","Over-Reactive Instability","Pathological Rigidity"]
MODELS=["STGCN","DCRNN","ASTGNN","GWNet","AGCRN","STEVE","DCGCN","ST-Norm","CCHMM","CausalGRIT","TF-MVGNN","PatchTST","iTransformer","TimesNet","FEDformer","DLinear","Autoformer","Chronos","TimesFM","Moirai","Lag-Llama"]
FAMILY_MEMBERS=OrderedDict({
    "Family I":["STGCN","DCRNN","ASTGNN","GWNet","AGCRN"],
    "Family II":["STEVE","DCGCN","ST-Norm"],
    "Family III":["CausalGRIT","CCHMM","TF-MVGNN"],
    "Family IV":["PatchTST","iTransformer","TimesNet","FEDformer","DLinear","Autoformer"],
    "Family V":["Chronos","TimesFM","Moirai","Lag-Llama"],
})
FAMILY_ORDER=list(FAMILY_MEMBERS)
MODEL2FAMILY={m:f for f,ms in FAMILY_MEMBERS.items() for m in ms}
GRAPH_NATIVE={"STGCN","DCRNN","ASTGNN","GWNet","AGCRN","STEVE","DCGCN","ST-Norm","CCHMM","CausalGRIT","TF-MVGNN"}
GRAPH_FREE=set(MODELS)-GRAPH_NATIVE

COLORS={"A":"#4E79A7","B":"#2A9D8F","stable":"#8AB17D","adaptive":"#4E79A7","over":"#E9C46A","rigid":"#E76F51","line":"#6C7A89","grid":"#D9DEE5","text":"#28364D"}
REGIME_COLORS={"Stable Invariance":COLORS["stable"],"Adaptive Response":COLORS["adaptive"],"Over-Reactive Instability":COLORS["over"],"Pathological Rigidity":COLORS["rigid"]}
plt.rcParams.update({"font.size":14,"axes.titlesize":17,"axes.labelsize":15,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":12,"figure.titlesize":18})


def log(s): print(s,flush=True)
def normtok(x): return re.sub(r"[^a-z0-9]+","",str(x).strip().lower())
def normalize_model(x):
    s=str(x).strip(); n=normtok(s)
    return {"graphwavenet":"GWNet","gwnet":"GWNet","causalgrt":"CausalGRIT","causalgrit":"CausalGRIT","stnorm":"ST-Norm","lagllama":"Lag-Llama","itransformer":"iTransformer","tfmvgnn":"TF-MVGNN","timesfm":"TimesFM"}.get(n,s)
def normalize_dataset(x):
    s=normtok(x)
    return {"nyctaxi":"NYCTaxi","nycbike1":"NYCBike1","nycbike2":"NYCBike2","bjtaxi":"BJTaxi","pemsbay":"PEMS-BAY","pems04":"PEMS04","pems4":"PEMS04","pems08":"PEMS08","pems8":"PEMS08","metrla":"METR-LA","synthetic":"synthetic","synthetich1":"synthetic"}.get(s,str(x))
def family_of(m): return MODEL2FAMILY.get(normalize_model(m),"Unmapped")
def pick_col(df,cands,required=False):
    for c in cands:
        if c in df.columns:return c
    mp={normtok(c):c for c in df.columns}
    for c in cands:
        if normtok(c) in mp:return mp[normtok(c)]
    if required: raise KeyError(f"Missing column; tried {cands}; existing={list(df.columns)}")
    return None
def num(s): return pd.to_numeric(s,errors="coerce")
def true_series(s): return s.astype(str).str.strip().str.lower().isin(["true","1","yes","y"])

def json_clean(x):
    if isinstance(x,dict): return {str(k):json_clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [json_clean(v) for v in x]
    if isinstance(x,np.ndarray): return [json_clean(v) for v in x.tolist()]
    if isinstance(x,(np.floating,float)):
        v=float(x); return v if math.isfinite(v) else None
    if isinstance(x,(np.integer,int)): return int(x)
    if isinstance(x,(np.bool_,bool)): return bool(x)
    return x

def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(json_clean(obj),indent=2,ensure_ascii=False,default=str,allow_nan=False),encoding="utf-8")
def save_table(df,stem,caption,label):
    stem.parent.mkdir(parents=True,exist_ok=True); df.to_csv(stem.with_suffix(".csv"),index=False)
    stem.with_suffix(".tex").write_text(df.to_latex(index=False,escape=False,caption=caption,label=label,position="!htbp"),encoding="utf-8")
def save_fig(fig,stem):
    stem.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"),dpi=350,bbox_inches="tight")
    plt.close(fig)
def style_ax(ax,grid=True):
    ax.set_axisbelow(True)
    if grid: ax.grid(True,color=COLORS["grid"],linewidth=.8,alpha=.65)
    for sp in ax.spines.values(): sp.set_color("#AEB8C2"); sp.set_linewidth(.9)
    ax.tick_params(colors=COLORS["text"]); ax.xaxis.label.set_color(COLORS["text"]); ax.yaxis.label.set_color(COLORS["text"]); ax.title.set_color(COLORS["text"])


def canonicalize(df,track,source_name,model_seed=0,eval_seeds=(0,1,2,3,4)):
    d=df.copy()
    mcol=pick_col(d,["model"],True); d["model"]=d[mcol].map(normalize_model)
    tcol=pick_col(d,["evaluation_track","track"])
    if tcol is not None:
        d["track"]=d[tcol].astype(str).str.upper().str.replace("TRACK","",regex=False).str.strip()
    else:d["track"]=track
    d=d[d.track.eq(track)].copy()
    scol=pick_col(d,["model_seed"])
    if scol is not None:d=d[num(d[scol]).eq(model_seed)].copy()
    it=pick_col(d,["intervention_type"],True); lv=pick_col(d,["paper_intensity_level","level"],True); es=pick_col(d,["intervention_seed","eval_seed"],True)
    d["intervention_type"]=d[it].astype(str); d["paper_intensity_level"]=d[lv].astype(str); d["intervention_seed"]=num(d[es]).astype("Int64")
    d=d[d.intervention_type.isin(STRUCTURAL_TYPES)&d.intervention_seed.isin(list(eval_seeds))].copy()
    if "causal_valid" in d:d=d[true_series(d.causal_valid)].copy()
    if "leakage_detected" in d:d["_leak"]=true_series(d.leakage_detected);d=d[~d._leak].copy()
    dc=pick_col(d,["DMIS","dmis","DMIS_cal","DMIScal"])
    if dc is None:
        rr=pick_col(d,["representation_response_raw"],True); ed=pick_col(d,["environment_distance"],True); d["DMIS_used"]=num(d[rr])/num(d[ed]).replace(0,np.nan)
    else:d["DMIS_used"]=num(d[dc])
    rc=pick_col(d,["RD","rd"]); d["RD_used"]=num(d[rc]) if rc else np.nan
    b=pick_col(d,["IPE0","baseline_ipe","baseline_mae","MAE0"]); k=pick_col(d,["IPEk","perturbed_ipe","perturbed_mae","MAEk"])
    d["Baseline_IPE"]=num(d[b]) if b else np.nan; d["Perturbed_IPE"]=num(d[k]) if k else np.nan
    d["family"]=d.model.map(family_of); d["source_name"]=source_name
    key=["model","intervention_type","paper_intensity_level","intervention_seed"]
    dup=d.duplicated(key,keep=False)
    if dup.any():
        sample=d.loc[dup,key+["DMIS_used","RD_used"]].head(20).to_dict("records")
        raise RuntimeError(f"{source_name}: canonical cohort still has duplicate keys; refusing silent deduplication. sample={sample}")
    return d


def load_synthetic_trackb(h1_root:Path,eval_seeds):
    snap=h1_root/"analysis/h1/intermediate/01_h1_raw_calibrated_rows.csv"
    if snap.exists():
        d=pd.read_csv(snap,low_memory=False)
        if "setting" in d.columns:d=d[d.setting.astype(str).str.lower().eq("synthetic")].copy()
        return canonicalize(d,"B",f"frozen H1 snapshot: {snap}",0,eval_seeds),str(snap)
    pieces=[]
    for m in MODELS:
        p=h1_root/"runs/synthetic"/m/"trackB/seed0/causalgap_results.csv"
        if not p.exists(): raise FileNotFoundError(f"Frozen Track-B file missing: {p}")
        pieces.append(pd.read_csv(p,low_memory=False))
    return canonicalize(pd.concat(pieces,ignore_index=True,sort=False),"B","canonical Track-B seed0 raw runs",0,eval_seeds),"seed0 raw runs"

def load_synthetic_tracka(root:Path,eval_seeds):
    pieces=[]
    for m in MODELS:
        p=root/"runs/synthetic"/m/"trackA/seed0/causalgap_results.csv"
        if not p.exists(): continue
        pieces.append(pd.read_csv(p,low_memory=False))
    if not pieces:return pd.DataFrame(),"missing"
    return canonicalize(pd.concat(pieces,ignore_index=True,sort=False),"A","Section5.8 Track-A seed0",0,eval_seeds),str(root)

def obs_path_candidates(root,ds,m,track):
    return [root/ds/m/f"track{track}"/"seed0/causalgap_results.csv"]
def load_observational(root_b:Path,root_a:Path):
    pieces=[]; missing=[]
    for ds in OBS_DATASETS:
        for m in MODELS:
            for tr,root in [("B",root_b),("A",root_a)]:
                p=root/ds/m/f"track{tr}"/"seed0/causalgap_results.csv"
                if not p.exists(): missing.append((ds,m,tr,str(p))); continue
                x=pd.read_csv(p,low_memory=False); x["dataset"]=ds
                # observational primary_fast uses intervention seed 0; retain whatever is actually present
                ev=sorted(num(x[pick_col(x,["intervention_seed"],True)]).dropna().astype(int).unique().tolist())
                y=canonicalize(x,tr,f"{ds}/{m}/Track{tr}",0,ev)
                y["dataset"]=ds; pieces.append(y)
    return (pd.concat(pieces,ignore_index=True,sort=False) if pieces else pd.DataFrame()),missing


def classify(dmis,rd,dt,rt):
    hd=dmis>=dt; hr=rd>=rt; out=np.empty(len(dmis),dtype=object)
    out[(~hd)&(~hr)]="Stable Invariance";out[hd&(~hr)]="Adaptive Response";out[hd&hr]="Over-Reactive Instability";out[(~hd)&hr]="Pathological Rigidity";return out

def recover_regime(regime_csv:Path):
    r=pd.read_csv(regime_csv,low_memory=False)
    if "evaluation_track" in r:
        t=r.evaluation_track.astype(str).str.upper().str.replace("TRACK","",regex=False).str.strip();
        if (t=="B").any():r=r[t=="B"].copy()
    lc=pick_col(r,["regime_q050"],True); dc=pick_col(r,["DMIS","DMIS_cal","DMIScal"],True); rc=pick_col(r,["RD"],True); tc=pick_col(r,["intervention_type"],True)
    r["_d"]=num(r[dc]);r["_r"]=num(r[rc]);r=r[np.isfinite(r._d)&np.isfinite(r._r)].copy()
    d0=float(r._d.median());r0=float(r._r.median()); pred=classify(r._d.to_numpy(),r._r.to_numpy(),np.full(len(r),d0),np.full(len(r),r0)); ag=float(np.mean(pred==r[lc].astype(str).to_numpy()))
    if ag>=1-1e-12:return "global_q050",{"__global__":(d0,r0)},{"saved_label_reproduction":ag,"source":str(regime_csv),"rows":len(r),"thresholds":{"__global__":[d0,r0]}}
    th={}; ps=pd.Series(index=r.index,dtype=object)
    for typ,g in r.groupby(tc,sort=False):
        th[str(typ)]=(float(g._d.median()),float(g._r.median())); dt,rt=th[str(typ)];ps.loc[g.index]=classify(g._d.to_numpy(),g._r.to_numpy(),np.full(len(g),dt),np.full(len(g),rt))
    ag2=float(np.mean(ps.astype(str).to_numpy()==r[lc].astype(str).to_numpy()))
    if ag2<1-1e-12:raise RuntimeError(f"Cannot reproduce frozen q=.50 regime labels: global={ag:.6f}, typewise={ag2:.6f}")
    return "type_q050",th,{"saved_label_reproduction":ag2,"source":str(regime_csv),"rows":len(r),"thresholds":th}
def threshold_arrays(df,mode,th):
    if mode=="global_q050":
        d,r=th["__global__"];return np.full(len(df),d),np.full(len(df),r)
    ds=np.array([th[str(t)][0] for t in df.intervention_type]);rs=np.array([th[str(t)][1] for t in df.intervention_type]);return ds,rs


def analyze_synthetic(a,b,regime_csv,out,strict):
    inter=out/"intermediate";tables=out/"tables";figs=out/"figures";diag=out/"diagnostics"
    for p in [inter,tables,figs,diag]:p.mkdir(parents=True,exist_ok=True)
    if a.empty:
        audit={"status":"missing_track_A","trackA_rows":0,"trackB_rows":len(b)};save_json(diag/"synthetic_dsai_audit.json",audit)
        if strict:raise RuntimeError("Synthetic Track A is missing. Run run_synthetic_track_a.py first.")
        log("[SYNTHETIC] Track A missing; analysis remains partial.");return audit
    key=["model","intervention_type","paper_intensity_level","intervention_seed"]
    aa=set(map(tuple,a[key].astype(str).to_numpy()));bb=set(map(tuple,b[key].astype(str).to_numpy()));missing_ab=sorted(bb-aa);extra_ab=sorted(aa-bb)
    if missing_ab or extra_ab:
        msg=f"A/B key mismatch: missing_in_A={len(missing_ab)}, extra_in_A={len(extra_ab)}"
        if strict:raise RuntimeError(msg)
        log("[SYNTHETIC] WARNING: "+msg)
    both=pd.concat([a,b],ignore_index=True);both.to_csv(inter/"01_synthetic_canonical_A_B_rows.csv",index=False)
    cfg=(both.groupby(["model","track","intervention_type","paper_intensity_level"],as_index=False).agg(DMIS=("DMIS_used","mean"),RD=("RD_used","mean"),Seeds=("intervention_seed","nunique")))
    mt=(cfg.groupby(["model","track"],as_index=False).agg(mean_DMIS=("DMIS","mean"),median_DMIS=("DMIS","median"),mean_RD=("RD","mean"),n_configs=("DMIS","size")))
    mt["rank_DMIS"]=mt.groupby("track").mean_DMIS.rank(ascending=False,method="average");mt["family"]=mt.model.map(family_of);mt.to_csv(inter/"02_synthetic_model_track_summary.csv",index=False)
    pa=mt[mt.track.eq("A")][["model","mean_DMIS","mean_RD","rank_DMIS"]].rename(columns={"mean_DMIS":"mean_DMIS_A","mean_RD":"mean_RD_A","rank_DMIS":"rank_DMIS_A"})
    pb=mt[mt.track.eq("B")][["model","mean_DMIS","mean_RD","rank_DMIS"]].rename(columns={"mean_DMIS":"mean_DMIS_B","mean_RD":"mean_RD_B","rank_DMIS":"rank_DMIS_B"})
    pm=pa.merge(pb,on="model",validate="one_to_one");pm["family"]=pm.model.map(family_of);pm.to_csv(inter/"03_synthetic_trackA_trackB_models.csv",index=False)
    kt=stats.kendalltau(pm.mean_DMIS_A,pm.mean_DMIS_B,variant="b",nan_policy="omit") if len(pm)>=2 else None
    sp=stats.spearmanr(pm.mean_DMIS_A,pm.mean_DMIS_B,nan_policy="omit") if len(pm)>=2 else None
    tau=float(kt.statistic) if kt is not None else np.nan;rho=float(sp.statistic) if sp is not None else np.nan

    mode,th,raudit=recover_regime(regime_csv);save_json(diag/"regime_protocol_audit.json",raudit)
    good=both[np.isfinite(both.DMIS_used)&np.isfinite(both.RD_used)].copy();dt,rt=threshold_arrays(good,mode,th);good["regime"]=classify(good.DMIS_used.to_numpy(),good.RD_used.to_numpy(),dt,rt);good.to_csv(inter/"04_synthetic_rows_with_regime.csv",index=False)
    dist = good.groupby(["track", "regime"]).size().rename("count").reset_index()
    full = pd.MultiIndex.from_product([["A", "B"], REGIME_ORDER], names=["track", "regime"]).to_frame(index=False)
    dist = full.merge(dist, on=["track", "regime"], how="left")
    dist["count"] = pd.to_numeric(dist["count"], errors="coerce").fillna(0).astype(int)
    # IMPORTANT: normalize the numeric `count` Series within each track.
    # V2 accidentally used `dist.groupby("track").count.transform(...)`, where
    # `.count` is a DataFrameGroupBy method rather than a Series and therefore
    # raised AttributeError after all Track-A jobs had successfully finished.
    denom = dist.groupby("track")["count"].transform("sum")
    dist["fraction"] = np.divide(
        dist["count"].to_numpy(dtype=float),
        denom.to_numpy(dtype=float),
        out=np.zeros(len(dist), dtype=float),
        where=denom.to_numpy(dtype=float) > 0,
    )
    dist.to_csv(inter/"05_synthetic_regime_distribution.csv", index=False)
    ar=good[good.track.eq("A")][key+["regime"]].rename(columns={"regime":"regime_A"});br=good[good.track.eq("B")][key+["regime"]].rename(columns={"regime":"regime_B"});pr=ar.merge(br,on=key,validate="one_to_one");agreement=float(np.mean(pr.regime_A==pr.regime_B)) if len(pr) else np.nan;pr.to_csv(inter/"06_synthetic_paired_regimes.csv",index=False);conf=pd.crosstab(pr.regime_A,pr.regime_B).reindex(index=REGIME_ORDER,columns=REGIME_ORDER,fill_value=0);conf.to_csv(inter/"07_synthetic_regime_confusion.csv")

    def diststr(tr):
        g=dist[dist.track.eq(tr)].set_index("regime");short={"Stable Invariance":"Stable","Adaptive Response":"Adaptive","Over-Reactive Instability":"Over","Pathological Rigidity":"Rigid"};return "; ".join(f"{short[r]} {100*float(g.loc[r,'fraction']):.1f}\\%" for r in REGIME_ORDER)
    table=pd.DataFrame([
        {"Comparison":"Model DMIS Ranking","Track A":f"Mean DMIS={mt.loc[mt.track.eq('A'),'mean_DMIS'].mean():.6f}","Track B":f"Mean DMIS={mt.loc[mt.track.eq('B'),'mean_DMIS'].mean():.6f}","Cross-Track Summary":f"Kendall $\\tau_b$={tau:.3f}"},
        {"Comparison":"Operational-Regime Distribution","Track A":diststr("A"),"Track B":diststr("B"),"Cross-Track Summary":f"Agreement={100*agreement:.1f}\\%"},
    ]);save_table(table,tables/"Table22_DSAI_TrackA_TrackB_audit","DSAI Track A/B audit in controlled synthetic dynamics. Tracks are reported separately and are never pooled.","tab:dsai_track_ab")

    # Figure A: A/B DMIS correspondence
    fig,ax=plt.subplots(figsize=(7.2,6.2));ax.scatter(pm.mean_DMIS_A,pm.mean_DMIS_B,s=68,color=COLORS["B"],alpha=.9,edgecolor="white",linewidth=.8,zorder=3);vals=np.r_[pm.mean_DMIS_A,pm.mean_DMIS_B];lo,hi=np.nanmin(vals),np.nanmax(vals);pad=.06*(hi-lo if hi>lo else 1);ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"--",color=COLORS["line"],lw=1.5);ax.set_xlabel("Track A Mean DMIS");ax.set_ylabel("Track B Mean DMIS");ax.set_title("DSAI Cross-Track Model Response",fontweight="bold");style_ax(ax);save_fig(fig,figs/"Fig58_DSAI_model_DMIS_correspondence")
    # Figure B: rank shift
    pm2=pm.sort_values("rank_DMIS_A");y=np.arange(len(pm2));fig,ax=plt.subplots(figsize=(8.8,max(6.0,.32*len(pm2)+1.8)));[ax.plot([r.rank_DMIS_A,r.rank_DMIS_B],[i,i],color="#B8C2CC",lw=2,zorder=1) for i,r in enumerate(pm2.itertuples())];ax.scatter(pm2.rank_DMIS_A,y,s=55,label="Track A",color=COLORS["A"],zorder=3);ax.scatter(pm2.rank_DMIS_B,y,s=55,label="Track B",color=COLORS["B"],zorder=3);ax.set_yticks(y,pm2.model);ax.set_xlabel("DMIS Rank (1 = Highest Response)");ax.set_title("DSAI Model-Rank Shift Across Access Tracks",fontweight="bold");ax.legend(frameon=False,ncol=2);style_ax(ax,False);ax.grid(axis="x",color=COLORS["grid"],alpha=.65);save_fig(fig,figs/"Fig58_DSAI_rank_shift")
    # Figure C: regime distribution
    piv=dist.pivot(index="track",columns="regime",values="fraction").reindex(index=["A","B"],columns=REGIME_ORDER).fillna(0);fig,ax=plt.subplots(figsize=(8.0,5.8));bottom=np.zeros(2);x=np.arange(2)
    for rg in REGIME_ORDER:
        vv=100*piv[rg].to_numpy();ax.bar(x,vv,bottom=bottom,width=.58,label=rg,color=REGIME_COLORS[rg],edgecolor="white",lw=.7);bottom+=vv
    ax.set_xticks(x,["Track A","Track B"]);ax.set_ylabel("Fraction of Instances (%)");ax.set_ylim(0,100);ax.set_title("Operational-Regime Distribution Across DSAI Tracks",fontweight="bold");ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,-.11));style_ax(ax);save_fig(fig,figs/"Fig58_DSAI_regime_distribution")
    # Figure D: regime confusion
    arr=conf.to_numpy(float);rs=arr.sum(1,keepdims=True);norm=np.divide(arr,rs,out=np.zeros_like(arr),where=rs>0);fig,ax=plt.subplots(figsize=(7.2,6.2));im=ax.imshow(norm,vmin=0,vmax=1,cmap="Blues",aspect="auto")
    for i in range(4):
        for j in range(4):ax.text(j,i,f"{int(arr[i,j])}\n{100*norm[i,j]:.1f}%",ha="center",va="center",fontsize=11,color="white" if norm[i,j]>.55 else COLORS["text"])
    labs=["Stable","Adaptive","Over-Reactive","Rigidity"];ax.set_xticks(range(4),labs,rotation=20,ha="right");ax.set_yticks(range(4),labs);ax.set_xlabel("Track B Regime");ax.set_ylabel("Track A Regime");ax.set_title("Cross-Track Operational-Regime Correspondence",fontweight="bold");fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label="Row Fraction");style_ax(ax,False);save_fig(fig,figs/"Fig58_DSAI_regime_confusion")

    audit={"status":"complete" if len(pm)==21 and not missing_ab and not extra_ab else "partial","trackA_rows":len(a),"trackB_rows":len(b),"paired_models":len(pm),"kendall_tau_b":tau,"kendall_p":float(kt.pvalue) if kt else None,"spearman_rho":rho,"spearman_p":float(sp.pvalue) if sp else None,"regime_agreement":agreement,"paired_regime_instances":len(pr),"missing_in_A":len(missing_ab),"extra_in_A":len(extra_ab)};save_json(diag/"synthetic_dsai_audit.json",audit);log(f"[SYNTHETIC] A={len(a)} B={len(b)} paired_models={len(pm)} tau_b={tau:.4f} regime_agreement={agreement:.4f}");return audit


def analyze_obs(df,missing,out,strict):
    inter=out/"intermediate";tables=out/"tables";figs=out/"figures";diag=out/"diagnostics"
    for p in [inter,tables,figs,diag]:p.mkdir(parents=True,exist_ok=True)
    if df.empty:
        audit={"status":"missing","missing_jobs":len(missing)};save_json(diag/"observational_stress_audit.json",audit);return audit
    den=df.Baseline_IPE.abs().replace(0,np.nan);df["error_change_pct"]=100*(df.Perturbed_IPE-df.Baseline_IPE)/den;df.to_csv(inter/"10_observational_rows.csv",index=False)
    cfg=df.groupby(["dataset","model","family","track","intervention_type","paper_intensity_level"],as_index=False).agg(DMIS=("DMIS_used","mean"),Baseline_IPE=("Baseline_IPE","mean"),Perturbed_IPE=("Perturbed_IPE","mean"),Error_Change_Pct=("error_change_pct","mean"),Seeds=("intervention_seed","nunique"))
    model=cfg.groupby(["dataset","model","family","track"],as_index=False).agg(DMIS=("DMIS","mean"),Baseline_IPE=("Baseline_IPE","mean"),Perturbed_IPE=("Perturbed_IPE","mean"),Error_Change_Pct=("Error_Change_Pct","mean"),Configs=("DMIS","size"));model.to_csv(inter/"11_observational_model_summary.csv",index=False)
    ds=model.groupby(["dataset","track"],as_index=False).agg(Response_Mean=("DMIS","mean"),Response_Median=("DMIS","median"),Error_Change_Mean_Pct=("Error_Change_Pct","mean"),Baseline_IPE=("Baseline_IPE","mean"),Perturbed_IPE=("Perturbed_IPE","mean"),Models=("model","nunique"));ds.to_csv(inter/"12_observational_dataset_summary.csv",index=False)
    fam=model.groupby(["dataset","family","track"],as_index=False).agg(Response_Mean=("DMIS","mean"),Error_Change_Mean_Pct=("Error_Change_Pct","mean"),Models=("model","nunique"));fam.to_csv(inter/"13_observational_family_summary.csv",index=False)
    rows=[]
    for name in OBS_DATASETS:
        g=ds[ds.dataset.eq(name)];ra=g[g.track.eq("A")];rb=g[g.track.eq("B")]
        rows.append({"Dataset":name,"Track A Response":f"{ra.iloc[0].Response_Median:.4f}" if len(ra) else "Unavailable","Track B Response":f"{rb.iloc[0].Response_Median:.4f}" if len(rb) else "Unavailable","Track A Error Change":f"{ra.iloc[0].Error_Change_Mean_Pct:+.2f}\\%" if len(ra) else "Unavailable","Track B Error Change":f"{rb.iloc[0].Error_Change_Mean_Pct:+.2f}\\%" if len(rb) else "Unavailable","Status":"Complete" if len(ra) and len(rb) and int(ra.iloc[0].Models)>=21 and int(rb.iloc[0].Models)>=21 else "Partial"})
    save_table(pd.DataFrame(rows),tables/"Table23_observational_topology_input_stress","Topology-input stress tests on observational datasets. These are not intervention-consistent RD/regime claims.","tab:observational_stress")
    save_table(fam,tables/"TableS_observational_family_stress","Family-level topology-input stress summaries.","tab:observational_family_stress")

    # Figure E/F dataset bars
    for metric,title,ylabel,stem in [("Response_Median","Topology-Input Representation Response","Median DMIS","Fig58_observational_response_by_dataset"),("Error_Change_Mean_Pct","Topology-Input Prediction Sensitivity","Mean Error Change (%)","Fig58_observational_error_change_by_dataset")]:
        piv=ds.pivot(index="dataset",columns="track",values=metric).reindex(OBS_DATASETS);x=np.arange(len(piv));w=.36;fig,ax=plt.subplots(figsize=(11.2,5.8));ax.bar(x-w/2,piv.get("A",pd.Series(index=piv.index,dtype=float)),w,label="Track A",color=COLORS["A"],alpha=.92);ax.bar(x+w/2,piv.get("B",pd.Series(index=piv.index,dtype=float)),w,label="Track B",color=COLORS["B"],alpha=.92);ax.axhline(0,color=COLORS["line"],lw=1);ax.set_xticks(x,OBS_DATASETS,rotation=22,ha="right");ax.set_ylabel(ylabel);ax.set_title(title,fontweight="bold");ax.legend(frameon=False,ncol=2);style_ax(ax);save_fig(fig,figs/stem)
    # Figure G/H family heatmaps for Track B and A difference
    for tr in ["A","B"]:
        for metric,title,stem,cmap in [("Response_Mean",f"Track {tr} Response by Model Family",f"Fig58_observational_family_response_track{tr}","YlGnBu"),("Error_Change_Mean_Pct",f"Track {tr} Error Sensitivity by Model Family",f"Fig58_observational_family_error_track{tr}","RdYlBu_r")]:
            p=fam[fam.track.eq(tr)].pivot(index="family",columns="dataset",values=metric).reindex(index=FAMILY_ORDER,columns=OBS_DATASETS);fig,ax=plt.subplots(figsize=(11.2,5.6));im=ax.imshow(p.to_numpy(float),aspect="auto",cmap=cmap);ax.set_xticks(range(len(OBS_DATASETS)),OBS_DATASETS,rotation=22,ha="right");ax.set_yticks(range(len(FAMILY_ORDER)),FAMILY_ORDER);ax.set_title(title,fontweight="bold");fig.colorbar(im,ax=ax,fraction=.035,pad=.03);style_ax(ax,False);save_fig(fig,figs/stem)
    # Figure I: A-vs-B response scatter for observational model-dataset pairs
    aa=model[model.track.eq("A")][["dataset","model","DMIS"]].rename(columns={"DMIS":"DMIS_A"});bb=model[model.track.eq("B")][["dataset","model","DMIS"]].rename(columns={"DMIS":"DMIS_B"});pp=aa.merge(bb,on=["dataset","model"],validate="one_to_one")
    if len(pp):
        fig,ax=plt.subplots(figsize=(7.2,6.2));ax.scatter(pp.DMIS_A,pp.DMIS_B,s=38,color=COLORS["B"],alpha=.72,edgecolor="white",linewidth=.4);vals=np.r_[pp.DMIS_A,pp.DMIS_B];lo,hi=np.nanmin(vals),np.nanmax(vals);pad=.05*(hi-lo if hi>lo else 1);ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"--",color=COLORS["line"],lw=1.4);ax.set_xlabel("Track A DMIS");ax.set_ylabel("Track B DMIS");ax.set_title("Observational Cross-Track Response",fontweight="bold");style_ax(ax);save_fig(fig,figs/"Fig58_observational_cross_track_response")

    gf=model[(model.track.eq("A"))&model.model.isin(GRAPH_FREE)];gf_dmis=float(np.nanmax(np.abs(gf.DMIS))) if len(gf) else np.nan;gf_err=float(np.nanmax(np.abs(gf.Error_Change_Pct))) if len(gf) else np.nan
    missA=sum(1 for x in missing if x[2]=="A");missB=sum(1 for x in missing if x[2]=="B");audit={"status":"complete" if missA==0 and missB==0 else "partial","rows":len(df),"datasets":sorted(df.dataset.unique()),"models":int(df.model.nunique()),"tracks":sorted(df.track.unique()),"missing_trackA_jobs":missA,"missing_trackB_jobs":missB,"trackA_graph_free_max_abs_DMIS":gf_dmis,"trackA_graph_free_max_abs_error_change_pct":gf_err,"scope":"topology-input stress only"};save_json(diag/"observational_stress_audit.json",audit);pd.DataFrame(missing,columns=["dataset","model","track","expected_path"]).to_csv(diag/"observational_missing_jobs.csv",index=False)
    if strict and (missA or missB):raise RuntimeError(f"Observational incomplete: missing A={missA}, B={missB}")
    log(f"[OBS] rows={len(df)} tracks={audit['tracks']} missingA={missA} missingB={missB}");return audit


def find_i405(pr):
    for p in [pr/"experiment_results/access_sensitivity/i405_event.csv",pr/"experiment_results/i405/i405_event.csv",pr/"data/i405/i405_event.csv",pr/"datasets/i405/i405_event.csv"]:
        if p.exists():return p
    return None
def analyze_i405(path,tau,out,require):
    tables=out/"tables";figs=out/"figures";diag=out/"diagnostics";[p.mkdir(parents=True,exist_ok=True) for p in [tables,figs,diag]]
    template=out/"i405_event_template.csv"
    if path is None or not path.exists():
        pd.DataFrame(columns=["detector_id","group","baseline_ipe","event_ipe","RD"]).to_csv(template,index=False);audit={"status":"missing","template":str(template),"scope":"RD-only; no DMIS/regime"};save_json(diag/"i405_audit.json",audit);log(f"[I405] missing; template -> {template}");
        if require:raise FileNotFoundError(template)
        return audit
    d=pd.read_csv(path);bc=pick_col(d,["baseline_ipe","IPE0","baseline_mae"],True);ec=pick_col(d,["event_ipe","IPEk","event_mae"],True);rc=pick_col(d,["RD","rd"]);ic=pick_col(d,["detector_id","station_id","sensor_id","id"]);gc=pick_col(d,["group","detector_group","role"]);d["baseline_ipe"]=num(d[bc]);d["event_ipe"]=num(d[ec])
    if rc:d["RD"]=num(d[rc]);src=rc
    else:
        if tau is None:raise ValueError("I-405 CSV lacks RD; pass --i405-tau. No denominator is guessed.")
        d["RD"]=(d.event_ipe-d.baseline_ipe)/(d.baseline_ipe+tau);src=f"computed tau={tau}"
    d=d[np.isfinite(d.baseline_ipe)&np.isfinite(d.event_ipe)&np.isfinite(d.RD)].copy();rows=[{"Evaluation Unit":"Detector aggregate","Baseline IPE":d.baseline_ipe.mean(),"Event IPE":d.event_ipe.mean(),"RD":d.RD.mean()}]
    if gc:
        up=d[d[gc].astype(str).str.lower().str.contains("upstream",na=False)]
        if len(up):rows.append({"Evaluation Unit":"Upstream-detector sensitivity","Baseline IPE":up.baseline_ipe.mean(),"Event IPE":up.event_ipe.mean(),"RD":up.RD.mean()})
    save_table(pd.DataFrame(rows),tables/"Table24_I405_RD_only","I-405 exploratory RD-only case. No DMIS or operational regime label is assigned.","tab:i405_rd_only")
    labels=d[ic].astype(str).tolist() if ic else [str(i+1) for i in range(len(d))];o=np.argsort(d.RD.to_numpy());fig,ax=plt.subplots(figsize=(9,max(4.8,.3*len(d)+1.8)));y=np.arange(len(d));ax.barh(y,d.RD.to_numpy()[o],color=COLORS["A"],alpha=.9);ax.set_yticks(y,np.asarray(labels)[o]);ax.axvline(0,color=COLORS["line"],lw=1);ax.set_xlabel("RD");ax.set_ylabel("Detector");ax.set_title("I-405 Exploratory Detector-Level Degradation",fontweight="bold");style_ax(ax);save_fig(fig,figs/"Fig58_I405_detector_RD");audit={"status":"complete","source":str(path),"rows":len(d),"rd_source":src,"mean_baseline_ipe":d.baseline_ipe.mean(),"mean_event_ipe":d.event_ipe.mean(),"mean_rd":d.RD.mean()};save_json(diag/"i405_audit.json",audit);return audit


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--project-root",type=Path,default=Path("/data/wanganna/CausalGap"));ap.add_argument("--h1-root",type=Path,default=None);ap.add_argument("--synthetic-tracka-root",type=Path,default=None);ap.add_argument("--observational-trackb-root",type=Path,default=None);ap.add_argument("--observational-tracka-root",type=Path,default=None);ap.add_argument("--regime-csv",type=Path,default=None);ap.add_argument("--i405-csv",type=Path,default=None);ap.add_argument("--i405-tau",type=float,default=None);ap.add_argument("--output-root",type=Path,default=None);ap.add_argument("--eval-seeds",nargs="+",type=int,default=[0,1,2,3,4]);ap.add_argument("--strict",action="store_true");ap.add_argument("--require-i405",action="store_true");args=ap.parse_args()
    pr=args.project_root.resolve();h1=args.h1_root or pr/"experiment_results/incremental_value";sa=args.synthetic_tracka_root or pr/"experiment_results/access_sensitivity/synthetic_track_a";ob=args.observational_trackb_root or pr/"experiment_results/observational_diagnostics/work_runs";oa=args.observational_tracka_root or pr/"experiment_results/access_sensitivity/observational_track_a/work_runs";reg=args.regime_csv or pr/"unified/regime_stability_results/Controlled_Synthetic_regime_assignments.csv";out=args.output_root or pr/"experiment_results/access_sensitivity";out.mkdir(parents=True,exist_ok=True)
    log("================ CausalGap DSAI + Stress V2 ================");log(f"[PATH] H1 Track B: {h1}");log(f"[PATH] Synthetic Track A: {sa}");log(f"[PATH] Obs Track B: {ob}");log(f"[PATH] Obs Track A: {oa}");log(f"[PATH] Regime: {reg}");log(f"[PATH] Output: {out}")
    b,bsrc=load_synthetic_trackb(h1,args.eval_seeds);a,asrc=load_synthetic_tracka(sa,args.eval_seeds);log(f"[LOAD] Synthetic B rows={len(b)} source={bsrc}");log(f"[LOAD] Synthetic A rows={len(a)} source={asrc}")
    syn=analyze_synthetic(a,b,reg,out,args.strict);obsdf,missing=load_observational(ob,oa);obs=analyze_obs(obsdf,missing,out,args.strict);ip=args.i405_csv or find_i405(pr);i4=analyze_i405(ip,args.i405_tau,out,args.require_i405);final={"synthetic":syn,"observational":obs,"i405":i4,"paths":{"h1":str(h1),"synthetic_trackA":str(sa),"obs_trackB":str(ob),"obs_trackA":str(oa),"regime":str(reg),"i405":str(ip) if ip else None,"output":str(out)}};save_json(out/"00_final_audit.json",final)
    log("\n================ Saved Outputs ================");[log("[SAVED] "+str(p)) for p in sorted(out.rglob("*")) if p.is_file() and p.suffix.lower() in {".csv",".tex",".json",".png",".pdf",".svg"}];log("================================================")

if __name__=="__main__":main()
