from .stgcn import STGCN
from .tslib import TSLibModel
from .dsai import DSAIWrapper
from .adapters import (
    DCRNNAdapter, ASTGNNAdapter, GWNetAdapter, AGCRNAdapter, STEVEAdapter,
    DCGCNAdapter, STNormAdapter, CCHMMAdapter, CausalGRITAdapter, TFMVGNNAdapter
)
from .adapters.paper_graphs import build_causalgrit_prior, build_tfmvgnn_te_graph
from .foundation import FoundationSubprocessModel
from .registry import get_spec
from ..config import canonical_model, FOUNDATION_MODELS


def build_model(name,bundle,args):
    name=canonical_model(name)
    if name=='STGCN':
        base=STGCN(bundle.num_nodes,bundle.history_len,bundle.pred_len,ks=args.ks,kt=args.kt,dropout=args.dropout)
    elif name=='DCRNN':
        base=DCRNNAdapter(bundle.num_nodes,bundle.history_len,bundle.pred_len,hidden_dim=args.d_model,num_layers=max(1,args.e_layers))
    elif name=='ASTGNN':
        base=ASTGNNAdapter(bundle.num_nodes,bundle.history_len,bundle.pred_len,d_model=args.d_model,n_heads=args.n_heads,e_layers=args.e_layers,dropout=args.dropout)
    elif name=='GWNet':
        base=GWNetAdapter(bundle.num_nodes,bundle.history_len,bundle.pred_len,dropout=args.dropout)
    elif name=='AGCRN':
        base=AGCRNAdapter(bundle.num_nodes,bundle.history_len,bundle.pred_len,hidden_dim=args.d_model,num_layers=max(1,args.e_layers))
    elif name=='STEVE':
        base=STEVEAdapter(bundle.num_nodes,bundle.history_len,bundle.pred_len,d_model=args.d_model,dropout=args.dropout)
    elif name=='DCGCN':
        base=DCGCNAdapter(bundle.num_nodes,bundle.history_len,bundle.pred_len,d_model=args.d_model,e_layers=args.e_layers,dropout=args.dropout)
    elif name=='ST-Norm':
        base=STNormAdapter(bundle.num_nodes,bundle.history_len,bundle.pred_len,channels=max(16,args.d_model//2),dropout=args.dropout)
    elif name=='CCHMM':
        base=CCHMMAdapter(bundle.num_nodes,bundle.history_len,bundle.pred_len,d_model=args.d_model,dropout=args.dropout)
    elif name=='CausalGRIT':
        prior, cache_path, prep_meta = build_causalgrit_prior(
            bundle.name, bundle.train_x, bundle.adj, bundle.scaler,
            rf_trees=getattr(args,'causalgrit_rf_trees',32),
            rf_max_samples=getattr(args,'causalgrit_rf_max_samples',4096),
            rf_eval_samples=getattr(args,'causalgrit_rf_eval_samples',256),
            k_hop=getattr(args,'causalgrit_k_hop',2),
            max_adjustment_nodes=getattr(args,'causalgrit_max_adjustment_nodes',32),
            force_rebuild=getattr(args,'rebuild_paper_graph_cache',False),
            seed=getattr(args,'paper_graph_seed',2026), verbose=True,
        )
        base=CausalGRITAdapter(
            bundle.num_nodes,bundle.history_len,bundle.pred_len,prior,bundle.adj,
            d_model=args.d_model,flow_steps=getattr(args,'causalgrit_flow_steps',4),
            refine_rank=getattr(args,'causalgrit_refine_rank',16),dropout=args.dropout,
            egc_eps=getattr(args,'causalgrit_egc_eps',0.10),
            kl_beta_max=getattr(args,'causalgrit_kl_beta_max',1e-3),
            kl_steepness=getattr(args,'causalgrit_kl_steepness',10.0),
            lambda_main=getattr(args,'causalgrit_lambda_main',1.0),
            lambda_consistency=getattr(args,'causalgrit_lambda_consistency',0.1),
            graph_reg_weight=getattr(args,'causalgrit_graph_reg_weight',1e-4),
        )
        base.paper_graph_cache_path=str(cache_path); base.paper_graph_preprocess_meta=prep_meta
    elif name=='TF-MVGNN':
        te, cache_path, prep_meta = build_tfmvgnn_te_graph(
            bundle.name,bundle.train_x,bundle.adj,
            te_bins=getattr(args,'tfmvgnn_te_bins',8),
            te_threshold=getattr(args,'tfmvgnn_te_threshold',0.6),
            te_max_samples=getattr(args,'tfmvgnn_te_max_samples',8192),
            force_rebuild=getattr(args,'rebuild_paper_graph_cache',False), verbose=True,
        )
        base=TFMVGNNAdapter(
            bundle.num_nodes,bundle.history_len,bundle.pred_len,te,
            hidden_dim=args.d_model,k=getattr(args,'tfmvgnn_k',3),dropout=args.dropout,
        )
        base.paper_graph_cache_path=str(cache_path); base.paper_graph_preprocess_meta=prep_meta
    elif name in ('PatchTST','iTransformer','TimesNet','FEDformer','DLinear','Autoformer'):
        base=TSLibModel(name,bundle.num_nodes,bundle.history_len,bundle.pred_len,d_model=args.d_model,d_ff=args.d_ff,e_layers=args.e_layers,d_layers=args.d_layers,n_heads=args.n_heads,dropout=args.dropout,moving_avg=args.moving_avg,factor=args.factor,patch_len=getattr(args,'patch_len',None),stride=getattr(args,'stride',None))
    elif name in FOUNDATION_MODELS:
        base=FoundationSubprocessModel(
            name,bundle.num_nodes,bundle.history_len,bundle.pred_len,
            interval_minutes=bundle.interval_minutes,
            num_samples=getattr(args,'foundation_num_samples',100),
            scaler_mean=bundle.scaler.mean,scaler_std=bundle.scaler.std,
            worker_startup_timeout=getattr(args,'foundation_worker_startup_timeout',180),
            worker_infer_timeout=getattr(args,'foundation_worker_infer_timeout',600),
            adaptive_split=bool(getattr(args,'foundation_adaptive_split',True)),
            max_split_depth=getattr(args,'foundation_max_split_depth',8),
            single_window_retries=getattr(args,'foundation_single_window_retries',1),
            cache_enabled=bool(getattr(args,'foundation_cache',False)),
            cache_dir=getattr(args,'foundation_cache_dir',None),
            cache_readonly=bool(getattr(args,'foundation_cache_readonly',False)),
        )
    else: raise ValueError(name)

    spec=get_spec(name); base.native_uses_graph=bool(spec.native_uses_external_graph)
    track=str(getattr(args,'track','A')).upper()
    if track=='A':
        base.evaluation_track='A'; return base
    if track=='B':
        return DSAIWrapper(base,bundle.adj,coords=bundle.coords,coordinate_source=bundle.coordinate_source,dynamic_mix=getattr(args,'dsai_dynamic_mix',0.05))
    raise ValueError(f'Unknown track {track}')
