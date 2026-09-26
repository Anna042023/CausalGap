"""Single source of truth for the 21 runnable CausalGap baselines."""
from dataclasses import dataclass
from pathlib import Path
from ..config import PROJECT_ROOT

@dataclass(frozen=True)
class ModelSpec:
    name: str
    class_id: int
    training_mode: str            # supervised | zero_shot
    source_path: Path
    native_uses_external_graph: bool
    adaptation: str
    representation_hook: str
    source_status: str = 'official/source code'

SPECS = {
    'STGCN': ModelSpec('STGCN',1,'supervised',PROJECT_ROOT/'class1'/'STGCN',True,
                       'direct_horizon_matched_readout','last ST block/readout pre-head'),
    'DCRNN': ModelSpec('DCRNN',1,'supervised',PROJECT_ROOT/'class1'/'DCRNN',True,
                       'faithful PyTorch backend port of official TF1 DCGRU seq2seq','final decoder recurrent hidden state'),
    'ASTGNN': ModelSpec('ASTGNN',1,'supervised',PROJECT_ROOT/'class1'/'ASTGNN',True,
                       'traffic benchmark horizon/context adapter preserving graph+temporal-attention mechanism','final graph-temporal encoder state'),
    'GWNet': ModelSpec('GWNet',1,'supervised',PROJECT_ROOT/'class1'/'GWNet',True,
                       'official architecture with Conv1d-to-Conv2d compatibility correction','end1 hidden state immediately before final end2 prediction head'),
    'AGCRN': ModelSpec('AGCRN',1,'supervised',PROJECT_ROOT/'class1'/'AGCRN',False,
                       'official adaptive-graph recurrent encoder, horizon matched','last recurrent hidden state'),
    'STEVE': ModelSpec('STEVE',2,'supervised',PROJECT_ROOT/'class2'/'STEVE',True,
                       'released STEVE prediction core port under common traffic-only input; unavailable auxiliary labels/channels are not fabricated','released invariant ST latent'),
    'DCGCN': ModelSpec('DCGCN',2,'supervised',PROJECT_ROOT/'class2'/'DCGCN',True,
                       'joint traffic-only dynamic-causal-graph adaptation','final causal graph-temporal latent'),
    'ST-Norm': ModelSpec('ST-Norm',2,'supervised',PROJECT_ROOT/'class2'/'ST-Norm',False,
                         'official ST-Norm/WaveNet traffic forecasting path','pre-output WaveNet hidden feature'),
    'CCHMM': ModelSpec('CCHMM',3,'supervised',PROJECT_ROOT/'class3'/'CCHMM',True,
                       'traffic-only adaptation; unavailable modalities explicitly omitted','causal latent recurrent state'),
    'CausalGRIT': ModelSpec('CausalGRIT',3,'supervised',Path(__file__).resolve().parent/'adapters'/'causalgrit.py',True,
                       'paper-faithful scalable reimplementation; no official source code available','final gated-belief GRU state before prediction head',
                       'paper reimplementation (SIGSPATIAL 2025; no official code)'),
    'TF-MVGNN': ModelSpec('TF-MVGNN',3,'supervised',Path(__file__).resolve().parent/'adapters'/'tfmvgnn.py',True,
                       'paper-faithful PyTorch reimplementation; TE estimator details explicitly documented where paper is underspecified','final DL-GNN+GRU hidden state before horizon head',
                       'paper reimplementation (Neural Computing and Applications 2025; no official code)'),
    'PatchTST': ModelSpec('PatchTST',4,'supervised',PROJECT_ROOT/'class4'/'Time-Series-Library',False,'native TSLib node-as-variable + short-window patch adaptation','encoder output before head'),
    'iTransformer': ModelSpec('iTransformer',4,'supervised',PROJECT_ROOT/'class4'/'Time-Series-Library',False,'native TSLib node-as-variable + model-specific formal profile','encoder output before projection'),
    'TimesNet': ModelSpec('TimesNet',4,'supervised',PROJECT_ROOT/'class4'/'Time-Series-Library',False,'native TSLib node-as-variable + model-specific formal profile','last TimesBlock hidden before projection'),
    'FEDformer': ModelSpec('FEDformer',4,'supervised',PROJECT_ROOT/'class4'/'Time-Series-Library',False,'native TSLib node-as-variable + short-window-safe decomposition/frequency settings','decoder latent before projection'),
    'DLinear': ModelSpec('DLinear',4,'supervised',PROJECT_ROOT/'class4'/'Time-Series-Library',False,'native TSLib node-as-variable + extended early-stopped training budget','seasonal/trend pre-projection features'),
    'Autoformer': ModelSpec('Autoformer',4,'supervised',PROJECT_ROOT/'class4'/'Time-Series-Library',False,'native TSLib node-as-variable + short-window-safe decomposition and corrected epoch/patience protocol','decoder latent before projection'),
    'Chronos': ModelSpec('Chronos',5,'zero_shot',PROJECT_ROOT/'class5'/'Chronos',False,'local checkpoint zero-shot','Chronos encoder embedding'),
    'TimesFM': ModelSpec('TimesFM',5,'zero_shot',PROJECT_ROOT/'class5'/'TimesFM',False,'local checkpoint zero-shot','final TimesFM transformer hidden state'),
    'Moirai': ModelSpec('Moirai',5,'zero_shot',PROJECT_ROOT/'class5'/'Moirai',False,'local checkpoint zero-shot','Moirai transformer encoder representation'),
    'Lag-Llama': ModelSpec('Lag-Llama',5,'zero_shot',PROJECT_ROOT/'class5'/'Lag-Llama',False,'local checkpoint zero-shot','final Lag-Llama transformer hidden state'),
}

def get_spec(name):
    if name not in SPECS:
        raise KeyError(name)
    return SPECS[name]

def names(): return list(SPECS)
