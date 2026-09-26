import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('CAUSALGAP_ROOT', '/data/wanganna/CausalGap')).expanduser().resolve()
DATA_ROOT = PROJECT_ROOT / 'datasets'
RUN_ROOT = Path(os.environ.get('CAUSALGAP_RUN_ROOT', str(PROJECT_ROOT / 'runs'))).expanduser().resolve()
MODEL_WEIGHT_ROOT = PROJECT_ROOT / 'model_weights'
ENV_ROOT = PROJECT_ROOT / 'envs'

STGCN_OFFICIAL = PROJECT_ROOT / 'class1' / 'STGCN'
DCRNN_OFFICIAL = PROJECT_ROOT / 'class1' / 'DCRNN'
ASTGNN_OFFICIAL = PROJECT_ROOT / 'class1' / 'ASTGNN'
GWNET_OFFICIAL = PROJECT_ROOT / 'class1' / 'GWNet'
AGCRN_OFFICIAL = PROJECT_ROOT / 'class1' / 'AGCRN'
STEVE_OFFICIAL = PROJECT_ROOT / 'class2' / 'STEVE'
DCGCN_OFFICIAL = PROJECT_ROOT / 'class2' / 'DCGCN'
STNORM_OFFICIAL = PROJECT_ROOT / 'class2' / 'ST-Norm'
CCHMM_OFFICIAL = PROJECT_ROOT / 'class3' / 'CCHMM'
TSLIB_ROOT = PROJECT_ROOT / 'class4' / 'Time-Series-Library'
CHRONOS_OFFICIAL = PROJECT_ROOT / 'class5' / 'Chronos'
TIMESFM_OFFICIAL = PROJECT_ROOT / 'class5' / 'TimesFM'
MOIRAI_OFFICIAL = PROJECT_ROOT / 'class5' / 'Moirai'
LAGLLAMA_OFFICIAL = PROJECT_ROOT / 'class5' / 'Lag-Llama'

DATASET_ALIASES = {
    'metr-la': 'METR-LA', 'metr_la': 'METR-LA',
    'pems-bay': 'PEMS-BAY', 'pems_bay': 'PEMS-BAY',
    'pems04': 'PEMS04', 'pems08': 'PEMS08',
    'nyctaxi': 'NYCTaxi', 'nycbike1': 'NYCBike1',
    'nycbike2': 'NYCBike2', 'bjtaxi': 'BJTaxi',
}
MODEL_ALIASES = {
    'stgcn':'STGCN','dcrnn':'DCRNN','astgnn':'ASTGNN','gwnet':'GWNet','graphwavenet':'GWNet','agcrn':'AGCRN',
    'steve':'STEVE','dcgcn':'DCGCN','st-norm':'ST-Norm','stnorm':'ST-Norm','cchmm':'CCHMM',
    'causalgrit':'CausalGRIT','causal-grit':'CausalGRIT','tf-mvgnn':'TF-MVGNN','tfmvgnn':'TF-MVGNN',
    'patchtst':'PatchTST','itransformer':'iTransformer','timesnet':'TimesNet','fedformer':'FEDformer','dlinear':'DLinear','autoformer':'Autoformer',
    'chronos':'Chronos','timesfm':'TimesFM','moirai':'Moirai','lag-llama':'Lag-Llama','lagllama':'Lag-Llama',
}
CLASS4_MODELS = {'PatchTST','iTransformer','TimesNet','FEDformer','DLinear','Autoformer'}
FOUNDATION_MODELS = {'Chronos','TimesFM','Moirai','Lag-Llama'}
ALL_MODELS = tuple(MODEL_ALIASES[k] for k in MODEL_ALIASES if '-' not in k or k in ('st-norm','lag-llama'))

@dataclass(frozen=True)
class DatasetSpec:
    interval_minutes: int
    temporal_protocol: str
    source: str
    report_primary_mape: bool
    notes: str = ''

DATASET_SPECS = {
    'NYCTaxi': DatasetSpec(30, 'ST-SSL pre-generated historical context -> next 1 step', 'ST-SSL', False,'Native 30-minute resolution; do not resample to 5 minutes.'),
    'NYCBike1': DatasetSpec(60, 'ST-SSL pre-generated historical context -> next 1 step', 'ST-SSL', False,'Native 1-hour resolution; do not resample to 5 minutes.'),
    'NYCBike2': DatasetSpec(30, 'ST-SSL pre-generated historical context -> next 1 step', 'ST-SSL', False,'Native 30-minute resolution; do not resample to 5 minutes.'),
    'BJTaxi': DatasetSpec(30, 'ST-SSL pre-generated historical context -> next 1 step', 'ST-SSL', False,'Native 30-minute resolution; do not resample to 5 minutes.'),
    'METR-LA': DatasetSpec(5, 'previous 12 native steps -> next 12 steps', 'DCRNN-format', True),
    'PEMS-BAY': DatasetSpec(5, 'previous 12 native steps -> next 12 steps', 'DCRNN-format', True),
    'PEMS04': DatasetSpec(5, 'previous 12 native steps -> next 12 steps', 'ASTGCN raw', True),
    'PEMS08': DatasetSpec(5, 'previous 12 native steps -> next 12 steps', 'ASTGCN raw', True),
}

def canonical_dataset(name: str) -> str:
    return DATASET_ALIASES.get(name.strip().lower(), name)
def canonical_model(name: str) -> str:
    key=name.strip().lower()
    if key not in MODEL_ALIASES: raise KeyError(f'Unknown model {name}. Supported: {sorted(set(MODEL_ALIASES.values()))}')
    return MODEL_ALIASES[key]
def dataset_spec(name: str) -> DatasetSpec:
    name=canonical_dataset(name)
    if name not in DATASET_SPECS: raise KeyError(f'No DatasetSpec for {name}')
    return DATASET_SPECS[name]
