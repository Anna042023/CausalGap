from pathlib import Path
import os
PROJECT = Path(os.environ.get('CAUSALGAP_ROOT', '/data/wanganna/CausalGap'))
CODE = PROJECT / 'unified'
ROOT = PROJECT / 'abstract_sumo_52'
SOURCE = ROOT / 'source'
EXPERIMENT = PROJECT / 'experiment_results' / 'sumo_replication'
H = F = 12
BIN = 300
WARMUP = 900
T0 = WARMUP + H * BIN
END = T0 + F * BIN
MPH = 2.2369362920544
LEVELS = {
    'edge_removal': [('s=0.05', .05), ('s=0.10', .10), ('s=0.15', .15), ('s=0.20', .20)],
    'weight_modification': [('gamma=0.8', .8), ('gamma=0.6', .6), ('gamma=0.4', .4), ('gamma=0.2', .2)],
    'node_isolation': [('s=0.10', .10), ('s=0.20', .20), ('s=0.40', .40), ('s=0.60', .60)]
}
