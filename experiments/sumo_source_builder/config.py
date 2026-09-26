from pathlib import Path

PROJECT_ROOT = Path('/data/wanganna/CausalGap')
UNIFIED_ROOT = PROJECT_ROOT / 'unified'
DATA_ROOT = PROJECT_ROOT / 'datasets' / 'METR-LA'
SENSOR_GRAPH_ROOT = DATA_ROOT / 'sensor_graph'
LOCATIONS_CSV = SENSOR_GRAPH_ROOT / 'graph_sensor_locations.csv'
H5_PATH = DATA_ROOT / 'metr-la.h5'

WORK_ROOT = PROJECT_ROOT / 'sumo_metrla_builder'
NETWORK_ROOT = WORK_ROOT / 'network'
MAPPING_ROOT = WORK_ROOT / 'mapping'
CALIB_ROOT = WORK_ROOT / 'calibration'
ROUTE_ROOT = WORK_ROOT / 'routes'
LOG_ROOT = WORK_ROOT / 'logs'
SOURCE_ROOT = PROJECT_ROOT / 'sumo_h1_source'

Pems_FLOW_CSV = CALIB_ROOT / 'pems_hourly_flow.csv'
CALIBRATED_OD_JSON = CALIB_ROOT / 'calibrated_od.json'

# Keep these exactly aligned with the current H1 V2 protocol already used by the project.
H1_LEVELS = {
    'edge_removal': [
        dict(level='s=0.05', phase='weak', severity=0.05, gamma=None),
        dict(level='s=0.10', phase='weak', severity=0.10, gamma=None),
        dict(level='s=0.15', phase='weak', severity=0.15, gamma=None),
        dict(level='s=0.20', phase='strong', severity=0.20, gamma=None),
    ],
    'weight_modification': [
        dict(level='gamma=0.8', phase='weak', severity=None, gamma=0.8),
        dict(level='gamma=0.6', phase='weak', severity=None, gamma=0.6),
        dict(level='gamma=0.4', phase='weak', severity=None, gamma=0.4),
        dict(level='gamma=0.2', phase='strong', severity=None, gamma=0.2),
    ],
    'node_isolation': [
        dict(level='s=0.10', phase='weak', severity=0.10, gamma=None),
        dict(level='s=0.20', phase='weak', severity=0.20, gamma=None),
        dict(level='s=0.40', phase='weak', severity=0.40, gamma=None),
        dict(level='s=0.60', phase='strong', severity=0.60, gamma=None),
    ],
}

EVAL_INTERVENTION_SEEDS = [0, 1, 2, 3, 4]
SIMULATION_SEEDS_FINAL = [100, 101, 102]
STEP_SECONDS = 1.0
BIN_SECONDS = 300
HISTORY_STEPS = 12
FUTURE_STEPS = 12
WARMUP_SECONDS = 1800
T0_SECONDS = WARMUP_SECONDS + HISTORY_STEPS * BIN_SECONDS
END_SECONDS = T0_SECONDS + FUTURE_STEPS * BIN_SECONDS
MPS_TO_MPH = 2.2369362920544

# OSM download padding around the 207 METR-LA sensor bounding box.
BBOX_PAD_DEG = 0.03
OSM_TILES = 16
MAX_SENSOR_EDGE_DISTANCE_M = 500.0
