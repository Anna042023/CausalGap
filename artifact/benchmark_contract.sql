-- Executable SQLite reference contract introduced for the ICDE EAB rewrite.
-- It does NOT contain or reconstruct the original forecasting run records.
-- Tensor files may remain columnar/binary artifacts with stable logical keys.
PRAGMA foreign_keys = ON;

CREATE TABLE artifact (
    artifact_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK(length(sha256)=64
                              AND sha256 NOT GLOB '*[^0-9a-f]*'),
    location TEXT NOT NULL
);
CREATE TABLE dataset (
    dataset_id TEXT PRIMARY KEY,
    interval_seconds INTEGER NOT NULL CHECK(interval_seconds>0)
);
CREATE TABLE topology_version (
    dataset_id TEXT NOT NULL REFERENCES dataset(dataset_id),
    graph_version TEXT NOT NULL,
    parent_version TEXT,
    edge_artifact TEXT NOT NULL REFERENCES artifact(artifact_id),
    node_order_artifact TEXT NOT NULL REFERENCES artifact(artifact_id),
    PRIMARY KEY(dataset_id,graph_version),
    FOREIGN KEY(dataset_id,parent_version)
        REFERENCES topology_version(dataset_id,graph_version)
);
CREATE TABLE observation (
    dataset_id TEXT NOT NULL REFERENCES dataset(dataset_id),
    trajectory_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    timestamp_utc INTEGER NOT NULL,
    feature TEXT NOT NULL,
    value REAL,
    valid INTEGER NOT NULL CHECK(valid IN (0,1)),
    partition_id TEXT NOT NULL,
    source_artifact TEXT NOT NULL REFERENCES artifact(artifact_id),
    PRIMARY KEY(dataset_id,trajectory_id,node_id,timestamp_utc,feature),
    CHECK(valid=0 OR value IS NOT NULL)
);
CREATE TABLE scenario (
    scenario_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES dataset(dataset_id),
    setting TEXT NOT NULL CHECK(setting IN
        ('controlled','abstract_sumo','observational_input_test')),
    baseline_graph TEXT NOT NULL,
    changed_graph TEXT NOT NULL,
    baseline_trajectory TEXT NOT NULL,
    changed_trajectory TEXT NOT NULL,
    intervention_type TEXT NOT NULL,
    intensity REAL NOT NULL,
    perturbation_seed INTEGER NOT NULL,
    exogenous_pair_id TEXT,
    support_artifact TEXT NOT NULL REFERENCES artifact(artifact_id),
    outcome_consistent INTEGER NOT NULL CHECK(outcome_consistent IN (0,1)),
    FOREIGN KEY(dataset_id,baseline_graph)
        REFERENCES topology_version(dataset_id,graph_version),
    FOREIGN KEY(dataset_id,changed_graph)
        REFERENCES topology_version(dataset_id,graph_version),
    CHECK(setting!='observational_input_test' OR outcome_consistent=0)
);
CREATE TABLE calibration (
    calibration_id TEXT PRIMARY KEY,
    config_artifact TEXT NOT NULL REFERENCES artifact(artifact_id),
    membership_artifact TEXT NOT NULL REFERENCES artifact(artifact_id),
    tau REAL NOT NULL CHECK(tau>=0),
    tau_source_artifact TEXT NOT NULL REFERENCES artifact(artifact_id)
);
CREATE TABLE evaluation_run (
    run_id TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL REFERENCES scenario(scenario_id),
    model_id TEXT NOT NULL,
    model_family TEXT NOT NULL,
    training_seed INTEGER NOT NULL,
    checkpoint_artifact TEXT NOT NULL REFERENCES artifact(artifact_id),
    access_mode TEXT NOT NULL CHECK(access_mode IN ('native','common')),
    interface_config TEXT NOT NULL REFERENCES artifact(artifact_id),
    extraction_config TEXT NOT NULL REFERENCES artifact(artifact_id),
    calibration_id TEXT NOT NULL REFERENCES calibration(calibration_id),
    partition_role TEXT NOT NULL CHECK(partition_role IN
        ('training','validation','calibration','test')),
    exposure_role TEXT NOT NULL CHECK(exposure_role IN
        ('baseline','weak','strong')),
    status TEXT NOT NULL CHECK(status IN
        ('complete','failed','excluded','unavailable')),
    exclusion_reason TEXT,
    CHECK(status='complete' OR exclusion_reason IS NOT NULL)
);
CREATE TABLE forecast_atom (
    run_id TEXT NOT NULL REFERENCES evaluation_run(run_id),
    environment TEXT NOT NULL CHECK(environment IN ('baseline','changed')),
    origin INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    horizon INTEGER NOT NULL CHECK(horizon BETWEEN 1 AND 12),
    prediction REAL,
    target REAL,
    target_source TEXT NOT NULL REFERENCES artifact(artifact_id),
    valid INTEGER NOT NULL CHECK(valid IN (0,1)),
    PRIMARY KEY(run_id,environment,origin,node_id,horizon),
    CHECK(valid=0 OR (prediction IS NOT NULL AND target IS NOT NULL))
);
CREATE TABLE representation_pair (
    run_id TEXT NOT NULL REFERENCES evaluation_run(run_id),
    origin INTEGER NOT NULL,
    input_window_artifact TEXT NOT NULL REFERENCES artifact(artifact_id),
    row_order_artifact TEXT NOT NULL REFERENCES artifact(artifact_id),
    baseline_graph_representation TEXT NOT NULL REFERENCES artifact(artifact_id),
    changed_graph_representation TEXT NOT NULL REFERENCES artifact(artifact_id),
    degenerate INTEGER NOT NULL CHECK(degenerate IN (0,1)),
    PRIMARY KEY(run_id,origin)
);
CREATE TABLE h1_unit (
    unit_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_family TEXT NOT NULL,
    dataset_id TEXT NOT NULL REFERENCES dataset(dataset_id),
    access_mode TEXT NOT NULL CHECK(access_mode IN ('native','common')),
    intervention_type TEXT NOT NULL,
    aggregation_key TEXT NOT NULL,
    outer_fold TEXT NOT NULL,
    strong_target_artifact TEXT NOT NULL REFERENCES artifact(artifact_id),
    UNIQUE(model_id,dataset_id,access_mode,intervention_type,aggregation_key)
);
CREATE TABLE weak_summary_input (
    unit_id TEXT NOT NULL REFERENCES h1_unit(unit_id),
    weak_level INTEGER NOT NULL CHECK(weak_level BETWEEN 1 AND 3),
    rd REAL NOT NULL,
    dmis_cal REAL NOT NULL,
    normalized_intensity REAL NOT NULL,
    source_artifact TEXT NOT NULL REFERENCES artifact(artifact_id),
    PRIMARY KEY(unit_id,weak_level)
);

-- These views identify violations. Zero rows is meaningful only after a
-- complete, independently verified ingest and a nonempty run manifest.
CREATE VIEW violation_missing_counterpart AS
SELECT b.run_id,b.environment,b.origin,b.node_id,b.horizon
FROM forecast_atom b
LEFT JOIN forecast_atom k
  ON b.run_id=k.run_id AND b.origin=k.origin
 AND b.node_id=k.node_id AND b.horizon=k.horizon
 AND b.environment!=k.environment
WHERE k.run_id IS NULL;

CREATE VIEW violation_support_disagreement AS
SELECT b.run_id,b.origin,b.node_id,b.horizon,b.valid AS baseline_valid,
       k.valid AS changed_valid
FROM forecast_atom b JOIN forecast_atom k
  ON b.run_id=k.run_id AND b.origin=k.origin
 AND b.node_id=k.node_id AND b.horizon=k.horizon
WHERE b.environment='baseline' AND k.environment='changed'
  AND b.valid!=k.valid;

CREATE VIEW violation_strong_calibration AS
SELECT run_id FROM evaluation_run
WHERE partition_role='calibration' AND exposure_role='strong';

CREATE VIEW violation_weak_cardinality AS
SELECT u.unit_id,COUNT(w.weak_level) AS levels
FROM h1_unit u LEFT JOIN weak_summary_input w ON u.unit_id=w.unit_id
GROUP BY u.unit_id HAVING COUNT(w.weak_level)!=3;

CREATE VIEW paired_errors AS
SELECT b.run_id,
       AVG(ABS(b.prediction-b.target)) AS mae0,
       AVG(ABS(k.prediction-k.target)) AS maek,
       COUNT(*) AS scored_atoms
FROM forecast_atom b JOIN forecast_atom k
  ON b.run_id=k.run_id AND b.origin=k.origin
 AND b.node_id=k.node_id AND b.horizon=k.horizon
JOIN evaluation_run r ON r.run_id=b.run_id
WHERE b.environment='baseline' AND k.environment='changed'
  AND b.valid=1 AND k.valid=1 AND r.status='complete'
  AND NOT EXISTS (SELECT 1 FROM violation_missing_counterpart v
                  WHERE v.run_id=b.run_id)
  AND NOT EXISTS (SELECT 1 FROM violation_support_disagreement v
                  WHERE v.run_id=b.run_id)
GROUP BY b.run_id;

CREATE VIEW paired_degradation AS
SELECT p.run_id,s.setting,s.outcome_consistent,p.mae0,p.maek,p.scored_atoms,
       CASE WHEN p.mae0+c.tau>0
       THEN (p.maek-p.mae0)/(p.mae0+c.tau) ELSE NULL END AS rd
FROM paired_errors p
JOIN evaluation_run r ON r.run_id=p.run_id
JOIN scenario s ON s.scenario_id=r.scenario_id
JOIN calibration c ON c.calibration_id=r.calibration_id
WHERE s.outcome_consistent=1;

-- This relation never pools native and common interfaces, and retains
-- status accounting outside the metric view.
CREATE VIEW run_accounting AS
SELECT s.setting,r.model_id,r.access_mode,r.status,COUNT(*) AS runs
FROM evaluation_run r JOIN scenario s ON s.scenario_id=r.scenario_id
GROUP BY s.setting,r.model_id,r.access_mode,r.status;
