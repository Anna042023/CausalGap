"""Portable SQL for the new controlled relational experiment.

Forecast keys are logical keys on ingestion. The checked query audits every key
of each selected run before permitting any pair from that run into an aggregate.
"""

INTERFACE = "common"
TAU = 1.0


def naive_sql():
    return """
WITH errors AS (
 SELECT b.run_id, COUNT(*) AS pairs,
        SUM(ABS(b.prediction-b.target)) AS error0,
        SUM(ABS(k.prediction-k.target)) AS errork
 FROM forecast_atom b JOIN forecast_atom k
   ON b.run_id=k.run_id AND b.origin=k.origin
  AND b.node_id=k.node_id AND b.horizon=k.horizon
 JOIN evaluation_run r ON r.run_id=b.run_id
 JOIN scenario s ON s.scenario_id=r.scenario_id
 WHERE b.environment=0 AND k.environment=1
   AND b.valid=1 AND k.valid=1 AND r.interface='common'
   AND s.outcome_consistent=1
 GROUP BY b.run_id
)
SELECT run_id,pairs,error0,errork,
       1.0*error0/pairs AS mae0,1.0*errork/pairs AS maek,
       1.0*(errork-error0)/(error0+1.0*pairs) AS rd
FROM errors ORDER BY run_id
""".strip()


def groups_sql():
    return """
SELECT a.run_id,a.origin,a.node_id,a.horizon,
       SUM(CASE WHEN a.environment=0 THEN 1 ELSE 0 END) AS n0,
       SUM(CASE WHEN a.environment=1 THEN 1 ELSE 0 END) AS nk,
       MAX(CASE WHEN a.environment=0 THEN a.valid ELSE NULL END) AS v0,
       MAX(CASE WHEN a.environment=1 THEN a.valid ELSE NULL END) AS vk,
       SUM(CASE WHEN a.graph_version !=
           CASE WHEN a.environment=0 THEN s.baseline_graph ELSE s.changed_graph END
           THEN 1 ELSE 0 END) AS wrong_graph,
       MAX(CASE WHEN a.environment=0 THEN ABS(a.prediction-a.target)
           ELSE NULL END) AS error0,
       MAX(CASE WHEN a.environment=1 THEN ABS(a.prediction-a.target)
           ELSE NULL END) AS errork
FROM forecast_atom a JOIN evaluation_run r ON r.run_id=a.run_id
JOIN scenario s ON s.scenario_id=r.scenario_id
WHERE r.interface='common' AND s.outcome_consistent=1
GROUP BY a.run_id,a.origin,a.node_id,a.horizon
""".strip()


def violation_sql():
    return f"""
WITH key_groups AS ({groups_sql()})
SELECT run_id,
       MAX(CASE WHEN n0=0 OR nk=0 THEN 1 ELSE 0 END) AS missing_counterpart,
       MAX(CASE WHEN n0>1 OR nk>1 THEN 1 ELSE 0 END) AS duplicate_key,
       MAX(CASE WHEN wrong_graph>0 THEN 1 ELSE 0 END) AS wrong_graph,
       MAX(CASE WHEN n0>0 AND nk>0 AND v0!=vk THEN 1 ELSE 0 END) AS support_mismatch
FROM key_groups
WHERE n0!=1 OR nk!=1 OR v0!=vk OR wrong_graph>0
GROUP BY run_id ORDER BY run_id
""".strip()


def pair_rows_sql():
    return f"""
WITH key_groups AS ({groups_sql()}),
bad_runs AS (
 SELECT DISTINCT run_id FROM key_groups
 WHERE n0!=1 OR nk!=1 OR v0!=vk OR wrong_graph>0
)
SELECT p.run_id,p.origin,p.node_id,p.horizon,p.error0,p.errork
FROM key_groups p LEFT JOIN bad_runs b ON b.run_id=p.run_id
WHERE b.run_id IS NULL AND p.v0=1 AND p.vk=1
""".strip()


def aggregate_sql(source):
    return f"""
WITH errors AS (
 SELECT run_id,COUNT(*) AS pairs,SUM(error0) AS error0,SUM(errork) AS errork
 FROM {source} GROUP BY run_id
)
SELECT run_id,pairs,error0,errork,
       1.0*error0/pairs AS mae0,1.0*errork/pairs AS maek,
       1.0*(errork-error0)/(error0+1.0*pairs) AS rd
FROM errors ORDER BY run_id
""".strip()


def checked_sql():
    return aggregate_sql(f"({pair_rows_sql()}) validated")


def cached_sql():
    return aggregate_sql("validated_pair")

