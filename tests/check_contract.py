#!/usr/bin/env python3
"""Check SQL semantics with artificial, in-memory fixtures only.

The identifiers, forecast values, and artifact hashes below are invented test
data. These checks are not forecasting experiments or performance measurements.
The contract path is resolved relative to this file. From the package root,
run: python tests/check_contract.py
"""

from pathlib import Path
import sqlite3
import unittest


CONTRACT = Path(__file__).resolve().parents[1] / "artifact/benchmark_contract.sql"
ARTIFACT = "artificial-fixture-artifact"


class ContractChecks(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.addCleanup(self.db.close)
        self.db.executescript(CONTRACT.read_text(encoding="utf-8"))
        self.db.execute(
            "INSERT INTO artifact VALUES (?,?,?,?)",
            (ARTIFACT, "test-fixture", "0" * 64, "fixture://not-a-real-file"),
        )
        self.db.execute("INSERT INTO dataset VALUES ('fixture-data',300)")
        self.db.execute(
            "INSERT INTO topology_version VALUES ('fixture-data','g0',NULL,?,?)",
            (ARTIFACT, ARTIFACT),
        )
        self.db.execute(
            "INSERT INTO topology_version VALUES ('fixture-data','g1','g0',?,?)",
            (ARTIFACT, ARTIFACT),
        )
        self.db.execute(
            "INSERT INTO calibration VALUES ('fixture-calibration',?,?,1.0,?)",
            (ARTIFACT, ARTIFACT, ARTIFACT),
        )

    def add_run(self, name, setting="controlled", consistent=1,
                partition="test", exposure="strong"):
        self.db.execute(
            """INSERT INTO scenario (
                scenario_id,dataset_id,setting,baseline_graph,changed_graph,
                baseline_trajectory,changed_trajectory,intervention_type,
                intensity,perturbation_seed,exogenous_pair_id,support_artifact,
                outcome_consistent
            ) VALUES (?,'fixture-data',?,'g0','g1','r0','r1','removal',
                      0.2,0,'fixture-pair',?,?)""",
            (name, setting, ARTIFACT, consistent),
        )
        self.db.execute(
            """INSERT INTO evaluation_run (
                run_id,scenario_id,model_id,model_family,training_seed,
                checkpoint_artifact,access_mode,interface_config,
                extraction_config,calibration_id,partition_role,exposure_role,
                status,exclusion_reason
            ) VALUES (?,?,'fixture-model','fixture-family',0,?,'native',?, ?,
                      'fixture-calibration',?,?,'complete',NULL)""",
            (name, name, ARTIFACT, ARTIFACT, ARTIFACT, partition, exposure),
        )

    def add_atom(self, run, environment, origin, node, horizon, prediction,
                 target, valid=1):
        self.db.execute(
            "INSERT INTO forecast_atom VALUES (?,?,?,?,?,?,?,?,?)",
            (run, environment, origin, node, horizon, prediction, target,
             ARTIFACT, valid),
        )

    def add_matched_atoms(self, run):
        # Independent expected absolute errors: baseline = 2,4,6,0 (mean 3),
        # changed = 5,1,9,5 (mean 5). With tau=1, expected RD is 1/2.
        # Different origins, nodes, horizons, and targets expose incorrect joins.
        pairs = [
            (1200, "n1", 1, 12, 10, 105, 100),
            (1200, "n1", 2, 16, 20, 199, 200),
            (1200, "n2", 1, 36, 30, 309, 300),
            (1500, "n1", 1, 40, 40, 395, 400),
        ]
        for origin, node, horizon, p0, y0, pk, yk in pairs:
            self.add_atom(run, "baseline", origin, node, horizon, p0, y0)
            self.add_atom(run, "changed", origin, node, horizon, pk, yk)
        # Shared invalid support must be excluded from both error denominators.
        self.add_atom(run, "baseline", 1500, "masked", 1, 10000, 0, valid=0)
        self.add_atom(run, "changed", 1500, "masked", 1, -20000, 0, valid=0)

    def ids(self, view, column="run_id"):
        # Only fixed relation/column names in this file are passed here.
        return {row[0] for row in self.db.execute(f"SELECT {column} FROM {view}")}

    def test_matched_pair_has_expected_errors_and_rd(self):
        self.add_run("matched")
        self.add_matched_atoms("matched")
        rows = self.db.execute("SELECT * FROM paired_degradation").fetchall()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["run_id"], row["setting"], row["outcome_consistent"]),
                         ("matched", "controlled", 1))
        self.assertEqual(row["scored_atoms"], 4)
        self.assertAlmostEqual(row["mae0"], 3.0)
        self.assertAlmostEqual(row["maek"], 5.0)
        self.assertAlmostEqual(row["rd"], 0.5)

    def test_only_consistent_outcomes_enter_degradation(self):
        for name, setting, consistent in [
            ("controlled", "controlled", 1),
            ("sumo", "abstract_sumo", 1),
            ("observational", "observational_input_test", 0),
            ("unverified", "controlled", 0),
        ]:
            self.add_run(name, setting, consistent)
            self.add_matched_atoms(name)
        self.assertEqual(self.ids("paired_errors"),
                         {"controlled", "sumo", "observational", "unverified"})
        provenance = {
            tuple(row) for row in self.db.execute(
                "SELECT run_id,setting,outcome_consistent FROM paired_degradation"
            )
        }
        self.assertEqual(provenance,
                         {("controlled", "controlled", 1),
                          ("sumo", "abstract_sumo", 1)})
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("UPDATE scenario SET outcome_consistent=1 "
                            "WHERE scenario_id='observational'")

    def test_missing_counterpart_invalidates_entire_run(self):
        self.add_run("intact")
        self.add_matched_atoms("intact")
        for environment in ("baseline", "changed"):
            name = "orphan-" + environment
            self.add_run(name)
            self.add_matched_atoms(name)
            self.add_atom(name, environment, 1800, "extra", 3, 8, 7)
        violations = {tuple(row) for row in self.db.execute(
            "SELECT * FROM violation_missing_counterpart"
        )}
        self.assertEqual(violations,
                         {("orphan-baseline", "baseline", 1800, "extra", 3),
                          ("orphan-changed", "changed", 1800, "extra", 3)})
        self.assertEqual(self.ids("paired_errors"), {"intact"})
        self.assertEqual(self.ids("paired_degradation"), {"intact"})

    def test_support_disagreement_invalidates_entire_run(self):
        self.add_run("intact")
        self.add_matched_atoms("intact")
        for baseline_valid, changed_valid in ((0, 1), (1, 0)):
            name = f"mismatched-{baseline_valid}-{changed_valid}"
            self.add_run(name)
            self.add_matched_atoms(name)
            self.add_atom(name, "baseline", 1800, "extra", 3, 8, 7,
                          baseline_valid)
            self.add_atom(name, "changed", 1800, "extra", 3, 9, 7,
                          changed_valid)
        violations = {tuple(row) for row in self.db.execute(
            "SELECT * FROM violation_support_disagreement"
        )}
        self.assertEqual(violations,
                         {("mismatched-0-1", 1800, "extra", 3, 0, 1),
                          ("mismatched-1-0", 1800, "extra", 3, 1, 0)})
        self.assertEqual(self.ids("paired_errors"), {"intact"})
        self.assertEqual(self.ids("paired_degradation"), {"intact"})

    def test_strong_calibration_is_detected(self):
        self.add_run("leaked", partition="calibration", exposure="strong")
        self.add_run("weak-calibration", partition="calibration", exposure="weak")
        self.add_run("strong-test", partition="test", exposure="strong")
        self.assertEqual(self.ids("violation_strong_calibration"), {"leaked"})

    def test_weak_level_cardinality_and_domain(self):
        for unit, levels in (("empty", ()), ("incomplete", (1, 3)),
                             ("complete", (1, 2, 3))):
            self.db.execute(
                """INSERT INTO h1_unit VALUES (
                    ?,?,'fixture-family','fixture-data','native','removal',
                    'fixture-aggregation','fixture-fold',?)""",
                (unit, "fixture-model-" + unit, ARTIFACT),
            )
            for level in levels:
                self.db.execute(
                    "INSERT INTO weak_summary_input VALUES (?,?,0.1,0.2,0.3,?)",
                    (unit, level, ARTIFACT),
                )
        violations = {tuple(row) for row in self.db.execute(
            "SELECT unit_id,levels FROM violation_weak_cardinality"
        )}
        self.assertEqual(violations, {("empty", 0), ("incomplete", 2)})
        for invalid_level in (0, 4, 1):
            # Out-of-domain levels and duplicate level 1 cannot create a
            # spurious fourth observation for the same H1 unit.
            with self.subTest(level=invalid_level):
                with self.assertRaises(sqlite3.IntegrityError):
                    self.db.execute(
                        "INSERT INTO weak_summary_input VALUES "
                        "('complete',?,0.1,0.2,0.3,?)",
                        (invalid_level, ARTIFACT),
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
