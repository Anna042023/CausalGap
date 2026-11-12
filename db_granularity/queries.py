"""Nested aggregation grains over the unchanged R3 validated-pair relation."""

PAIR_COLUMNS = ("run_id", "origin", "node_id", "horizon", "error0", "errork")
KEY_COLUMNS = PAIR_COLUMNS[:4]
GRAINS = (
    {"grain": "run", "keys": KEY_COLUMNS[:1], "label": "Run", "expected_groups": 252},
    {"grain": "origin", "keys": KEY_COLUMNS[:2], "label": "Run + origin", "expected_groups": 756},
    {"grain": "node", "keys": KEY_COLUMNS[:3], "label": "Run + origin + node", "expected_groups": 24948},
    {"grain": "horizon", "keys": KEY_COLUMNS[:4], "label": "Full pair key", "expected_groups": 299376},
)
GRAIN_MAP = {g["grain"]: g for g in GRAINS}
SCHEMA = """
CREATE TABLE validated_pair (
 run_id INTEGER NOT NULL,
 origin INTEGER NOT NULL,
 node_id INTEGER NOT NULL,
 horizon INTEGER NOT NULL,
 error0 BIGINT NOT NULL,
 errork BIGINT NOT NULL
)
"""


def query_sql(grain):
    keys = ", ".join(GRAIN_MAP[grain]["keys"])
    return (f"SELECT {keys}, COUNT(*) AS pairs, SUM(error0) AS error0, "
            f"SUM(errork) AS errork FROM validated_pair GROUP BY {keys} "
            f"ORDER BY {keys}")
