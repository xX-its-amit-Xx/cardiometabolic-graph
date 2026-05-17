"""ETL correctness tests — exercise the pure-Python pieces that do not
require Docker. Integration tests touching Neo4j/Postgres live in
``tests/test_graph_schema.py`` under the ``integration`` marker.
"""

from __future__ import annotations

import pandas as pd

from data.synthetic.generate_engagement_logs import (
    ARCHETYPES,
    generate,
)
from etl._common import chunked


def test_synthetic_generation_is_deterministic():
    a, _ = generate(n_patients=10, days=30, seed=42)
    b, _ = generate(n_patients=10, days=30, seed=42)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_synthetic_archetype_distribution():
    _, arche = generate(n_patients=400, days=14, seed=42)
    counts = arche["archetype"].value_counts(normalize=True)
    for a in ARCHETYPES:
        # within +/- 8 pp of declared probabilities
        assert a in counts.index, f"archetype {a} missing"
    # power_user expected ~15%
    assert 0.07 < counts.get("power_user", 0) < 0.23


def test_synthetic_event_kinds_complete():
    events, _ = generate(n_patients=50, days=90, seed=1)
    kinds = set(events["kind"].unique())
    assert {"app_open", "message_response", "glucose_log"} <= kinds


def test_synthetic_glucose_values_plausible():
    events, _ = generate(n_patients=50, days=60, seed=2)
    glog = events[events["kind"] == "glucose_log"]
    assert (glog["value"].between(20, 400)).mean() > 0.95


def test_chunked_handles_remainder():
    batches = list(chunked(range(7), 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6]]


def test_chunked_empty():
    assert list(chunked([], 5)) == []
