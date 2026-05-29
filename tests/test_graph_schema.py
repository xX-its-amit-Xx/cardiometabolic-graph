"""Schema contract tests.

Unit-level tests (no DB needed) check that:
  * the constraint file parses and references every declared label,
  * the schema markdown lists each node type at least once.

Integration tests (marked ``integration``) connect to Neo4j and verify the
constraints were applied. They are skipped unless ``NEO4J_URI`` is set.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONSTRAINTS = ROOT / "schema" / "cypher_constraints.cql"
SCHEMA_MD = ROOT / "schema" / "graph_schema.md"

EXPECTED_LABELS = {
    "Patient",
    "Encounter",
    "LabResult",
    "Vital",
    "Medication",
    "BehavioralEvent",
    "Pathway",
    "Gene",
    "Metabolite",
}


def test_constraints_file_exists():
    assert CONSTRAINTS.exists(), CONSTRAINTS


def test_constraints_cover_every_node_label():
    txt = CONSTRAINTS.read_text()
    missing = []
    for label in EXPECTED_LABELS:
        if not re.search(rf"FOR \(\w+:{label}\)", txt):
            missing.append(label)
    assert not missing, f"missing constraint for: {missing}"


def test_schema_markdown_lists_every_label():
    md = SCHEMA_MD.read_text()
    for label in EXPECTED_LABELS:
        assert label in md, f"label {label} not documented in schema_graph.md"


def test_constraints_are_idempotent_keywords():
    """Every CREATE statement must include IF NOT EXISTS — re-running must
    not raise on an already-initialised database."""
    for stmt in CONSTRAINTS.read_text().split(";"):
        stripped = stmt.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if stripped.upper().startswith("CREATE"):
            assert "IF NOT EXISTS" in stripped.upper(), stripped


@pytest.mark.integration
def test_constraints_applied_in_neo4j():
    if not os.getenv("NEO4J_URI"):
        pytest.skip("NEO4J_URI not configured")
    from etl._common import neo4j_session

    with neo4j_session() as session:
        result = session.run("SHOW CONSTRAINTS").data()
        names = {row.get("name", "") for row in result}
        # at least one per primary key constraint we declared
        for expected in ("patient_id", "encounter_id", "lab_id", "pathway_id"):
            assert any(expected in n for n in names), f"constraint {expected} missing"
