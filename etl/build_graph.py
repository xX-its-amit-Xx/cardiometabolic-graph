"""Materialize the cardiometabolic knowledge graph in Neo4j from Postgres staging.

This script reads the staged Postgres tables produced by ``load_mimic.py``,
``load_nhanes.py``, and ``load_pathways.py`` plus the parquet synthetic
engagement events, and writes nodes + relationships into Neo4j as defined in
``schema/graph_schema.md``.

It is:
  * idempotent — uses MERGE on every node/edge keyed by the primary key
  * chunked — batches writes via UNWIND for throughput
  * resumable — each batch commits independently

Usage:
    python -m etl.build_graph --batch 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from ._common import (
    PostgresConfig,
    chunked,
    log,
    neo4j_session,
    now_utc,
    synthetic_path,
)

CYPHER_CONSTRAINTS = Path(__file__).resolve().parent.parent / "schema" / "cypher_constraints.cql"


def apply_constraints(session) -> None:
    if not CYPHER_CONSTRAINTS.exists():
        log.warning("constraints file missing: %s", CYPHER_CONSTRAINTS)
        return
    log.info("applying constraints from %s", CYPHER_CONSTRAINTS)
    text_ = CYPHER_CONSTRAINTS.read_text(encoding="utf-8")
    for stmt in [s.strip() for s in text_.split(";") if s.strip() and not s.strip().startswith("//")]:
        # also skip block comments
        if stmt.startswith("/*"):
            continue
        session.run(stmt)


# ---------- Patients & encounters from MIMIC ----------

def write_patients(session, pg_engine, batch: int) -> int:
    sql = text("""
        SELECT subject_id, gender, anchor_age, anchor_year, source_system
        FROM mimic.patients
    """)
    cypher = """
        UNWIND $rows AS row
        MERGE (p:Patient {patient_id: row.patient_id})
        SET p.sex = row.sex,
            p.birth_year = row.birth_year,
            p.cohort = row.cohort,
            p.source_system = row.source_system,
            p.ingested_at = datetime($ts)
    """
    total = 0
    with pg_engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    if df.empty:
        return 0
    df["patient_id"] = "MIMIC" + df["subject_id"].astype(str)
    df["sex"] = df["gender"]
    df["birth_year"] = df["anchor_year"].astype("Int64") - df["anchor_age"].astype("Int64")
    df["cohort"] = "mimic-iv"
    rows = df[["patient_id", "sex", "birth_year", "cohort", "source_system"]].to_dict("records")
    for batch_rows in chunked(rows, batch):
        session.run(cypher, rows=batch_rows, ts=now_utc().isoformat())
        total += len(batch_rows)
    log.info("Patient nodes (MIMIC): %d", total)
    return total


def write_encounters(session, pg_engine, batch: int) -> int:
    sql = text("""
        SELECT a.hadm_id, a.subject_id, a.admittime, a.dischtime, a.admission_type
        FROM mimic.admissions a
    """)
    cypher = """
        UNWIND $rows AS row
        MATCH (p:Patient {patient_id: row.patient_id})
        MERGE (e:Encounter {encounter_id: row.encounter_id})
        SET e.start_ts = datetime(row.start_ts),
            e.end_ts = CASE WHEN row.end_ts IS NULL THEN null ELSE datetime(row.end_ts) END,
            e.encounter_type = row.encounter_type,
            e.patient_id = row.patient_id
        MERGE (p)-[:HAS_ENCOUNTER]->(e)
    """
    with pg_engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    if df.empty:
        return 0
    df["encounter_id"] = "ENC" + df["hadm_id"].astype(str)
    df["patient_id"] = "MIMIC" + df["subject_id"].astype(str)
    df["start_ts"] = pd.to_datetime(df["admittime"]).dt.tz_localize("UTC").astype(str)
    df["end_ts"] = pd.to_datetime(df["dischtime"]).dt.tz_localize("UTC").astype(str)
    df.loc[df["end_ts"] == "NaT", "end_ts"] = None
    df["encounter_type"] = df["admission_type"].fillna("UNKNOWN")
    rows = df[["encounter_id", "patient_id", "start_ts", "end_ts", "encounter_type"]].to_dict("records")
    total = 0
    for batch_rows in chunked(rows, batch):
        session.run(cypher, rows=batch_rows)
        total += len(batch_rows)
    log.info("Encounter nodes: %d", total)
    return total


def write_labs(session, pg_engine, batch: int) -> int:
    sql = text("""
        SELECT lab_id, subject_id, hadm_id, loinc, name, value, unit, charttime
        FROM mimic.labevents
        WHERE loinc IS NOT NULL AND value IS NOT NULL
    """)
    cypher = """
        UNWIND $rows AS row
        MATCH (p:Patient {patient_id: row.patient_id})
        OPTIONAL MATCH (e:Encounter {encounter_id: row.encounter_id})
        MERGE (l:LabResult {lab_id: row.lab_id})
        SET l.patient_id = row.patient_id,
            l.encounter_id = row.encounter_id,
            l.loinc = row.loinc,
            l.name = row.name,
            l.value = row.value,
            l.unit = row.unit,
            l.taken_ts = datetime(row.taken_ts)
        FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
            MERGE (e)-[:HAS_LAB]->(l)
        )
    """
    with pg_engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    if df.empty:
        return 0
    df["lab_id"] = "LAB" + df["lab_id"].astype(str)
    df["patient_id"] = "MIMIC" + df["subject_id"].astype(str)
    df["encounter_id"] = df["hadm_id"].apply(lambda x: f"ENC{int(x)}" if pd.notna(x) else None)
    df["taken_ts"] = pd.to_datetime(df["charttime"]).dt.tz_localize("UTC").astype(str)
    rows = df[["lab_id", "patient_id", "encounter_id", "loinc", "name", "value", "unit", "taken_ts"]].to_dict("records")
    total = 0
    for batch_rows in chunked(rows, batch):
        session.run(cypher, rows=batch_rows)
        total += len(batch_rows)
    log.info("LabResult nodes: %d", total)
    return total


# ---------- Behavioral events: synthetic engagement + NHANES ----------

def write_synthetic_engagement(session, batch: int) -> int:
    path = synthetic_path() / "engagement_events.parquet"
    if not path.exists():
        log.warning("synthetic engagement events not found: %s", path)
        return 0
    df = pd.read_parquet(path)
    df["event_id"] = "SYE" + df.index.astype(str)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).astype(str)
    # Make sure patients exist as nodes too (synthetic IDs not in MIMIC)
    patients = df[["patient_id"]].drop_duplicates()
    patients["sex"] = "U"
    patients["birth_year"] = 1980
    patients["cohort"] = "synthetic"
    patients["source_system"] = "synthetic-v1"
    p_cypher = """
        UNWIND $rows AS row
        MERGE (p:Patient {patient_id: row.patient_id})
        ON CREATE SET p.sex = row.sex, p.birth_year = row.birth_year,
                      p.cohort = row.cohort, p.source_system = row.source_system
    """
    for chunk in chunked(patients.to_dict("records"), batch):
        session.run(p_cypher, rows=chunk)

    e_cypher = """
        UNWIND $rows AS row
        MATCH (p:Patient {patient_id: row.patient_id})
        MERGE (b:BehavioralEvent {event_id: row.event_id})
        SET b.patient_id = row.patient_id,
            b.kind = row.kind,
            b.ts = datetime(row.ts),
            b.value = row.value,
            b.source_system = 'synthetic-v1'
        MERGE (p)-[:ENGAGED_WITH]->(b)
    """
    rows = df[["event_id", "patient_id", "kind", "ts", "value"]].to_dict("records")
    total = 0
    for chunk in chunked(rows, batch):
        session.run(e_cypher, rows=chunk)
        total += len(chunk)
    log.info("BehavioralEvent (synthetic): %d", total)
    return total


# ---------- Pathways, genes, metabolites ----------

def write_pathways(session, pg_engine, batch: int) -> int:
    with pg_engine.connect() as conn:
        pdf = pd.read_sql(text("SELECT pathway_id, name, source FROM pathways.pathway"), conn)
        gdf = pd.read_sql(text("SELECT gene_symbol, pathway_id FROM pathways.gene_pathway"), conn)
        mdf = pd.read_sql(text("SELECT chebi_id, pathway_id FROM pathways.metabolite_pathway"), conn)
        hdf = pd.read_sql(text("SELECT parent_id, child_id FROM pathways.pathway_hierarchy"), conn)

    if pdf.empty:
        log.warning("no pathways staged; skipping graph write")
        return 0

    total = 0
    p_cypher = """
        UNWIND $rows AS row
        MERGE (p:Pathway {pathway_id: row.pathway_id})
        SET p.name = row.name, p.source = row.source
    """
    for chunk in chunked(pdf.to_dict("records"), batch):
        session.run(p_cypher, rows=chunk)
        total += len(chunk)

    if not gdf.empty:
        g_cypher = """
            UNWIND $rows AS row
            MERGE (g:Gene {gene_symbol: row.gene_symbol})
            WITH g, row
            MATCH (p:Pathway {pathway_id: row.pathway_id})
            MERGE (g)-[:PARTICIPATES_IN]->(p)
        """
        for chunk in chunked(gdf.to_dict("records"), batch):
            session.run(g_cypher, rows=chunk)

    if not mdf.empty:
        m_cypher = """
            UNWIND $rows AS row
            MERGE (m:Metabolite {chebi_id: row.chebi_id})
            WITH m, row
            MATCH (p:Pathway {pathway_id: row.pathway_id})
            MERGE (m)-[:IN_PATHWAY]->(p)
        """
        for chunk in chunked(mdf.to_dict("records"), batch):
            session.run(m_cypher, rows=chunk)

    if not hdf.empty:
        h_cypher = """
            UNWIND $rows AS row
            MATCH (p1:Pathway {pathway_id: row.parent_id})
            MATCH (p2:Pathway {pathway_id: row.child_id})
            MERGE (p1)-[:REGULATES {direction: 'parent_of'}]->(p2)
        """
        for chunk in chunked(hdf.to_dict("records"), batch):
            session.run(h_cypher, rows=chunk)

    log.info("Pathway/Gene/Metabolite written")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--skip-clinical", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-pathways", action="store_true")
    args = parser.parse_args()

    pg_engine = create_engine(PostgresConfig.from_env().dsn, future=True)

    with neo4j_session() as session:
        apply_constraints(session)
        if not args.skip_clinical:
            write_patients(session, pg_engine, args.batch)
            write_encounters(session, pg_engine, args.batch)
            write_labs(session, pg_engine, args.batch)
        if not args.skip_synthetic:
            write_synthetic_engagement(session, args.batch)
        if not args.skip_pathways:
            write_pathways(session, pg_engine, args.batch)

    log.info("graph build complete")


if __name__ == "__main__":
    main()
