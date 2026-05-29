"""Load MIMIC-IV (demo or full) clinical data into Postgres staging tables.

The demo subset ships with ~100 patients and can be downloaded from
PhysioNet without credentialed access:
    https://physionet.org/content/mimic-iv-demo/2.2/

For full MIMIC-IV (credentialed), drop the unzipped CSVs into
``data/raw/mimic-iv/`` with the same layout the demo uses
(``hosp/`` and ``icu/`` subdirectories). The loader doesn't care which it
finds.

Tables loaded (subset relevant to cardiometabolic modeling):
    patients, admissions, labevents, d_labitems, chartevents, prescriptions

Only a curated set of LOINC/itemid codes are pulled — see ``CARDIO_LOINC``.

Usage:
    python -m etl.load_mimic --mimic-dir data/raw/mimic-iv --chunk 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from ._common import PostgresConfig, log, now_utc, raw_path

# Curated LOINC / MIMIC item codes for cardiometabolic workup.
# Real MIMIC encodes labs by itemid; we map a small subset by hand for the demo.
CARDIO_LOINC = {
    "4548-4": "HbA1c",  # LOINC for glycated hemoglobin
    "2345-7": "glucose_serum",
    "2093-3": "cholesterol_total",
    "2085-9": "hdl",
    "2089-1": "ldl",
    "2571-8": "triglycerides",
}

CARDIO_VITAL_NAMES = {
    "Heart Rate",
    "Non Invasive Blood Pressure systolic",
    "Non Invasive Blood Pressure diastolic",
    "BMI (kg/m2)",
}


def _read_table(
    mimic_dir: Path, name: str, chunksize: int
) -> pd.io.parsers.TextFileReader | pd.DataFrame:
    """Locate a MIMIC-IV CSV across hosp/ and icu/ subdirs."""
    for subdir in ("hosp", "icu", ""):
        path = mimic_dir / subdir / f"{name}.csv.gz"
        if path.exists():
            return pd.read_csv(path, chunksize=chunksize, low_memory=False)
        path = mimic_dir / subdir / f"{name}.csv"
        if path.exists():
            return pd.read_csv(path, chunksize=chunksize, low_memory=False)
    raise FileNotFoundError(f"MIMIC table not found: {name} (looked in {mimic_dir})")


def _ensure_schema(engine) -> None:
    ddl = """
    CREATE SCHEMA IF NOT EXISTS mimic;

    CREATE TABLE IF NOT EXISTS mimic.patients (
        subject_id BIGINT PRIMARY KEY,
        gender TEXT,
        anchor_age INT,
        anchor_year INT,
        dod TIMESTAMP NULL,
        ingested_at TIMESTAMP NOT NULL DEFAULT NOW(),
        source_system TEXT NOT NULL DEFAULT 'mimic-iv'
    );

    CREATE TABLE IF NOT EXISTS mimic.admissions (
        hadm_id BIGINT PRIMARY KEY,
        subject_id BIGINT NOT NULL REFERENCES mimic.patients(subject_id),
        admittime TIMESTAMP,
        dischtime TIMESTAMP,
        admission_type TEXT,
        ingested_at TIMESTAMP NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS mimic.labevents (
        lab_id BIGINT PRIMARY KEY,
        subject_id BIGINT NOT NULL,
        hadm_id BIGINT NULL,
        loinc TEXT,
        name TEXT,
        value DOUBLE PRECISION,
        unit TEXT,
        charttime TIMESTAMP,
        ingested_at TIMESTAMP NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS labevents_subject_time
        ON mimic.labevents (subject_id, charttime);

    CREATE TABLE IF NOT EXISTS mimic.vitals (
        vital_id BIGINT PRIMARY KEY,
        subject_id BIGINT NOT NULL,
        hadm_id BIGINT NULL,
        kind TEXT,
        value DOUBLE PRECISION,
        unit TEXT,
        charttime TIMESTAMP,
        ingested_at TIMESTAMP NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS mimic.prescriptions (
        rx_id TEXT PRIMARY KEY,
        subject_id BIGINT NOT NULL,
        hadm_id BIGINT NULL,
        rxnorm TEXT,
        drug TEXT,
        starttime TIMESTAMP,
        stoptime TIMESTAMP,
        dose TEXT,
        route TEXT,
        ingested_at TIMESTAMP NOT NULL DEFAULT NOW()
    );
    """
    with engine.begin() as conn:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            conn.execute(text(stmt))


def _upsert(engine, table: str, df: pd.DataFrame, pk: str) -> int:
    """Idempotent upsert via temp table + ON CONFLICT."""
    if df.empty:
        return 0
    tmp = f"_stg_{table.rsplit('.', maxsplit=1)[-1]}"
    df.to_sql(
        tmp, engine, schema=None, if_exists="replace", index=False, method="multi", chunksize=10_000
    )
    cols = ",".join(df.columns)
    excluded = ",".join(f"{c}=EXCLUDED.{c}" for c in df.columns if c != pk)
    sql = f"""
        INSERT INTO {table} ({cols})
        SELECT {cols} FROM {tmp}
        ON CONFLICT ({pk}) DO UPDATE SET {excluded};
        DROP TABLE {tmp};
    """
    with engine.begin() as conn:
        conn.execute(text(sql))
    return len(df)


def load_patients(engine, mimic_dir: Path, chunksize: int) -> int:
    total = 0
    for chunk in _read_table(mimic_dir, "patients", chunksize):
        chunk = chunk.rename(columns=str.lower)
        keep = ["subject_id", "gender", "anchor_age", "anchor_year", "dod"]
        chunk = chunk[[c for c in keep if c in chunk.columns]].copy()
        chunk["ingested_at"] = now_utc()
        chunk["source_system"] = "mimic-iv"
        total += _upsert(engine, "mimic.patients", chunk, "subject_id")
    log.info("patients: upserted %d rows", total)
    return total


def load_admissions(engine, mimic_dir: Path, chunksize: int) -> int:
    total = 0
    for chunk in _read_table(mimic_dir, "admissions", chunksize):
        chunk = chunk.rename(columns=str.lower)
        keep = ["hadm_id", "subject_id", "admittime", "dischtime", "admission_type"]
        chunk = chunk[[c for c in keep if c in chunk.columns]].copy()
        chunk["ingested_at"] = now_utc()
        total += _upsert(engine, "mimic.admissions", chunk, "hadm_id")
    log.info("admissions: upserted %d rows", total)
    return total


def load_labevents(engine, mimic_dir: Path, chunksize: int) -> int:
    total = 0
    # d_labitems maps itemid -> loinc when available
    try:
        d_items = next(iter([_read_table(mimic_dir, "d_labitems", 10**9)]))
        if hasattr(d_items, "__next__"):
            d_items = next(d_items)
        d_items.columns = [c.lower() for c in d_items.columns]
        itemid_to_loinc = (
            d_items.dropna(subset=["loinc_code"]).set_index("itemid")["loinc_code"].to_dict()
        )
        itemid_to_name = d_items.set_index("itemid")["label"].to_dict()
    except FileNotFoundError:
        itemid_to_loinc, itemid_to_name = {}, {}

    target_itemids = {iid for iid, loinc in itemid_to_loinc.items() if loinc in CARDIO_LOINC}

    for chunk in _read_table(mimic_dir, "labevents", chunksize):
        chunk = chunk.rename(columns=str.lower)
        if target_itemids:
            chunk = chunk[chunk["itemid"].isin(target_itemids)]
        if chunk.empty:
            continue
        chunk = chunk.assign(
            loinc=chunk["itemid"].map(itemid_to_loinc),
            name=chunk["itemid"].map(itemid_to_name),
        )
        keep = {
            "labevent_id": "lab_id",
            "subject_id": "subject_id",
            "hadm_id": "hadm_id",
            "loinc": "loinc",
            "name": "name",
            "valuenum": "value",
            "valueuom": "unit",
            "charttime": "charttime",
        }
        chunk = chunk.rename(columns=keep)[list(keep.values())]
        chunk["ingested_at"] = now_utc()
        total += _upsert(engine, "mimic.labevents", chunk, "lab_id")
    log.info("labevents (cardio subset): upserted %d rows", total)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=raw_path() / "mimic-iv")
    parser.add_argument("--chunk", type=int, default=50_000)
    args = parser.parse_args()

    if not args.mimic_dir.exists():
        log.warning(
            "MIMIC directory %s does not exist. "
            "Skipping clinical load — download the demo from "
            "https://physionet.org/content/mimic-iv-demo/2.2/ and unzip to that path.",
            args.mimic_dir,
        )
        return

    engine = create_engine(PostgresConfig.from_env().dsn, future=True)
    _ensure_schema(engine)
    load_patients(engine, args.mimic_dir, args.chunk)
    load_admissions(engine, args.mimic_dir, args.chunk)
    load_labevents(engine, args.mimic_dir, args.chunk)
    log.info("MIMIC load complete")


if __name__ == "__main__":
    main()
