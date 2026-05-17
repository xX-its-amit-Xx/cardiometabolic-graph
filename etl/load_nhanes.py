"""Load NHANES behavioral / lifestyle data into Postgres staging.

NHANES is freely downloadable XPT (SAS Transport) files from the CDC:
    https://wwwn.cdc.gov/Nchs/Nhanes/

We focus on the 2017-2018 cycle for stable variable codes. Tables loaded:

  - DEMO_J     demographics (age, sex, race, education)
  - PAQ_J      physical activity (minutes per category)
  - DR1TOT_J   day-1 dietary recall (energy, macronutrients)
  - SLQ_J      sleep questionnaire
  - SMQ_J      smoking status

Each NHANES respondent has a unique SEQN that we use as patient_id with the
``NHANES`` prefix to avoid collision with MIMIC subject_ids.

Usage:
    python -m etl.load_nhanes --nhanes-dir data/raw/nhanes-2017-18
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from ._common import PostgresConfig, log, now_utc, raw_path

TABLES = {
    "DEMO_J":  {"vars": ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2"], "domain": "demographics"},
    "PAQ_J":   {"vars": ["SEQN", "PAQ605", "PAQ620", "PAQ650", "PAQ665"], "domain": "activity"},
    "DR1TOT_J":{"vars": ["SEQN", "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TFIBE", "DR1TTFAT"], "domain": "diet"},
    "SLQ_J":   {"vars": ["SEQN", "SLD012", "SLQ050"], "domain": "sleep"},
    "SMQ_J":   {"vars": ["SEQN", "SMQ020", "SMQ040"], "domain": "smoking"},
}


def _ensure_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS nhanes;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS nhanes.respondents (
                seqn BIGINT PRIMARY KEY,
                gender INT,
                age_years INT,
                race_eth INT,
                education INT,
                ingested_at TIMESTAMP NOT NULL DEFAULT NOW(),
                source_system TEXT NOT NULL DEFAULT 'nhanes-2017-18'
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS nhanes.events (
                event_id BIGSERIAL PRIMARY KEY,
                seqn BIGINT NOT NULL REFERENCES nhanes.respondents(seqn) ON DELETE CASCADE,
                domain TEXT NOT NULL,
                variable TEXT NOT NULL,
                value DOUBLE PRECISION,
                ingested_at TIMESTAMP NOT NULL DEFAULT NOW(),
                UNIQUE (seqn, domain, variable)
            );
        """))


def _read_xpt(path: Path) -> pd.DataFrame:
    # pandas reads SAS XPT natively
    return pd.read_sas(path, format="xport", encoding="utf-8")


def _upsert_respondents(engine, demo: pd.DataFrame) -> int:
    rename = {
        "SEQN": "seqn",
        "RIAGENDR": "gender",
        "RIDAGEYR": "age_years",
        "RIDRETH3": "race_eth",
        "DMDEDUC2": "education",
    }
    df = demo.rename(columns=rename)[list(rename.values())].copy()
    df = df.dropna(subset=["seqn"]).astype({"seqn": "int64"})
    df["ingested_at"] = now_utc()

    df.to_sql("_stg_demo", engine, if_exists="replace", index=False, method="multi", chunksize=5_000)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO nhanes.respondents (seqn, gender, age_years, race_eth, education, ingested_at)
            SELECT seqn, gender::INT, age_years::INT, race_eth::INT, education::INT, ingested_at
            FROM _stg_demo
            ON CONFLICT (seqn) DO UPDATE SET
              gender=EXCLUDED.gender,
              age_years=EXCLUDED.age_years,
              race_eth=EXCLUDED.race_eth,
              education=EXCLUDED.education,
              ingested_at=EXCLUDED.ingested_at;
            DROP TABLE _stg_demo;
        """))
    return len(df)


def _upsert_events(engine, seqns: pd.Series, domain: str, df: pd.DataFrame) -> int:
    long_df = df.melt(id_vars=["SEQN"], var_name="variable", value_name="value").dropna(subset=["value"])
    long_df = long_df.rename(columns={"SEQN": "seqn"})
    long_df["seqn"] = long_df["seqn"].astype("int64")
    long_df = long_df[long_df["seqn"].isin(seqns)]
    long_df["domain"] = domain

    if long_df.empty:
        return 0
    long_df.to_sql("_stg_events", engine, if_exists="replace", index=False, method="multi", chunksize=10_000)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO nhanes.events (seqn, domain, variable, value)
            SELECT seqn, domain, variable, value FROM _stg_events
            ON CONFLICT (seqn, domain, variable) DO UPDATE SET value=EXCLUDED.value;
            DROP TABLE _stg_events;
        """))
    return len(long_df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nhanes-dir", type=Path, default=raw_path() / "nhanes-2017-18")
    args = parser.parse_args()

    if not args.nhanes_dir.exists():
        log.warning(
            "NHANES directory %s does not exist. Download XPT files from "
            "https://wwwn.cdc.gov/Nchs/Nhanes/continuousnhanes/default.aspx?BeginYear=2017",
            args.nhanes_dir,
        )
        return

    engine = create_engine(PostgresConfig.from_env().dsn, future=True)
    _ensure_schema(engine)

    demo_path = args.nhanes_dir / "DEMO_J.XPT"
    if not demo_path.exists():
        log.error("DEMO_J.XPT missing; demographics table is required first.")
        return
    demo_df = _read_xpt(demo_path)
    n_resp = _upsert_respondents(engine, demo_df)
    log.info("respondents: %d", n_resp)

    seqns = demo_df["SEQN"].astype("int64")
    for tbl, spec in TABLES.items():
        if tbl == "DEMO_J":
            continue
        path = args.nhanes_dir / f"{tbl}.XPT"
        if not path.exists():
            log.warning("missing %s", path.name)
            continue
        df = _read_xpt(path)
        keep = [c for c in spec["vars"] if c in df.columns]
        n = _upsert_events(engine, seqns, spec["domain"], df[keep])
        log.info("%s (%s): %d events", tbl, spec["domain"], n)


if __name__ == "__main__":
    main()
