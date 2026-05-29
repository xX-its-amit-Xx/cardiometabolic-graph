"""Shared helpers for ETL loaders.

All loaders are designed to be:
  * idempotent (re-running produces no duplicates),
  * chunked (work on the full MIMIC-IV or just the demo),
  * traceable (every row carries source_system + ingested_at).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.getenv("CMG_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("cmg.etl")


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    db: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> PostgresConfig:
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            db=os.getenv("POSTGRES_DB", "cmg"),
            user=os.getenv("POSTGRES_USER", "cmg"),
            password=os.getenv("POSTGRES_PASSWORD", "cmg-demo-pass"),
        )

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> Neo4jConfig:
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "cmg-demo-pass"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


@contextmanager
def neo4j_session() -> Iterator:
    """Yield a Neo4j session, importing the driver lazily so unit tests
    that don't touch the DB don't need it installed at import time."""
    from neo4j import GraphDatabase

    cfg = Neo4jConfig.from_env()
    driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
    try:
        with driver.session(database=cfg.database) as session:
            yield session
    finally:
        driver.close()


def now_utc() -> datetime:
    return datetime.now(UTC)


def chunked(iterable, size: int):
    """Yield successive chunks of size ``size`` from ``iterable``."""
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def raw_path() -> Path:
    return Path(os.getenv("CMG_DATA_RAW", "data/raw")).resolve()


def processed_path() -> Path:
    return Path(os.getenv("CMG_DATA_PROCESSED", "data/processed")).resolve()


def synthetic_path() -> Path:
    return Path(os.getenv("CMG_DATA_SYNTHETIC", "data/synthetic")).resolve()
