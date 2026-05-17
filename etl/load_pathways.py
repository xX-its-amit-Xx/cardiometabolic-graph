"""Load Reactome + KEGG pathway data into Postgres staging.

Sources:
  * Reactome  (CC-BY) -- https://reactome.org/download-data
                         files: ReactomePathways.txt, NCBI2Reactome.txt,
                                ChEBI2Reactome.txt
  * KEGG      (free academic) -- https://rest.kegg.jp/list/pathway/hsa

We restrict to the three super-pathways most relevant to cardiometabolic
disease (configurable via --focus):

    R-HSA-74160   Signaling by insulin
    R-HSA-70171   Glycolysis
    R-HSA-556833  Metabolism of lipids

Plus all child pathways via Reactome's hierarchy if ``ReactomePathwaysRelation.txt``
is provided.

Usage:
    python -m etl.load_pathways --reactome-dir data/raw/reactome
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from ._common import PostgresConfig, log, raw_path

DEFAULT_FOCUS = ("R-HSA-74160", "R-HSA-70171", "R-HSA-556833")


def _ensure_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS pathways;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pathways.pathway (
                pathway_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                species TEXT
            );
            CREATE TABLE IF NOT EXISTS pathways.gene_pathway (
                gene_symbol TEXT NOT NULL,
                pathway_id TEXT NOT NULL REFERENCES pathways.pathway(pathway_id),
                PRIMARY KEY (gene_symbol, pathway_id)
            );
            CREATE TABLE IF NOT EXISTS pathways.metabolite_pathway (
                chebi_id TEXT NOT NULL,
                pathway_id TEXT NOT NULL REFERENCES pathways.pathway(pathway_id),
                PRIMARY KEY (chebi_id, pathway_id)
            );
            CREATE TABLE IF NOT EXISTS pathways.pathway_hierarchy (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                PRIMARY KEY (parent_id, child_id)
            );
        """))


def _expand_focus(focus: tuple[str, ...], hierarchy: pd.DataFrame | None) -> set[str]:
    if hierarchy is None:
        return set(focus)
    keep = set(focus)
    while True:
        children = hierarchy[hierarchy["parent_id"].isin(keep)]["child_id"]
        new = set(children) - keep
        if not new:
            return keep
        keep |= new


def load_reactome(engine, reactome_dir: Path, focus: tuple[str, ...]) -> None:
    pathways_path = reactome_dir / "ReactomePathways.txt"
    hier_path = reactome_dir / "ReactomePathwaysRelation.txt"
    g2p_path = reactome_dir / "NCBI2Reactome.txt"
    m2p_path = reactome_dir / "ChEBI2Reactome.txt"

    if not pathways_path.exists():
        log.warning("Reactome pathways file missing: %s", pathways_path)
        return

    pdf = pd.read_csv(pathways_path, sep="\t", header=None, names=["pathway_id", "name", "species"])
    pdf = pdf[pdf["species"] == "Homo sapiens"].copy()
    pdf["source"] = "reactome"

    hier = None
    if hier_path.exists():
        hier = pd.read_csv(hier_path, sep="\t", header=None, names=["parent_id", "child_id"])

    keep = _expand_focus(focus, hier)
    pdf = pdf[pdf["pathway_id"].isin(keep)]
    log.info("reactome pathways (after focus): %d", len(pdf))

    with engine.begin() as conn:
        pdf[["pathway_id", "name", "source", "species"]].to_sql(
            "_stg_pw", conn, if_exists="replace", index=False
        )
        conn.execute(text("""
            INSERT INTO pathways.pathway (pathway_id, name, source, species)
            SELECT pathway_id, name, source, species FROM _stg_pw
            ON CONFLICT (pathway_id) DO UPDATE SET
                name=EXCLUDED.name, source=EXCLUDED.source, species=EXCLUDED.species;
            DROP TABLE _stg_pw;
        """))

    if hier is not None:
        hier = hier[hier["parent_id"].isin(keep) & hier["child_id"].isin(keep)]
        with engine.begin() as conn:
            hier.to_sql("_stg_h", conn, if_exists="replace", index=False)
            conn.execute(text("""
                INSERT INTO pathways.pathway_hierarchy (parent_id, child_id)
                SELECT parent_id, child_id FROM _stg_h
                ON CONFLICT DO NOTHING;
                DROP TABLE _stg_h;
            """))

    if g2p_path.exists():
        g = pd.read_csv(
            g2p_path,
            sep="\t",
            header=None,
            names=["entrez_id", "pathway_id", "url", "name", "evidence", "species"],
        )
        g = g[(g["species"] == "Homo sapiens") & g["pathway_id"].isin(keep)]
        g["gene_symbol"] = g["entrez_id"].astype(str)
        with engine.begin() as conn:
            g[["gene_symbol", "pathway_id"]].drop_duplicates().to_sql(
                "_stg_gp", conn, if_exists="replace", index=False
            )
            conn.execute(text("""
                INSERT INTO pathways.gene_pathway (gene_symbol, pathway_id)
                SELECT gene_symbol, pathway_id FROM _stg_gp
                ON CONFLICT DO NOTHING;
                DROP TABLE _stg_gp;
            """))

    if m2p_path.exists():
        m = pd.read_csv(
            m2p_path,
            sep="\t",
            header=None,
            names=["chebi_id", "pathway_id", "url", "name", "evidence", "species"],
        )
        m = m[(m["species"] == "Homo sapiens") & m["pathway_id"].isin(keep)]
        m["chebi_id"] = m["chebi_id"].astype(str)
        with engine.begin() as conn:
            m[["chebi_id", "pathway_id"]].drop_duplicates().to_sql(
                "_stg_mp", conn, if_exists="replace", index=False
            )
            conn.execute(text("""
                INSERT INTO pathways.metabolite_pathway (chebi_id, pathway_id)
                SELECT chebi_id, pathway_id FROM _stg_mp
                ON CONFLICT DO NOTHING;
                DROP TABLE _stg_mp;
            """))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactome-dir", type=Path, default=raw_path() / "reactome")
    parser.add_argument("--focus", nargs="*", default=list(DEFAULT_FOCUS))
    args = parser.parse_args()

    engine = create_engine(PostgresConfig.from_env().dsn, future=True)
    _ensure_schema(engine)

    if args.reactome_dir.exists():
        load_reactome(engine, args.reactome_dir, tuple(args.focus))
    else:
        log.warning(
            "Reactome dir %s missing; download from https://reactome.org/download-data",
            args.reactome_dir,
        )


if __name__ == "__main__":
    main()
