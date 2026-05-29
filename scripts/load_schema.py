"""Load Neo4j schema constraints and indexes from cypher_constraints.cql."""

import sys
from pathlib import Path

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "cmg-demo-pass"
SCHEMA_FILE = Path(__file__).resolve().parent.parent / "schema" / "cypher_constraints.cql"


def parse_statements(text: str) -> list[str]:
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("//")]
    stripped = "\n".join(lines)
    parts = [s.strip() for s in stripped.split(";") if s.strip()]
    return parts


def main() -> int:
    text = SCHEMA_FILE.read_text(encoding="utf-8-sig")
    statements = parse_statements(text)
    print(f"Found {len(statements)} statements in {SCHEMA_FILE.name}")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        with driver.session() as session:
            for stmt in statements:
                first_line = stmt.split("\n", 1)[0]
                try:
                    session.run(stmt).consume()
                    print(f"  OK   {first_line[:80]}")
                except Exception as e:
                    print(f"  FAIL {first_line[:80]} -> {e}")

            result = session.run("SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties")
            print("\nActive constraints:")
            for record in result:
                print(
                    f"  {record['name']:30s} {record['type']:25s} {record['labelsOrTypes']} {record['properties']}"
                )

            result = session.run(
                "SHOW INDEXES YIELD name, type, labelsOrTypes, properties WHERE type <> 'LOOKUP'"
            )
            print("\nActive indexes (non-LOOKUP):")
            for record in result:
                print(
                    f"  {record['name']:30s} {record['type']:15s} {record['labelsOrTypes']} {record['properties']}"
                )
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
