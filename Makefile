.PHONY: help install lint format test test-all up down synth etl etl-parquet \
        train train-gnn evaluate dashboard pipeline pipeline-parquet clean

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  install            Install package with dev extras"
	@echo "  lint               Run ruff + black --check"
	@echo "  format             Run ruff --fix + black"
	@echo "  test               Run pytest (excluding slow/integration)"
	@echo "  test-all           Run full pytest suite"
	@echo "  up                 docker compose up the stack (Postgres + Neo4j + Jupyter)"
	@echo "  down               docker compose down"
	@echo "  synth              Generate synthetic engagement logs + labs"
	@echo "  etl                Postgres-backed ETL + Neo4j graph build (needs make up)"
	@echo "  etl-parquet        Docker-free ETL: MIMIC demo -> parquet (no DB needed)"
	@echo "  train              Train LightGBM models"
	@echo "  train-gnn          Train PyG GNN (needs torch + torch_geometric)"
	@echo "  evaluate           Aggregate trained-model results into docs/figures/"
	@echo "  evaluate-use-cases Field-standard metrics for every cookbook -> docs/EVALUATION.md"
	@echo "  dashboard          Launch Streamlit dashboard"
	@echo "  pipeline           Docker path: synth -> etl -> train -> evaluate"
	@echo "  pipeline-parquet   Docker-free path: synth -> etl-parquet -> train -> evaluate"
	@echo "  clean              Remove caches and build artefacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	black --check .

format:
	ruff check . --fix
	black .

test:
	pytest -m "not slow and not integration"

test-all:
	pytest

up:
	docker compose up -d
	@echo "Neo4j at http://localhost:7474 (neo4j/cmg-demo-pass)"
	@echo "Jupyter at http://localhost:8888 (token: cmg)"

down:
	docker compose down

synth:
	$(PYTHON) -m data.synthetic.generate_engagement_logs --out data/synthetic --patients 500 --days 180
	$(PYTHON) -m data.synthetic.generate_labs --out data/processed/labs.parquet

etl:
	$(PYTHON) -m etl.load_mimic
	$(PYTHON) -m etl.load_nhanes
	$(PYTHON) -m etl.load_pathways
	$(PYTHON) -m etl.build_graph

# Docker-free ETL path: parquet-only, no Postgres, no Neo4j required.
# Use this when you only have the MIMIC-IV demo and don't want to bring up
# the full data services. See README's "Docker-free quick-start" section.
etl-parquet:
	$(PYTHON) -m etl.load_mimic_parquet --merge-synthetic-labs

train:
	$(PYTHON) -m models.train --target hba1c --model gbm
	$(PYTHON) -m models.train --target engagement_dropout --model gbm

train-gnn:
	$(PYTHON) -m models.train --target hba1c --model gnn

evaluate:
	$(PYTHON) -m models.evaluate --report docs/figures/model_report.md

# Field-standard clinical-ML metrics for every cookbook example.
# Writes docs/EVALUATION.md + the figures under docs/figures/evaluation/.
evaluate-use-cases:
	$(PYTHON) -m models.delta_regressor
	$(PYTHON) scripts/evaluate_all_use_cases.py

dashboard:
	streamlit run dashboard/app.py

# Full pipeline assumes the Docker stack is up (`make up`). For the
# Docker-free path use `make pipeline-parquet` instead.
pipeline: synth etl train evaluate

pipeline-parquet: synth etl-parquet train evaluate

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
