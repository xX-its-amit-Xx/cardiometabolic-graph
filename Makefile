.PHONY: help install lint format test pipeline synth etl train evaluate dashboard up down clean

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  install      Install package with dev extras"
	@echo "  lint         Run ruff + black --check"
	@echo "  format       Run ruff --fix + black"
	@echo "  test         Run pytest (excluding slow/integration)"
	@echo "  test-all     Run full pytest suite"
	@echo "  up           docker compose up the stack"
	@echo "  down         docker compose down"
	@echo "  synth        Generate synthetic engagement logs"
	@echo "  etl          Run all ETL loaders + build graph"
	@echo "  train        Train all models"
	@echo "  evaluate     Evaluate trained models"
	@echo "  dashboard    Launch Streamlit dashboard"
	@echo "  pipeline     End-to-end: synth -> etl -> train -> evaluate"
	@echo "  clean        Remove caches and build artefacts"

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

etl:
	$(PYTHON) -m etl.load_mimic
	$(PYTHON) -m etl.load_nhanes
	$(PYTHON) -m etl.load_pathways
	$(PYTHON) -m etl.build_graph

train:
	$(PYTHON) -m models.train --target hba1c --model gnn
	$(PYTHON) -m models.train --target hba1c --model gbm
	$(PYTHON) -m models.train --target engagement_dropout --model gbm

evaluate:
	$(PYTHON) -m models.evaluate --report docs/figures/model_report.md

dashboard:
	streamlit run dashboard/app.py

pipeline: synth etl train evaluate

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
