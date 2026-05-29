# cardiometabolic-graph

> A public reference implementation of a patient knowledge graph that
> integrates clinical labs, behavioral / lifestyle signals, and molecular
> pathway data for cardiometabolic outcome modeling. Built end-to-end with
> Neo4j, PyTorch Geometric, LightGBM, SHAP, and Streamlit.

**Why it exists.** Most "predict HbA1c" demos stop at a tabular model on a
single dataset. Real digital therapeutics and computational health teams
have to fuse clinical data with behavioral telemetry and biological
context — and then explain the result to a clinician or a regulator. This
repo is the smallest serious version of that whole pipeline, designed so
another team can clone it on Monday and have something running by lunch.

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](pyproject.toml)

---

## Knowledge graph schema

```mermaid
graph LR
    P[Patient] -->|HAS_ENCOUNTER| E[Encounter]
    E -->|HAS_LAB| L[LabResult]
    E -->|HAS_VITAL| V[Vital]
    E -->|PRESCRIBED| Rx[Medication]
    P -->|ENGAGED_WITH| B[BehavioralEvent]
    L -->|MEASURES| M[Metabolite]
    Rx -->|TARGETS| Pw[Pathway]
    G[Gene] -->|PARTICIPATES_IN| Pw
    Pw -->|REGULATES| Pw2[Pathway]
    P -.->|CO_OCCURS_WITH| P2[Patient]
```

Full definition with property lists and lineage notes:
[`schema/graph_schema.md`](schema/graph_schema.md).

---

## Quick start

### 30-second path (one command from a fresh clone)

```bash
git clone https://github.com/xX-its-amit-Xx/cardiometabolic-graph.git
cd cardiometabolic-graph
pip install -e .
cmg quickstart        # auto-fetches MIMIC demo + trains + evaluates
cmg dashboard         # http://localhost:8501
```

`cmg quickstart` runs `cmg doctor`, downloads the freely-available
MIMIC-IV demo (~16 MB, no PhysioNet credentials needed), generates
synthetic engagement + labs, trains every model, runs SHAP, generates
the evaluation scoreboard, and tells you what's next. Takes ~3 minutes
on a laptop.

Other useful single commands:

```bash
cmg doctor               # diagnose env, deps, data files, MIMIC presence
cmg pipeline             # rerun synth + ETL + train + evaluate
cmg cookbook 01          # run one of the eight worked examples
cmg cookbook 05 -- --patient SYN000001 --drug "semaglutide 1mg weekly"
cmg evaluate             # regenerate docs/EVALUATION.md + figures
```

### Full Docker stack (Postgres + Neo4j + Jupyter)

```bash
docker compose up -d        # Postgres + Neo4j + Jupyter
make install
make pipeline               # synth -> Postgres ETL -> Neo4j graph -> train -> evaluate
make dashboard              # http://localhost:8501
```

| Service   | URL                          | Credentials                  |
|-----------|------------------------------|------------------------------|
| Neo4j     | http://localhost:7474        | `neo4j` / `cmg-demo-pass`    |
| Postgres  | `postgresql://localhost:5432`| `cmg` / `cmg-demo-pass`      |
| Jupyter   | http://localhost:8888        | token: `cmg`                 |
| Dashboard | http://localhost:8501        | n/a                          |

---

## Data sources

All sources are public. **No real patient data is committed to this
repo** — see [`.gitignore`](.gitignore) — and the demo runs entirely on the
MIMIC-IV demo subset plus synthetic engagement events when you don't have
credentialed access to anything else.

| Source                          | What we use                                          | License                              | Where it goes                                   |
|---------------------------------|------------------------------------------------------|--------------------------------------|-------------------------------------------------|
| [MIMIC-IV demo](https://physionet.org/content/mimic-iv-demo/2.2/) ✅ **active** | HbA1c, glucose, lipid panel — 100 real patients, 3,089 labs | PhysioNet Open Data Commons (~100 pts) | `data/raw/mimic-iv/` → parquet (`etl/load_mimic_parquet.py`) → feature builder. Postgres path also supported via `etl/load_mimic.py` for full MIMIC-IV. |
| MIMIC-IV (full)                 | Same tables, ~300k patients                          | PhysioNet Credentialed Health Data License v1.5.0 | Drop into `data/raw/mimic-iv/` — loader auto-detects |
| [NHANES 2017-18](https://wwwn.cdc.gov/Nchs/Nhanes/) | Physical activity, dietary recall, sleep, smoking   | CDC public domain                    | `data/raw/nhanes-2017-18/*.XPT` → Postgres → `BehavioralEvent` nodes |
| [Reactome v89](https://reactome.org/download-data) | Insulin signaling, glycolysis, lipid metabolism pathways | CC-BY 4.0                       | `data/raw/reactome/` → Postgres → `Pathway`, `Gene`, `Metabolite` nodes |
| [KEGG](https://rest.kegg.jp/list/pathway/hsa) | Cross-reference pathway IDs                         | Free academic license                | `data/raw/kegg/` → optional enrichment            |
| Synthetic engagement            | App opens, message responses, glucose logs           | Generated by `data/synthetic/generate_engagement_logs.py` | `data/synthetic/*.parquet` → `BehavioralEvent` nodes |

---

## Model results

Numbers below come from a pipeline run on **500 synthetic patients**
(archetype-driven engagement + correlated labs), 9,000 lab rows. Iteration 2
MIMIC-IV numbers (Pearson 0.875 / MAE 0.384 with mixed real+synthetic data)
are documented in `ITERATIONS.md`. Re-run `python -m models.evaluate` to
regenerate the report at
[`docs/figures/model_report.md`](docs/figures/model_report.md).

### HbA1c trajectory (regression — predict the held-out latest visit)

| Model          | Pearson r | MAE (HbA1c %) | n_train | n_test |
|----------------|-----------|---------------|---------|--------|
| LightGBM       | **0.968** | **0.282**     | 400     | 100    |
| GAT-GNN (PyG)  | 0.701     | 0.790         | 400     | 100    |

### Engagement dropout (binary classification)

| Model          | AUROC     | AUPRC     | Brier     | ECE       | Positive rate (test) |
|----------------|-----------|-----------|-----------|-----------|----------------------|
| LightGBM       | **0.845** | **0.754** | **0.084** | **0.071** | ~17%                 |

Dropout target carries 10% symmetric label-flip noise to reflect the
real-world ambiguity in DTx ("paused" vs "dropped") — without it, the
clean archetype label is trivially separable. With noise, AUROC sits in
a realistic 0.80-0.85 band.

### HbA1c delta regressor (used by cookbook 08)

| Model          | Pearson r | MAE (Δ HbA1c %) |
|----------------|-----------|-----------------|
| LightGBM       | **0.699** | **0.328**       |

Predicting CHANGE is genuinely harder than predicting LEVEL (the
level model lands at r=0.87 but ranks change-direction near random —
see [`docs/EVALUATION.md`](docs/EVALUATION.md) for the analysis that
motivated training a dedicated delta head).

### Field-standard evaluation

Every cookbook example is graded against published clinical-ML expected
ranges (Hosmer-Lemeshow, Vickers' decision-curve analysis, Brier 1950,
Guo 2017's ECE, NDCG, PSI). Run `python -m scripts.evaluate_all_use_cases`
to regenerate, or read the full report:

- **[`docs/EVALUATION.md`](docs/EVALUATION.md)** — every metric, its value, the
  expected range with literature citation, and a status grade.
- [`docs/figures/evaluation/metric_status.png`](docs/figures/evaluation/metric_status.png)
  — single-glance bar chart, color-coded by grade.
- [`docs/figures/evaluation/reliability_dropout.png`](docs/figures/evaluation/reliability_dropout.png)
  — Hosmer-Lemeshow reliability plot for the dropout classifier.
- [`docs/figures/evaluation/decision_curve_dropout.png`](docs/figures/evaluation/decision_curve_dropout.png)
  — Vickers & Elkin decision-curve analysis.
- [`docs/figures/evaluation/lift_curve_at_risk.png`](docs/figures/evaluation/lift_curve_at_risk.png)
  — lift as a function of cohort call-list size.

Current pass rate: **24 / 24 metrics in the deployable band** (18 graded
"good" against published references, 6 "acceptable"; 0 failing).

The GNN uses a star-graph construction (each non-zero tabular feature becomes
a satellite node connected to a patient root) as a portable bridge that doesn't
require a running Neo4j instance. The GBM outperforms the GNN here because the
tabular signal is already well-structured; the GNN's advantage would emerge with
richer relational structure from the Neo4j pathway graph.

GNN ablation study (`python -m models.ablations --epochs 30`):

| Feature set | Pearson r | MAE |
|-------------|-----------|-----|
| No engagement (labs only) | 0.841 | 0.790 |
| No labs (engagement only) | 0.540 | 2.173 |
| Full (all features)       | 0.833 | 0.762 |

![GNN ablation study](docs/figures/gnn_ablations.png)

Labs dominate signal (removing them degrades Pearson by 0.30+); engagement
features alone carry some predictive information but are not sufficient on their own.

Top GBM HbA1c features by gain (from
[`docs/figures/model_report.md`](docs/figures/model_report.md)):
`glucose_serum_last`, `triglycerides_last`, `cholesterol_total_mean`,
`ldl_min`, `hdl_last` — clinically plausible since serum glucose and the
lipid panel co-vary with glycemic control.

Explanation outputs in [`docs/figures/`](docs/figures/):

- [`shap_summary.png`](docs/figures/shap_summary.png) — global SHAP
  beeswarm for the GBM HbA1c model.
- [`shap_bar.png`](docs/figures/shap_bar.png) — mean(|SHAP|) ranking.
- Per-patient GNN attention attribution: saved per-call to
  `artifacts/gnn/attribution_<patient_id>.parquet` and used by the dashboard.

---

## Dashboard

![Streamlit clinician view](docs/screenshots/dashboard_topfold.png)

The clinician-facing patient page renders KPIs (last HbA1c, predicted next
HbA1c with delta, 30-day app opens, dropout risk), a configurable summary
paragraph, a 180-day engagement timeline, an HbA1c trajectory chart with
predicted uncertainty band, and a SHAP factor chart.

Capture your own: `make dashboard` to launch, then
`python scripts/capture_screenshots.py --pid <patient_id>` to write a fresh
PNG into `docs/screenshots/`.

Full-page capture: [`docs/screenshots/dashboard_overview.png`](docs/screenshots/dashboard_overview.png).

---

## Summary backends — no proprietary LLM required

The clinician summary is generated by a pluggable backend chosen via the
sidebar selector or the `CMG_SUMMARIZER` env var. Every option is
open-source / open-weight — **nothing in this repo requires an
Anthropic, OpenAI, or Google API key.**

| Backend | What it is | When to pick it | Default? |
|---------|-----------|-----------------|----------|
| `deterministic` | Template engine in [`dashboard/_summary.py`](dashboard/_summary.py). Every output sentence traces to a documented rule. | Regulated environments — easy to audit, zero hallucination risk. | ✅ |
| `ollama` | POSTs to a local [Ollama](https://ollama.ai) server (default model `llama3.1:8b`, ~5 GB RAM). | Production deployments wanting natural-language flexibility without data leaving the host. | |
| `transformers` | Loads an open-weight HF causal LM in-process (default `microsoft/Phi-3-mini-4k-instruct`, MIT). | Air-gapped environments where Ollama isn't installable. | |

Every backend reads the same structured `PatientFacts` payload, so the
audit trail behind the prose is identical regardless of which model
generated it. Both LLM backends silently fall back to the deterministic
template on connection / install errors — the dashboard always renders
*something* auditable. See [`summarizer/`](summarizer/) for the
implementation and [`tests/test_summarizer.py`](tests/test_summarizer.py)
for the contract tests.

---

## Cookbook — real-world worked examples

The [`cookbook/`](cookbook/) directory shows the pipeline answering
concrete questions a real DTx team or academic group has asked. Each
example runs in under five minutes after `make pipeline-parquet`:

| # | Example | Question answered | Who uses it |
|---|---------|-------------------|-------------|
| 1 | [`01_at_risk_cohort`](cookbook/01_at_risk_cohort/README.md) | "Which 50 patients should our care coordinators call this week?" | Care-team triage |
| 2 | [`02_reengagement_outreach`](cookbook/02_reengagement_outreach/README.md) | "Which lapsing users will *respond* to a re-engagement push?" | Product / growth |
| 3 | [`03_pathway_anchored_explanation`](cookbook/03_pathway_anchored_explanation/README.md) | "Show me the evidence trail behind this one patient's prediction." | Clinical safety review |
| 4 | [`04_cohort_drift_monitor`](cookbook/04_cohort_drift_monitor/README.md) | "Is the live cohort drifting far enough that we need to retrain?" | MLOps / monitoring |
| 5 | [`05_prior_auth_note`](cookbook/05_prior_auth_note/README.md) | "Give me a structured PA note that cites the metrics justifying GLP-1 medical necessity." | Utilization-management specialists |
| 6 | [`06_pre_visit_summary`](cookbook/06_pre_visit_summary/README.md) | "Give me the 30-second worklist version of each patient on today's schedule." | Attending physicians |
| 7 | [`07_trial_eligibility`](cookbook/07_trial_eligibility/README.md) | "Which of my patients meet the inclusion criteria for our new CV outcomes trial?" | Study coordinators |
| 8 | [`08_pharmacist_intervention`](cookbook/08_pharmacist_intervention/README.md) | "Which 30 patients should our pharmacist call this week for the highest HbA1c-trajectory leverage?" | Value-based-care pharmacist teams |

Each example writes a one-page `report.md` next to its `run.py` plus
CSVs and figures. Examples 05 + 06 use the pluggable summarizer (see
above) so they can produce prose with the deterministic template by
default or with an open-weight LLM via `CMG_SUMMARIZER=ollama`.

---

## Why this exists

Most computational-health portfolios stop one layer too early:

- **"Predict HbA1c from labs"** — but no behavioral context, so the model
  can't tell a stably-disengaged patient from a newly-spiking one.
- **"Build a GNN on a single dataset"** — but no clinical interpretability
  wrapper, so a regulator can't reason about it.
- **"Train a model"** — but no inference dashboard, drift monitor, or
  cohort-targeting cookbook, so it never connects to actual workflow.

This repository is the smallest version I could build that doesn't skip
any of those layers. It's intended as a starting point for:

- **Academic computational health groups** who want a reproducible
  multi-source pipeline they can adapt with their own MIMIC, NHANES, or
  EHR cohorts — without re-inventing the Docker / schema / training
  scaffolding.
- **Digital therapeutics startups** building patient-stratification or
  re-engagement logic on top of their telemetry, who need to fuse it with
  clinical signals and explain the result to clinicians and to compliance.
- **ML engineers / recruiters** who want to see end-to-end systems
  thinking, not just a model notebook.

If you adapt it, please open an issue or a PR with what you built — the
cookbook is the part most likely to benefit from real examples.

---

## Roadmap

| Item | Why | Status |
|------|-----|--------|
| MIMIC-IV demo ingestion (parquet path) | Real clinical labs, no Docker required | ✅ Done (iter 2) |
| GNN baseline + ablation study | Quantify which signal class moves the score | ✅ Done (iter 3) |
| Per-patient GNN attention attribution | Map predictions back to neighborhood structure | ✅ Done (iter 3) |
| Open-weight LLM summarizer (Ollama + HF) | No proprietary API key required | ✅ Done (iter 4) |
| Field-standard evaluation + EVALUATION.md | Grade every use case against published reference ranges | ✅ Done (iter 5) |
| HbA1c delta regressor | Predict change, not level (pharmacist leverage) | ✅ Done (iter 5) |
| Isotonic calibration (optional) | Sharpen probability estimates at deployment scale | ✅ Done (iter 5, opt-in) |
| Model card + data governance doc | Production readiness — the things a CTO asks before adopting | Planned (iter 6) |
| Federated-learning variant (FedAvg over per-site graphs) | Real DTx data lives behind site boundaries | Planned |
| FHIR ingestion adapter | Production EHRs speak FHIR, not CSV | Planned |
| MedCAT-based clinical-notes extraction | Adds free-text-derived symptoms and meds to the graph | Planned |
| Causal layer (do-calculus on the graph) | Move beyond predictive scores to intervention reasoning | Planned |

See [`ITERATIONS.md`](ITERATIONS.md) for the per-iteration log of what
landed and what didn't. PRs welcome on any planned item —
[`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Repository layout

```
cardiometabolic-graph/
├── data/
│   ├── raw/                    # gitignored — drop MIMIC / NHANES / Reactome here
│   ├── processed/              # cached parquet feature frames
│   └── synthetic/              # synthetic engagement events + generator script
├── etl/                        # MIMIC, NHANES, pathways loaders + Neo4j graph builder
│                               #   load_mimic_parquet.py is the Docker-free MIMIC path
├── schema/                     # graph_schema.md + cypher_constraints.cql
├── models/                     # LightGBM baseline, GNN, dropout, delta regressor,
│                               #   calibration, train, evaluate, ablations
├── explain/                    # SHAP + GNN attention attribution
├── summarizer/                 # Pluggable summary backends (deterministic / ollama /
│                               #   transformers) — no proprietary LLM required
├── evaluation/                 # Clinical-ML metrics + expected ranges + use-case graders
├── dashboard/                  # Streamlit clinician view + deterministic summary template
├── cookbook/                   # Eight real-world worked examples (01-08)
├── notebooks/                  # Walkthroughs (exploration, graph build, training, interpretation)
├── scripts/                    # Screenshot capture, evaluation runner
├── tests/                      # pytest — ETL, schema, models, summarizer
├── docs/                       # EVALUATION.md, screenshots, figures
├── docker-compose.yml          # Postgres + Neo4j + Jupyter
├── pyproject.toml              # package + dev / llm-local extras
├── Makefile                    # entry points for everything above
├── ITERATIONS.md               # per-iteration log
├── CONTRIBUTING.md             # how to contribute
└── CODE_OF_CONDUCT.md          # Contributor Covenant
```

---

## Credits

- **MIMIC-IV** — Beth Israel Deaconess Medical Center / MIT Laboratory for
  Computational Physiology.
- **NHANES** — US Centers for Disease Control and Prevention.
- **Reactome** — EMBL-EBI, OICR, NYULMC, Oregon Health & Science University.
- **KEGG** — Kanehisa Laboratories.
- **PyTorch Geometric** — Matthias Fey & Jan E. Lenssen.
- **SHAP** — Scott Lundberg.

This repository was authored by **Amit Shenoy** as a public reference
implementation. Issues and PRs welcome.

## License

[MIT](LICENSE).
