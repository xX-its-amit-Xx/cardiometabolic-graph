# Cookbook

Real-world worked examples on top of the cardiometabolic graph. Each
example is a self-contained Python module under `cookbook/<NN_name>/` with:

- `README.md` — the clinical or product question, the data flow, expected
  outputs, and a short *"how a team would use this"* section.
- `run.py` — a single `python -m cookbook.<NN_name>.run` entry-point.
- `figures/` — any plots the example produces.

All examples are designed to **run end-to-end in under 5 minutes** on a
laptop after `make pipeline`, and **none of them require real patient
data** — they use the synthetic engagement stream and the MIMIC-IV demo
subset (or a fully synthetic fallback when MIMIC isn't installed).

## Index

| # | Example | What real-world question it answers | Used by |
|---|---------|--------------------------------------|---------|
| 1 | [`01_at_risk_cohort`](01_at_risk_cohort/README.md) | "Which 50 patients should our care coordinators call this week?" | DTx care-team triage |
| 2 | [`02_reengagement_outreach`](02_reengagement_outreach/README.md) | "Which lapsing users will respond to a re-engagement push?" | Product / growth ops |
| 3 | [`03_pathway_anchored_explanation`](03_pathway_anchored_explanation/README.md) | "Why does the model think this specific patient is rising?" | Clinical safety review |
| 4 | [`04_cohort_drift_monitor`](04_cohort_drift_monitor/README.md) | "Is our intake population this month different from last quarter?" | MLOps / monitoring |
| 5 | [`05_prior_auth_note`](05_prior_auth_note/README.md) | "Give me a structured PA note that cites the metrics justifying GLP-1 medical necessity." | PA / utilization-management specialists |
| 6 | [`06_pre_visit_summary`](06_pre_visit_summary/README.md) | "Give me the 30-second worklist version of each patient on today's schedule." | Attending physicians / front-desk MA |
| 7 | [`07_trial_eligibility`](07_trial_eligibility/README.md) | "Which patients meet the inclusion criteria for our new CV outcomes trial?" | Study coordinators / clinical research ops |
| 8 | [`08_pharmacist_intervention`](08_pharmacist_intervention/README.md) | "Which 30 patients should our pharmacist call this week for the highest HbA1c-trajectory leverage?" | Value-based-care pharmacist teams |

## How to run all of them

```bash
make pipeline-parquet                # one-time setup (Docker-free)
python -m cookbook.01_at_risk_cohort.run
python -m cookbook.02_reengagement_outreach.run
python -m cookbook.03_pathway_anchored_explanation.run --patient SYN000017
python -m cookbook.04_cohort_drift_monitor.run --window-days 30
python -m cookbook.05_prior_auth_note.run --patient SYN000001 --drug "semaglutide 1mg weekly"
python -m cookbook.06_pre_visit_summary.run --top 5
python -m cookbook.07_trial_eligibility.run --top 20
python -m cookbook.08_pharmacist_intervention.run --top 30
```

Outputs land under each example's `figures/` directory and a one-page
markdown report under `cookbook/<example>/report.md`.

### Switching the summarizer backend

Examples 05 and 06 produce natural-language prose. By default they use
the deterministic template engine (auditable, no LLM). To swap in an
open-weight model:

```bash
# Ollama, with the host running `ollama serve` and `ollama pull llama3.1:8b`
CMG_SUMMARIZER=ollama python -m cookbook.05_prior_auth_note.run --patient SYN000001

# In-process HuggingFace transformers (Phi-3 Mini by default)
CMG_SUMMARIZER=transformers python -m cookbook.06_pre_visit_summary.run --top 5
```

Every backend reads the same structured `PatientFacts` payload, so the
audit trail behind the prose is identical regardless of which model
generated it.

## Why a cookbook

A README explains *what* the repo does. A cookbook proves *what you can
actually do with it*. Each example here corresponds to a concrete request a
real DTx team or a real academic group has actually wanted to answer.
