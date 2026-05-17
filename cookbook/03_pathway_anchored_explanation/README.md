# Cookbook 03 — Pathway-anchored per-patient explanation

## The question (from a clinical safety review)

> "Your model says this patient is about to spike. Show me, with citations
> down to the data row, which lab values + behavioral events + biological
> pathway connections drove that. Don't show me a SHAP plot — show me an
> evidence trail I can put in front of a regulator."

This is the *audit story*. Tabular SHAP alone isn't enough; we need to walk
from the prediction back into the graph, name the pathways that are
implicated, and produce a one-page document the safety reviewer can sign.

## What this example does

For a single patient:

1. Pulls the patient's latest labs and engagement events directly from the
   graph (or local cache).
2. Pulls the patient's SHAP attribution row from the cached GBM analysis.
3. Maps each high-impact lab feature back to the Reactome pathway(s) that
   measure it via the `MEASURES` / `IN_PATHWAY` edges.
4. Renders a one-page markdown evidence trail:
   - Header: patient ID, prediction, observed last value, prediction delta.
   - "Evidence" section: for each top SHAP feature, the underlying data
     point and timestamp, plus the pathway citation (Reactome ID + link).
   - "Behavioral context" section: app engagement summary from the last
     30 days.
   - "Reviewer checklist": yes/no boxes a clinical reviewer signs off.

## How a real team would use this

* A clinical safety committee runs this for any prediction flagged
  high-risk in the dashboard.
* The output `evidence_<patient_id>.md` becomes the artifact of record —
  versioned in the same git repo as the model code, with the model
  version hash in the front-matter.

## Run

```bash
python -m cookbook.03_pathway_anchored_explanation.run --patient SYN000017
```
