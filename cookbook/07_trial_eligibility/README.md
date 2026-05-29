# Cookbook 07 — Clinical-trial eligibility screening

## The question (from a study coordinator at an academic medical center)

> "Our new T2D trial needs HbA1c 7.5–10.5, age 30–70, no recent statin
> change, and at least moderate digital-health engagement. Screening
> from the EHR manually takes a week. Can you give us a ranked candidate
> list in 30 seconds, with a per-patient explanation of which criteria
> they meet?"

Eligibility screening is **the** classic graph-query use case — every
inclusion/exclusion criterion maps to a path through the patient
knowledge graph, and a single Cypher (or pandas) query produces a
ranked candidate list.

## What this example does

1. Loads the patient feature frame + lab history.
2. Applies a configurable JSON trial spec (default: a fictional
   semaglutide cardiovascular outcomes trial inspired by SUSTAIN-6).
3. For each patient, evaluates every criterion and records a
   per-criterion pass/fail.
4. Ranks patients by:
   - First: number of criteria met (descending),
   - Then: engagement intensity (descending) — trials prefer adherent
     patients.
5. Writes `eligible_patients.csv` and a `report.md` showing the top 20
   candidates with their per-criterion grid.

## How a real team would use this

* Study coordinators run a fresh query weekly as the cohort fills.
* The per-criterion grid lets coordinators screen by phone with a
  cheat-sheet — no surprises during the consent visit.
* Recruitment ROI tracked over time: which criteria knock out the most
  candidates? Feeds back into protocol design.

## Run

```bash
# Default fictional GLP-1 trial spec
python -m cookbook.07_trial_eligibility.run --top 20

# Custom trial spec (any JSON conforming to TrialSpec dataclass)
python -m cookbook.07_trial_eligibility.run --spec my_trial.json
```
