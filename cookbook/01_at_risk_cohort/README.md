# Cookbook 01 — Weekly at-risk cohort for care coordinators

## The question (from a DTx care team)

> "Every Monday morning we have capacity for about 50 outreach calls. Give us
> the 50 patients most likely to have a clinically meaningful rise in HbA1c
> over the next 90 days — and tell us *why* you picked each one so we can
> tailor the conversation."

This is the highest-leverage application of the whole pipeline: rank patients
by predicted HbA1c rise, then give the coordinator the top three reasons.

## What this example does

1. Loads the trained GBM HbA1c regressor + cached SHAP values.
2. Computes `predicted_next - last_observed` for every patient.
3. Filters to patients whose last observed HbA1c was at or above 6.0% (so we
   don't waste outreach on stably-normal patients).
4. Ranks the top N by predicted rise.
5. For each, pulls the top three SHAP-positive features and writes a
   one-line *why we called you* explanation using the same template engine
   the dashboard uses.
6. Writes:
   - `cohort.csv`           — full ranked list
   - `report.md`            — the cohort with per-patient reasoning
   - `figures/cohort_distribution.png` — predicted rise distribution

## How a real team would use this

* Schedule the script weekly (cron / GitHub Action).
* Drop `cohort.csv` into the care-coordinator workqueue.
* Coordinators see the `report.md` reason line — *"flagged because: HbA1c
  uptrend over 90 days, low app engagement (3 opens / 30d), LDL elevated"*
  — before they pick up the phone.

## Run

```bash
python -m cookbook.01_at_risk_cohort.run --top 50 --threshold 6.0
```
