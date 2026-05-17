# Cookbook 04 — Cohort drift monitor

## The question (from an MLOps engineer)

> "Our HbA1c model has been live for six months. The intake population this
> month feels different from when we trained — younger, more app-savvy,
> different lab distributions. Tell me when the live cohort drifts far
> enough from the training cohort that we need to retrain."

Classic data drift problem, but for a *patient* model we care about three
distinct flavors of drift simultaneously:

1. **Clinical drift** — distribution of labs (HbA1c, glucose, LDL).
2. **Behavioral drift** — distribution of engagement statistics.
3. **Composition drift** — proportions of synthetic-vs-MIMIC vs other
   cohorts, since the population mix changes how the model generalizes.

## What this example does

1. Loads the cached feature frame (treats this as the "training reference").
2. Re-samples a recent window (default last 30 days) from the engagement
   stream and re-derives the engagement features for that window only.
3. Computes per-feature drift scores:
   - Numeric features: Population Stability Index (PSI) over 10 quantile
     bins.
   - Categorical / cohort: chi-square test on contingency table.
4. Flags any feature with PSI > 0.2 (the standard "alarm" threshold).
5. Emits:
   - `drift_table.csv` — every monitored feature with its PSI / p-value.
   - `report.md` — only the alarming features, with one-line interpretation.
   - `figures/psi_top.png` — top 15 drifting features by PSI.

## How a real team would use this

* Schedule weekly.
* Pipe the `report.md` into Slack via a GitHub Action when any feature
  trips the alarm threshold.
* The reference window can be parameterized to "last quarter" rather than
  the static training window — this catches *gradual* drift that wouldn't
  trip an absolute threshold.

## Run

```bash
python -m cookbook.04_cohort_drift_monitor.run --window-days 30 --psi-alarm 0.2
```
