# Cookbook 08 — Pharmacist intervention triggers

## The question (from a value-based-care pharmacist team)

> "We have one pharmacist for ~3,000 covered patients. Tell us which 30
> patients to call this week where a brief medication-adherence
> conversation has the highest chance of changing HbA1c trajectory."

Pharmacists in value-based primary care drive measurable outcomes when
they call the *right* patients — the leverage is in patient selection,
not script length.

## What this example does

We define a **pharmacist-leverage score** combining three signals:

1. **Adherence proxy:** number of `glucose_log` events in the last 30
   days vs the prior 30 days. A drop > 40% suggests the patient is
   disengaging from self-management — a strong predictor of upcoming
   medication non-adherence.
2. **Trajectory risk:** predicted HbA1c rise (from the trained GBM).
3. **Modifiability:** patients with a stable LDL but rising HbA1c are
   higher-leverage than patients with everything spiraling — the
   pharmacist can affect one variable, not three.

The leverage score is:
```
score = (predicted_hba1c_delta * 2.0)
      + (adherence_dropoff_fraction * 1.5)
      - (number_of_other_failing_metrics * 0.5)
```

Top N patients get a call list entry with:
- Phone-script bullets ("ask about last refill date", "verify they're
  still on once-weekly semaglutide", "check for GI side effects").
- A trajectory chart they can text the patient.

## How a real team would use this

* Pharmacist runs this Monday morning, calls top 30 in priority order.
* Outcomes tracked over 90 days: HbA1c delta in *called* vs *not-called*
  cohort.
* If lift is significant, the leverage score's weights are tuned via
  the at-risk cohort cookbook's feedback loop.

## Run

```bash
python -m cookbook.08_pharmacist_intervention.run --top 30
```
