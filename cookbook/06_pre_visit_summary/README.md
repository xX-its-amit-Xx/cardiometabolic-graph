# Cookbook 06 — Pre-visit summary for the attending

## The question (from a primary-care physician)

> "I have 20 patients on my schedule today. For each one, give me the
> 30-second version I should read before walking into the room — what's
> changed since the last visit, what should I ask about, and what's the
> top risk."

This is the *worklist* use case. A physician's morning prep is
non-negotiable, and a structured one-pager per patient — assembled from
the graph in under a second — pays for itself in the first encounter.

## What this example does

Given a list of patient IDs (or a date for "today's schedule"):

1. For each patient, pulls labs, engagement, predictions, SHAP factors.
2. Runs the summarizer at the **clinician** audience.
3. Adds a "questions to ask" block derived from rule-based triggers:
   - HbA1c trend rising → "ask about recent diet/stress changes"
   - LDL ≥130 → "discuss statin adherence"
   - dropout risk ≥40% → "explore barriers to app use"
4. Writes one markdown section per patient into `pre_visit_<date>.md` —
   ready to print or render as PDF.

## How a real team would use this

* Front-desk staff or MA runs the script at 7 AM against the day's
  schedule.
* The PDF is in the physician's chart before they sit down.
* "Questions to ask" surfaces the highest-leverage 30 seconds of the
  encounter.

## Run

```bash
# Top 5 patients (or any specific IDs)
python -m cookbook.06_pre_visit_summary.run --patients SYN000001 SYN000017 SYN000042

# Top N by predicted HbA1c delta
python -m cookbook.06_pre_visit_summary.run --top 10

# Patient-facing version (plain language)
python -m cookbook.06_pre_visit_summary.run --top 5 --audience patient
```
