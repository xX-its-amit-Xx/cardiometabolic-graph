# Cookbook 05 — Prior-auth note for GLP-1 escalation

## The question (from a DTx care team's PA specialist)

> "Payer is denying our GLP-1 prior-authorization requests because the
> notes we send don't cleanly cite the metrics that justify medical
> necessity. Can you generate a structured PA note for a given patient
> that pulls the exact HbA1c values, trends, and engagement evidence
> needed — in the format the payer expects?"

This is the *narrative + evidence* use case. The payer wants:
1. Patient demographics + diagnosis.
2. Most recent HbA1c with date.
3. Trend over the last 6 months.
4. Behavioral adherence evidence (we use app engagement as a proxy).
5. Failed prior-therapy attestation (synthesized here; real deployments
   pull from the EHR's `medication_history`).

## What this example does

For a target patient ID:

1. Loads the cached labs + engagement + predictions.
2. Builds a `PatientFacts` payload for the **prior_auth** audience and
   runs it through the summarizer (deterministic by default; switchable
   to Ollama / Phi-3 via `CMG_SUMMARIZER`).
3. Wraps the LLM/template prose in a structured PA form template
   (free-text fields plus a metric table).
4. Writes `pa_note_<patient_id>.md` — ready to copy into the payer
   portal.

## How a real team would use this

* PA specialists run this from a "pending PA" worklist.
* The structured metric table reduces back-and-forth — payers reject
  ~30% of GLP-1 PAs on documentation grounds; getting the metrics in
  the right place cuts that bounceback.
* The same payload feeds the appeals letter if the initial PA is denied.

## Run

```bash
python -m cookbook.05_prior_auth_note.run --patient SYN000001 --drug "semaglutide 1mg weekly"

# With an open-weight LLM polishing the prose:
CMG_SUMMARIZER=ollama python -m cookbook.05_prior_auth_note.run --patient SYN000001
```
