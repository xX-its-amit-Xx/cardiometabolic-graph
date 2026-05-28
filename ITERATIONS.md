# Iteration log

Each iteration moves the repo closer to "a health-tech company could actually
use this." Entries are dated and reference the commits that landed the work.

## Iteration 1 — 2026-05-27 — Bootstrap end-to-end with synthetic data

**Goal.** Take the scaffolded code from initial scaffold to *actually
producing real outputs* — figures, screenshots, cookbook reports.

**What landed**
- Python 3.12 venv on C: (D: was at 100% capacity).
- `data/synthetic/generate_labs.py` produces six cardiometabolic labs at
  three visits per patient with archetype-driven trajectories, plus a
  per-patient random-walk volatility term to avoid trivially predictable
  test sets.
- Feature builder holds out the last HbA1c visit as the regression target
  and uses synthetic archetype labels as the ground-truth dropout class
  when available.
- Full pipeline run on 500 synthetic patients:
  - HbA1c GBM: Pearson **0.968**, MAE **0.275** (n=100 test).
  - Engagement dropout GBM: AUROC **1.000**, AUPRC **1.000** (positive rate
    18%) — separable because synthetic early_dropout decay is far stronger
    than other archetypes. Will need overlap noise in iter 2.
- Real `shap_summary.png` and `shap_bar.png` committed; SHAP top
  contributors are clinically plausible (`glucose_serum_last`,
  `triglycerides_last`, lipid-panel members).
- All four cookbook examples ran end-to-end; reports/CSVs/figures
  committed.
- Headless Streamlit screenshot capture via Playwright; sidebar
  preselect logic so the captured patient is one with predictions.
- pytest suite: **13/13 passing** in 1.56 s.

**What didn't land**
- MIMIC-IV demo download — started, in flight at end of iteration.
- PyG GNN training — heavy install deferred until disk pressure resolved.
- Federated / FHIR / MedCAT roadmap items.

## Iteration 2 — 2026-05-27 — Real MIMIC-IV demo integration

**Goal.** Replace synthetic-only with the MIMIC-IV demo (100 real
patients) and retrain end-to-end so the README numbers describe real
clinical data.

**What landed**
- 16 MB MIMIC-IV demo zip downloaded from PhysioNet (no credentials
  required), unzipped to `data/raw/mimic-iv/`.
- `etl/load_mimic_parquet.py` — Postgres-less path that maps MIMIC
  itemids directly to canonical lab names (the demo's `d_labitems` lacks
  the `loinc_code` column we expected, so we hand-curated the 8-itemid
  cardiometabolic mapping for HbA1c, glucose, total/HDL/LDL cholesterol,
  triglycerides). Writes parquet for patients, encounters, labs.
- 3,089 real cardiometabolic labs extracted from MIMIC, merged with the
  9,000 synthetic labs (12,089 total).
- Real-world signal noise added: 10% symmetric label-flip on the dropout
  target (mirrors the "paused vs dropped" ambiguity DTx teams actually
  face), softer `early_dropout` decay (0.0900 → 0.0350 per day).
- Retrained on combined data:
  - HbA1c GBM: **Pearson 0.875, MAE 0.384** (down from synthetic-only
    0.968 / 0.275 — the MIMIC patients add genuine signal that the
    synthetic deterministic trajectories were drowning out).
  - Dropout GBM: **AUROC 0.820, AUPRC 0.741** (down from 1.0/1.0 —
    realistic now).
- Dashboard screenshot recaptured against a real MIMIC patient
  (`MIMIC10002930`, last HbA1c 4.4%, model predicts 5.4% with +1.00
  delta, dropout risk 60% triggers re-engagement recommendation).
- Cookbook 03 evidence trail also now runs on real MIMIC patients.
- All 13 unit tests still pass.

**What didn't land**
- PyG GNN training — heavy install still deferred.
- NHANES + Reactome ETL — still synthetic.

## Iteration 3 — planned

**Goal.** Realistic dropout signal + GNN baseline.

1. Add archetype-overlap noise: after generating engagement, swap 10% of
   each patient's events with events drawn from a uniformly random other
   archetype. This breaks the perfect AUROC.
2. Install PyG (CPU wheels, ~600 MB) and actually train the GNN. Compare
   head-to-head with the GBM in the README results table.
3. Capture per-patient GNN attention attribution for a featured case
   study and add it as a figure in the cookbook 03 entry.

## Iteration 4 — planned

**Goal.** Make the dashboard tell a more complete story.

1. Add a "cohort view" tab — distribution of predicted vs observed HbA1c
   across the whole cohort, with a brush filter on dropout risk.
2. Hook the at-risk cohort cookbook output directly into the sidebar as
   a "today's call list" view.
3. Add a small Sankey of the patient's graph neighborhood (Pathway →
   Gene → Metabolite → Lab) using `pyvis` or `plotly` sankey.

## Iteration 5 — planned

**Goal.** Production credibility — the things a health-tech CTO would ask
about before adoption.

1. Add a model card under `docs/model_card.md` following Google's spec:
   intended use, OOD behavior, known failure modes, data lineage.
2. Add `docs/data_governance.md` describing how a real deployment would
   handle MIMIC credentialing, FHIR ingress auth, PHI scrubbing in logs.
3. Add a `monitoring/` module: lightweight Prometheus metrics for ETL
   row counts, model drift, and inference latency.
4. CI: enable the integration tests by spinning up Neo4j + Postgres
   service containers in the GitHub Actions workflow.
