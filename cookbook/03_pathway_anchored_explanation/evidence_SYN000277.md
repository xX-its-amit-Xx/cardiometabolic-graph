---
patient_id: SYN000277
generated_at: 2026-05-27T05:53:41.082837+00:00
model_version: 82c6bc977e73
---

# Evidence trail — `SYN000277`

**Last observed HbA1c:** 8.14%
**Predicted next HbA1c:** 7.80%  (Δ -0.34)
**Engagement dropout risk:** 0%

## Evidence

### `glucose_serum_last` — SHAP +0.242 (raising)
- Observed value: **162**
- Linked pathways:
  - [R-HSA-70171](https://reactome.org/PathwayBrowser/#/R-HSA-70171) — Glycolysis
  - [R-HSA-74160](https://reactome.org/PathwayBrowser/#/R-HSA-74160) — Signaling by insulin

### `glucose_serum_mean` — SHAP +0.173 (raising)
- Observed value: **175**
- Linked pathways:
  - [R-HSA-70171](https://reactome.org/PathwayBrowser/#/R-HSA-70171) — Glycolysis
  - [R-HSA-74160](https://reactome.org/PathwayBrowser/#/R-HSA-74160) — Signaling by insulin

### `HbA1c_min` — SHAP +0.120 (raising)
- Observed value: **7.91**
- Linked pathways:
  - [R-HSA-70171](https://reactome.org/PathwayBrowser/#/R-HSA-70171) — Glycolysis

### `triglycerides_last` — SHAP +0.033 (raising)
- Observed value: **202**
- Linked pathways:
  - [R-HSA-556833](https://reactome.org/PathwayBrowser/#/R-HSA-556833) — Metabolism of lipids

### `hdl_last` — SHAP +0.033 (raising)
- Observed value: **47.8**
- Linked pathways:
  - [R-HSA-556833](https://reactome.org/PathwayBrowser/#/R-HSA-556833) — Metabolism of lipids

### `ev_bucket_3` — SHAP -0.032 (lowering)
- Observed value: **33**
- No direct pathway link in the curated map.

## Behavioral context (last 30 days)
- app_open: 48
- glucose_log: 26
- message_response: 11

## Reviewer checklist

- [ ] Evidence above is consistent with the patient's known history.
- [ ] Predicted trajectory is clinically plausible given evidence.
- [ ] Linked pathways match the clinical reasoning.
- [ ] No protected-class features appear to be driving the prediction.
- [ ] Behavioral context does not contradict the recommendation.

Reviewer signature: ____________________  Date: __________
