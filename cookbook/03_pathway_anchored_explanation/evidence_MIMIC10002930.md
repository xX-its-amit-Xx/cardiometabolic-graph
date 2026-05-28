---
patient_id: MIMIC10002930
generated_at: 2026-05-28T02:28:01.628641+00:00
model_version: 6d6442cdff0b
---

# Evidence trail — `MIMIC10002930`

**Last observed HbA1c:** 4.60%
**Predicted next HbA1c:** 5.40%  (Δ +0.80)
**Engagement dropout risk:** 60%

## Evidence

### `glucose_serum_mean` — SHAP -0.859 (lowering)
- Observed value: **90**
- Linked pathways:
  - [R-HSA-70171](https://reactome.org/PathwayBrowser/#/R-HSA-70171) — Glycolysis
  - [R-HSA-74160](https://reactome.org/PathwayBrowser/#/R-HSA-74160) — Signaling by insulin

### `glucose_serum_last` — SHAP -0.354 (lowering)
- Observed value: **75**
- Linked pathways:
  - [R-HSA-70171](https://reactome.org/PathwayBrowser/#/R-HSA-70171) — Glycolysis
  - [R-HSA-74160](https://reactome.org/PathwayBrowser/#/R-HSA-74160) — Signaling by insulin

### `glucose_serum_max` — SHAP -0.351 (lowering)
- Observed value: **119**
- Linked pathways:
  - [R-HSA-70171](https://reactome.org/PathwayBrowser/#/R-HSA-70171) — Glycolysis
  - [R-HSA-74160](https://reactome.org/PathwayBrowser/#/R-HSA-74160) — Signaling by insulin

### `triglycerides_min` — SHAP +0.075 (raising)
- Observed value: **24**
- Linked pathways:
  - [R-HSA-556833](https://reactome.org/PathwayBrowser/#/R-HSA-556833) — Metabolism of lipids

### `HbA1c_max` — SHAP -0.046 (lowering)
- Observed value: **5.8**
- Linked pathways:
  - [R-HSA-70171](https://reactome.org/PathwayBrowser/#/R-HSA-70171) — Glycolysis

### `glucose_serum_min` — SHAP -0.040 (lowering)
- Observed value: **58**
- Linked pathways:
  - [R-HSA-70171](https://reactome.org/PathwayBrowser/#/R-HSA-70171) — Glycolysis
  - [R-HSA-74160](https://reactome.org/PathwayBrowser/#/R-HSA-74160) — Signaling by insulin

## Behavioral context (last 30 days)
_No engagement events for this patient._

## Reviewer checklist

- [ ] Evidence above is consistent with the patient's known history.
- [ ] Predicted trajectory is clinically plausible given evidence.
- [ ] Linked pathways match the clinical reasoning.
- [ ] No protected-class features appear to be driving the prediction.
- [ ] Behavioral context does not contradict the recommendation.

Reviewer signature: ____________________  Date: __________
