# Model report

## HbA1c trajectory (regression)

| Model | Pearson r | MAE (HbA1c %) | n_train | n_test |
|-------|-----------|---------------|---------|--------|
| LightGBM | 0.875 | 0.384 | 480 | 120 |

## Engagement dropout (binary classification)

| Model | AUROC | AUPRC | n_train | n_test | pos rate (test) |
|-------|-------|-------|---------|--------|-----------------|
| LightGBM | 0.820 | 0.741 | 480 | 120 | 16.67% |

## Top features (GBM HbA1c)

- `glucose_serum_last` — 265
- `glucose_serum_mean` — 157
- `glucose_serum_max` — 88
- `triglycerides_last` — 75
- `glucose_serum_min` — 62
- `triglycerides_max` — 61
- `hdl_last` — 50
- `triglycerides_min` — 46
- `HbA1c_max` — 45
- `HbA1c_last` — 43
- `birth_year` — 40
- `HbA1c_min` — 38
- `cholesterol_total_last` — 36
- `triglycerides_mean` — 34
- `hdl_min` — 34
