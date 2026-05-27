# Model report

## HbA1c trajectory (regression)

| Model | Pearson r | MAE (HbA1c %) | n_train | n_test |
|-------|-----------|---------------|---------|--------|
| LightGBM | 0.968 | 0.275 | 400 | 100 |

## Engagement dropout (binary classification)

| Model | AUROC | AUPRC | n_train | n_test | pos rate (test) |
|-------|-------|-------|---------|--------|-----------------|
| LightGBM | 1.000 | 1.000 | 400 | 100 | 18.00% |

## Top features (GBM HbA1c)

- `glucose_serum_last` — 454
- `triglycerides_last` — 287
- `cholesterol_total_mean` — 263
- `ldl_min` — 252
- `hdl_last` — 233
- `triglycerides_min` — 204
- `triglycerides_max` — 203
- `hdl_mean` — 191
- `cholesterol_total_max` — 186
- `hdl_min` — 181
- `HbA1c_min` — 172
- `cholesterol_total_min` — 172
- `glucose_serum_min` — 172
- `ldl_max` — 172
- `glucose_serum_mean` — 168
