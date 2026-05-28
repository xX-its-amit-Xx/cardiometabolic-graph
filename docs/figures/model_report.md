# Model report

## HbA1c trajectory (regression)

| Model | Pearson r | MAE (HbA1c %) | n_train | n_test |
|-------|-----------|---------------|---------|--------|
| LightGBM | 0.968 | 0.282 | 400 | 100 |
| GAT-GNN  | 0.701 | 0.790 | 400 | 100 |

## Engagement dropout (binary classification)

| Model | AUROC | AUPRC | n_train | n_test | pos rate (test) |
|-------|-------|-------|---------|--------|-----------------|
| LightGBM | 0.841 | 0.813 | 400 | 100 | 21.00% |

## Top features (GBM HbA1c)

- `glucose_serum_last` — 320
- `triglycerides_last` — 151
- `HbA1c_min` — 132
- `hdl_last` — 125
- `glucose_serum_mean` — 115
- `triglycerides_max` — 92
- `hdl_mean` — 91
- `glucose_serum_min` — 91
- `ldl_min` — 88
- `cholesterol_total_max` — 85
- `ev_bucket_3` — 82
- `glucose_serum_max` — 76
- `ev_bucket_2` — 74
- `HbA1c_last` — 72
- `ldl_max` — 69
