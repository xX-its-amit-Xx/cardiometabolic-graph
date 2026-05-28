# Cohort drift monitor

_Comparing the most recent window to the cached training reference._

**9 alarming features** (PSI > 0.2).

![top PSI](figures/psi_top.png)

| Feature | PSI | Ref mean | Cur mean | Interpretation |
|---------|-----|----------|----------|----------------|
| `ev_bucket_2` | 6.50 | 31.64 | 5.66 | down 25.97 vs reference |
| `count_message_response_30d` | 0.39 | 11.96 | 14.35 | up 2.39 vs reference |
| `sum_message_response_30d` | 0.39 | 11.96 | 14.35 | up 2.39 vs reference |
| `ev_bucket_1` | 0.39 | 30.95 | 37.14 | up 6.19 vs reference |
| `sum_app_open_30d` | 0.36 | 38.97 | 46.76 | up 7.79 vs reference |
| `count_app_open_30d` | 0.36 | 38.97 | 46.76 | up 7.79 vs reference |
| `ev_bucket_0` | 0.34 | 30.70 | 36.84 | up 6.14 vs reference |
| `count_glucose_log_30d` | 0.32 | 15.44 | 18.52 | up 3.09 vs reference |
| `sum_glucose_log_30d` | 0.29 | 2075.36 | 2490.43 | up 415.07 vs reference |