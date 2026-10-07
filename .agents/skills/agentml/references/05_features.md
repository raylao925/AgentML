# 05 — Features

> Code: `src/features.py`. Registry: `doc/03_features_engineering.md`. Flags:
> `configs/baseline.yaml:features.flags`.

## Fold-safe requirement (hard)
Every transformer is fit on the **train fold only**, then applied to valid/test. This covers
imputers, scalers, encoders, target encoders, group aggregations, and any auto-feature step.
No statistic may be computed over valid/test rows.

## Feature families (toggle via flags)
| flag | family | leakage rule |
|---|---|---|
| `use_scaler` | numeric scale | fit on train fold |
| `use_onehot` | categorical one-hot | fit on train fold; `handle_unknown="ignore"` |
| `use_target_encoding` | smoothed target mean | **OOF / train-fold-only**, `smoothing`, `min_samples_leaf` |
| `use_lag_features` | `lag_k` | strictly past, within fold window |
| `use_rolling_features` | rolling mean/std | strictly past, within fold window |
| `use_group_aggregations` | group mean/std/count | computed on train-fold rows only |
| `use_featuretools` | DFS auto-features | DFS fit on train fold, primitives re-applied to valid |

## Feature registry (`doc/03_features_engineering.md`)
For each feature family record: name, definition, source columns, leakage assessment,
evidence (ablation run id), and status (adopted / rejected). Rejected features stay listed
with the reason so they are not re-proposed.

## Output contract
- `runs/<run_id>/artifacts/feature_list.json` — ordered output feature names.
- The trained preprocessors are saved with the model (single sklearn `Pipeline`).

## Ablation rule
A feature family is adopted only if it survives an ablation run: add the family alone and
compare primary metric (mean) with the current best `keep` run, per `search_space.yaml:policy`.
Change at most 1–2 families per run.

## Do not
- Do not use the `target` column (or any proxy/label-window column) as an input feature.
- Do not fit any encoding on the full dataset "for speed" — that is leakage.