# 06 — Modeling

> Code: `src/models.py` (factory), `src/train.py` (single), `src/train_multi_model.py`
> (multi). Design: `doc/07_modeling.md`. Domain: `configs/search_space.yaml`.

## Model factory
`models.py` maps `model.name` → estimator per task family. Every estimator must expose
`predict_proba` (classification) / `predict` (regression/ranking) and accept `random_state`.

| task family | allowed models |
|---|---|
| classification_binary | LightGBM, XGBoost, CatBoost, LogisticRegression |
| classification_multiclass | LightGBM, XGBoost, CatBoost |
| regression | LightGBM, XGBoost, CatBoost, ElasticNet |
| ranking | LGBMRanker, XGBRanker (needs `group` = query size) |
| time_series | LightGBM, XGBoost (time-based CV) |

## Procedure
1. Load `configs/baseline.yaml`; resolve CV from `doc/04_cv_strategy.md` (locked).
2. For each fold: `prepare_fold` → fit pipeline on train fold → train → predict valid.
3. Collect **OOF predictions** (required for ensembling) + per-fold metrics.
4. Save artifacts: `model.pkl`, `oof_predictions.npy` (+ index), `feature_list.json`,
   `dataset_profile_train.json` (recommended).

## Ranking specifics
- Group by `query_id`; folds must not split a query.
- Use `eval_at` (e.g. `[10]`); objective `lambdarank`.
- Metric = NDCG@k, higher-is-better.

## HPO
- Runtime: `python src/tune.py --config configs/baseline.yaml --model <M> --n-trials 60
  --timeout-s 7200` (or `agentml tune ...`). Reads `search_space.yaml:search.hyperparams.<M>`,
  objective = mean per-fold primary metric over the **locked** folds; prunes lagging trials;
  writes `runs/tune_*/{study.sqlite3, trials.csv, best_params.json}` + one ledger record per study
  plus one final-retrain run (`hpo_<M>_seed<seed>_cv<n>`, attribution: one param set per run).
- Search inside `search_space.yaml:hyperparams`. At most one hyperparam set per run when
  no feature change is present (attribution).
- Early stopping uses the **validation fold**, never the test set.
- Keep seeds fixed; use `seed_list` for ensemble diversity.
- HPO is queue item family #3 in `10_iteration_loop.md` §3 — run it when evidence says the
  current params sit far from the space midpoints.

## Output contract
`runs/<run_id>/`: `params.json`, `metrics.json` (per-fold + mean/std), `notes.md`,
`artifacts/*`.

## Do not
- Do not touch the test set during training/tuning.
- Do not change CV between runs without a doc update (Frozen zone).