# 06 — Experiment Log (Ledger + Run Artifacts)

> Purpose:
> - `results.json`: structured experiment records (for agent/script parsing, ranking, picking best run)
> - `runs/<run_id>/`: full artifacts per run (params / metrics / artifacts / notes)
>
> Principle: **Every run (KEEP or DISCARD) must be logged** for traceability, reproducibility, and comparability.

---

## A) Run Folder Contract (Must Produce)

Each experiment must create a folder `runs/<run_id>/` with at least:

- `runs/<run_id>/params.json`  
  - Effective config (model / CV / feature flags / seed)
- `runs/<run_id>/metrics.json`  
  - per-fold metrics + aggregate (mean/std) + key diagnostics (calibration / ndcg@k breakdown)
- `runs/<run_id>/notes.md`  
  - Must include: Hypothesis, What changed, Why, Outcome, Decision, Next step
- `runs/<run_id>/artifacts/` (as needed)
  - `model.*` (pkl / cbm / txt / onnx etc.)
  - `feature_list.json`
  - `oof_predictions.*` (parquet or csv recommended)
  - `test_prediction.*` (after inference, csv recommended)
- `runs/<run_id>/plots/` (at least feature importance)
  - Key plots (feature importance, ROC, PR, residuals, calibration curve, time split diagnostics)

> Tip: Use paths relative to project root to avoid breakage when moving the project.

---

## B) Ledger File: `results.json`

- Location: `projects/<project_slug>/results.json`
- Format: **JSON Array** (each element = one run record)
- Update policy: **append-only** (append only, never overwrite history; add correction record if needed)
- Sort: by `datetime` old→new (or new→old, but keep consistent)
- Minimum: every run must write one record (KEEP or DISCARD)

> NOTE (optional): For easier append and concurrency, use `results.jsonl` (one JSON object per line); this project uses `results.json` by default.

---

## C) `results.json` Schema (Recommended)

Each record **must** include these fields (add/remove as needed, but keep core fields):

```json
{
  "run_id": "20260313_1040_lgbm_baseline",
  "datetime": "2026-03-13T10:40:00+08:00",

  "task": {
    "family": "classification_binary",
    "target": "is_cancel",
    "primary_metric": "AUC",
    "secondary_metrics": ["LogLoss", "Brier", "F1"]
  },

  "data": {
    "data_version": "raw_snapshot_20260301",
    "row_count_train": 123456,
    "row_count_valid": 30864,
    "schema_hash": "sha1:xxxx",
    "leakage_policy_ref": "docs/01_data_card.md"
  },

  "cv": {
    "cv_type": "GroupKFold",
    "n_splits": 5,
    "random_state": 42,
    "group_key": "customer_id",
    "time_col": null,
    "gap_or_embargo": null
  },

  "features": {
    "feature_set_id": "fs_baseline_v1",
    "feature_version": "v1.0.3",
    "feature_flags": {
      "use_target_encoding": false,
      "use_lag_features": false,
      "use_rolling_features": false
    },
    "dropped_columns": ["leak_col_1", "post_event_status"]
  },

  "model": {
    "name": "LightGBM",
    "objective": "binary",
    "params": {
      "learning_rate": 0.05,
      "num_leaves": 64,
      "min_data_in_leaf": 50
    }
  },

  "metrics": {
    "primary": {
      "name": "AUC",
      "mean": 0.7812,
      "std": 0.0041,
      "per_fold": [0.7790, 0.7851, 0.7803, 0.7820, 0.7800]
    },
    "secondary": {
      "LogLoss_mean": 0.4920,
      "Brier_mean": 0.1830
    }
  },

  "resources": {
    "train_seconds": 312.4,
    "infer_seconds": 0.86,
    "peak_memory_gb": 4.2,
    "model_size_mb": 18.7
  },

  "artifacts": {
    "run_dir": "runs/20260313_1040_lgbm_baseline/",
    "params_path": "runs/20260313_1040_lgbm_baseline/params.json",
    "metrics_path": "runs/20260313_1040_lgbm_baseline/metrics.json",
    "notes_path": "runs/20260313_1040_lgbm_baseline/notes.md",
    "model_path": "runs/20260313_1040_lgbm_baseline/artifacts/model.pkl",
    "oof_path": "runs/20260313_1040_lgbm_baseline/artifacts/oof_predictions.parquet",
    "plots_dir": "runs/20260313_1040_lgbm_baseline/plots/"
  },

  "decision": {
    "status": "keep",
    "rule_version": "keep_discard_v1",
    "reason": "Primary metric improved with stable std; resources within limits."
  },

  "notes_short": "Baseline established. Next: try class_weight + TE-safe encoding."
}
```

### C.1) Task-specific Add-ons (Optional)

#### Ranking tasks—suggested add-ons:
- `task.query_key` (e.g. query_id / session_id)
- `metrics.primary.per_query` (if storing finer detail)
- `model.objective` (pairwise/listwise, ndcg)

#### Time-series tasks—suggested add-ons:
- `cv.time_col`、`cv.gap_or_embargo`
- `task.horizon`
- `data.time_range_train/valid`

---

## D) Keep/Discard Rule (Default)

> Goal: enable **automatic**, **consistent**, **auditable** decisions and avoid subjective judgment.

### D.1 Default Decision Logic

1) Find the **best KEEP** baseline in `results.json` (sort by `metrics.primary.mean`; ranking/regression by direction)
2) Compute delta vs best baseline: `delta = new_mean - best_mean` (or for RMSE: best_mean - new_mean)
3) Apply rules:

#### ✅ KEEP (default)
- `delta >= improve_threshold`
- and `std` not significantly worse (e.g. std increase <= 20%)
- and `resources` within limits (train/infer/memory/model_size)

#### ✅ KEEP (Tie-break, optional)
- `abs(delta) <= tie_margin`
- but secondary metrics clearly improve (LogLoss/Brier/latency)
- and no hard constraint violated (especially leakage / test rule)

#### ❌ DISCARD (default)
- KEEP conditions not met, or
- any leakage risk / CV rule violation / use of test for tuning

### D.2 Suggested Defaults (override in `AGENT_RULES.md`)
- `improve_threshold`:
  - classification AUC: +0.001 ~ +0.002
  - regression RMSE: relative improvement ≥ 0.2% (or absolute per scale)
  - ranking NDCG@10: +0.002
- `tie_margin`: 0.0002 (adjust per data size)
- `resources limits`: specified by project in `configs/baseline.yaml` or `AGENT_RULES.md`

---

## E) How to Write `results.json` (Append Policy)

### E.1 Append-only behavior
- Do not overwrite existing records
- If run fails (crash / timeout), still write a record:
  - `decision.status = "discard"`
  - `decision.reason = "runtime_error: ..."`
  - `metrics` may be empty or `null`

### E.2 Minimal Required Fields
Every record must have at least:
- `run_id`
- `datetime`
- `task.family`
- `cv.cv_type`
- `model.name`
- `metrics.primary.mean` (if successful)
- `decision.status`
- `artifacts.run_dir`

---

## F) Companion File: `runs/<run_id>/notes.md` Template

Recommended format for each run's notes.md:

```md
# Run {{run_id}}

## Task Summary (from 00_problem_statement)
- task:
- target:
- primary metric:
- constraints:

## Hypothesis
(Expected direction of change and why)

## What Changed
- files changed:
- config diff:
- feature diff:
- model diff:

## Results
- primary metric (mean/std):
- secondary:
- resource usage:

## Decision
- KEEP / DISCARD
- reason:

## Next Step
(Next direction worth trying)
```