# 03 — CV Strategy (Authoritative, Lock-in)

> This document is the single source of truth for data splitting and validation protocol.
> Once established, it is LOCKED and must not be silently changed by the agent.

---

## 0) Lock-in Status
- lock_in: true
- established_by: {{P0_user_specified | P2_auto_infer}}
- established_datetime: {{ISO_DATETIME}}
- change_policy:
  - Only change if user explicitly requests
  - Must update this doc with justification and record in runs/<run_id>/notes.md

---

## 1) Task Family & Metric
- task_family: {{classification_binary | classification_multiclass | regression | ranking | time_series}}
- target: {{TARGET_COLUMN}}
- primary_metric: {{AUC | LogLoss | RMSE | MAE | NDCG@10 | MAP@K ...}}
- metric_direction: {{higher_is_better | lower_is_better}}

---

## 2) CV Definition (Authoritative)

### 2.1 Split Type
- cv_type: {{StratifiedKFold | KFold | GroupKFold | TimeSeriesSplit | CustomTimeFolds}}
- n_splits: {{K}}
- random_state: {{SEED}}
- shuffle: {{true/false}}

### 2.2 Keys (User-defined or Inferred)
- group_key: {{null or column_name}}
- time_col: {{null or column_name}}
- ranking_query_id (if ranking): {{null or column_name}}
- gap_or_embargo (optional): {{null or e.g. "7d"}}

### 2.3 Rules
- All rows with same group_key MUST stay in the same fold (if group_key is set)
- Time-based validation MUST occur strictly after training window (if time_col is set)
- Ranking: all rows with same query_id MUST stay in the same fold (mandatory for ranking)
- Test set MUST NOT be used for tuning or selection (final report only)

---

## 3) Fold-safe Feature Policy (No Leakage)
- Any preprocessing (imputer/scaler/encoder) must be fit on train fold only
- Target encoding must be OOF / fold-safe only
- Aggregations must be computed using train fold only
- Time-series features (lag/rolling) must use past-only and fold-safe computation

---

## 4) Auto-infer Evidence & Rationale (Required if established_by = P2_auto_infer)

### 4.1 Evidence Collected
- df.info() summary:
  - row_count: {{N}}
  - columns_count: {{M}}
  - datetime candidates: {{[...]}}, parse success: {{yes/no}}
  - high-cardinality id candidates: {{[...]}}

- Target distribution:
  - class counts / stats: {{...}}

- Leakage risk signals:
  - repeated entities by candidate group_key: {{dup_rate}}
  - time range: {{min_date}} → {{max_date}}
  - any "post-event" columns detected: {{[...]}} (if any)

### 4.2 Why This CV
- Chosen cv_type: {{...}}
- Rationale (short):
  - {{e.g. time_col exists and horizon implies temporal leakage risk}}
  - {{e.g. same customer appears multiple times, require GroupKFold to prevent entity leakage}}
  - {{e.g. ranking task requires query integrity}}
- Alternatives considered and rejected:
  - {{...}}

---

## 5) Outputs Required Per Fold
- per_fold metric
- mean/std aggregate
- OOF predictions saved (for ensemble)
- resource usage (train_seconds, infer_seconds, memory)

---
