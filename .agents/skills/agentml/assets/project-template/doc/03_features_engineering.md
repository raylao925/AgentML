# 03 — Features Engineering

> Purpose: this document is the **feature registry** — one place to record which feature families
> exist, where they are implemented, which flags switch them on, and the fold-safety evidence.
> Modeling strategy / model choices live in `07_modeling.md`; EDA findings live in `02_eda.md`.

---

## 1) Feature Pipeline Contract (Fold-safe)
- Owner module: `src/features.py`
- Flags & params: `configs/baseline.yaml` → `features.flags` / `features.params`
- Allowed search boundary: `configs/search_space.yaml` → `allowed.feature_flags`
- Hard rule: every transformer is **fit on the train fold only**, then applied to valid/test.
  Target encoding must be OOF; aggregations must use train-fold rows only (see `AGENT_RULES.md` §1).

---

## 2) Feature Registry (one row per feature family)

| feature_set_id | family | implementation | flags | fold-safe | added_in_run | CV delta | decision |
|---|---|---|---|---|---|---|---|
| fs_baseline_v1 | raw + impute + onehot | `src/features.py::build_feature_pipeline` | `use_scaler`, `use_onehot` | yes | {{run_id}} | 0.000 | keep |

---

## 3) Column Handling Policy
- Excluded: target, `data.id_cols`, `data.drop_cols` (leakage columns — see `01_data_card.md`)
- Numeric: impute (median) + optional `StandardScaler`
- Categorical: impute (constant `__MISS__`) + one-hot (`handle_unknown="ignore"`) or native categorical
- Datetime: cyclical (sin/cos), day-of-week/month, relative time — **must not use future information**
- High-cardinality categorical: fold-safe target encoding (with smoothing) — fitted on train fold only

---

## 4) Auto Feature / AutoML
- `featuretools`: allowed only if fit **per fold**; record `trans_primitives`, `agg_primitives`, `max_depth` here.
- `pycaret` (or other AutoML): candidate discovery only — any adopted pipeline must be re-implemented
  explicitly in `src/features.py` so it stays reproducible and fold-safe.

---

## 5) Version & Traceability
- feature_version: `v0`
- data_version: see `configs/baseline.yaml` → `data.data_version`
- Every registry change must be logged in `runs/<run_id>/notes.md` and reflected in this table.
 