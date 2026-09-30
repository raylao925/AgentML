# 01 — Lifecycle & Gates

The agent runs one **autonomous loop** per iteration. Gates below are hard: do not pass a
gate until its exit condition is met.

## Lifecycle

```
0. Bootstrap   → agentml new / project.yaml read
1. Discover    → 03_task_discovery  (mode=customer) | sources/kaggle.md (mode=kaggle)
2. Ingest      → 04_ingestion       (data/processed + reports/data_manifest.json)
3. EDA         → 02_eda.md          (agentml eda)
4. CV lock-in  → 04_cv_strategy.md written + locked_hash   ← GATE
5. Baseline    → agentml run        (first run in runs/ + results.json)
6. Iterate     → Plan → Execute → Log → Keep/Discard (1–2 change families/run)
7. Ensemble    → 07_ensemble
8. Deliver     → 08_delivery
```

## Gate A — Task confirmed (customer mode)
- `project.yaml` `project.mode == customer` → `doc/00_problem_statement.md` must have
  `confirmed_by_user: true` before modeling. See `03_task_discovery.md`.

## Gate B — CV locked
- `doc/04_cv_strategy.md` exists, is non-empty, and contains `locked_hash:`.
- The hash matches the current split implementation (verified by `agentml guardrails`).
- Until then, only baseline/inference exploration is allowed — no hyperparameter search.

## Gate C — Data manifest
- `reports/data_manifest.json` exists (row count / schema / hash) for the current
  `data_version`. Re-run `agentml ingest` if the raw data changes.

## Gate D — Before delivery
- `agentml guardrails` passes (no leakage, no placeholder residue, ledger valid).
- Test set was **never** used for tuning.

## Iteration loop (Step A–D)

### Step A — Plan
Find the best `keep` run in `results.json`; pick one highest-expected-value change
(1–2 families: one feature family / one hyperparam set / one model class / one ensemble method).
Write hypothesis + expected direction + file-level diff into `runs/<run_id>/notes.md`.

### Step B — Execute
`run_id = YYYYMMDD_HHMM_<shortdesc>`. Produce `runs/<run_id>/{params,metrics,notes}` +
`artifacts/*`.

### Step C — Log
Append exactly one record to `results.json` (schema = `06_experiment_log.md`).

### Step D — Keep / Discard
Thresholds live in `configs/search_space.yaml:policy`. If discarded, still log a reason
(variance / speed / leakage risk / constraint violation).

## Reproducibility
- Fix seeds (`seed`, `seed_list`) and record `data_version`, `feature_version`, `code_hash`.
- Same data + config ⇒ same artifacts.