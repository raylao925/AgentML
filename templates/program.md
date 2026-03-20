# program.md — Autonomous ML Protocol (Project: {{PROJECT_NAME}})

## 0) Mission
You are an Auto-ML Agent. Your task is to maximize {{PRIMARY_METRIC}} (and track {{SECONDARY_METRICS}}) for {{TASK_FAMILY}} (tabular / time-series / ranking / classification / regression) **without introducing data leakage**.

If folders/files under `projects/<project_slug>/` per `structure.md` do not exist, create them.

You may:
- Modify `configs/*.yaml` (models, hyperparams, CV, feature flags)
- Modify `src/features.py` (add/remove/fix features)
- Modify `src/train.py` (training flow, early stopping, loss, ranking objective, etc.)
- Modify `src/evaluate.py` (not the metric definition itself; only bug fixes / efficiency)
- Modify `src/ensemble.py` (based on baseline CV score, run AutoML to produce an ensemble; goal: better ensemble CV score)

You must obey all hard constraints in `AGENT_RULES.md`.

---

## 1) Task Understanding (Read First)
You must read and extract:
1) `docs/*.md`: full ML task content
2) `configs/baseline.yaml`: full baseline config

Then write your "task summary" at the top of `runs/<run_id>/notes.md`.

## 1.5 CV Bootstrap (Only if CV is missing) — Lock-in Mode (1)

Before any modeling experiments, the agent MUST ensure `docs/03_cv_strategy.md` exists and is authoritative.

### Step 1 — Detect CV Authority
- If user prompt explicitly specifies CV / group_key / time_col / query_id → treat as P0
- Else if `docs/03_cv_strategy.md` exists and non-empty → treat as P1
- Else → treat as P2 (auto-infer)

### Step 2 — If P2 (Auto-infer CV)
The agent MUST:
1) Run minimal EDA / schema scan:
   - df.info() summary (dtypes, non-null counts)
   - candidate id/group columns (high-cardinality keys)
   - candidate time columns (datetime-like names/types)
   - target distribution and duplicates by group
2) Infer appropriate CV strategy:
   - time-based split if temporal leakage risk or time_col is present and task is temporal
   - GroupKFold if group leakage risk is present (same entity repeats)
   - ranking requires query_id integrity (GroupKFold by query_id)
   - StratifiedKFold for classification if no time/group constraint overrides
3) Write `docs/03_cv_strategy.md`:
   - include the inferred rule + evidence + rationale
   - set "Lock-in: true" section
4) From this point onward, treat CV as P1 (locked). No silent changes.

### Step 3 — Proceed to Baseline
- After CV is locked, run the baseline experiment and log to `results.json`.

---

## 2) Hard Constraints (Must Not Break)
1) **No Leakage**: Do not use columns marked leak in `docs/01_data_card.md`; do not compute anything that lets valid fold see future info.
2) **CV/Split rules immutable**: All comparisons must use splits defined in `docs/03_cv_strategy.md`. To change CV, update `docs/03_cv_strategy.md` first with clear justification.
3) **Test set may not be used for tuning**: Test only for final report or submission; never for model/feature/threshold selection.
4) **Metric definition fixed**: PRIMARY_METRIC calculation must not change (unless fixing a clear bug, with reason recorded).
5) **Reproducibility**: Fix seed each run; record data_version, feature_version, code_hash.

---

## 3) Allowed Search Space (What to Explore)
### 3.1 Feature work
- Missing-value handling, encoding, scaling (must be fold-safe)
- Time-series features: lag/rolling (must be computed by time+fold to avoid leakage)
- Group aggregation (train fold data only)

### 3.2 Model work
- Tabular: LightGBM/XGBoost/CatBoost/LogReg/ElasticNet
- Multi-class: softmax objectives / one-vs-rest
- Ranking: pairwise/listwise (e.g. LGBMRanker/XGBRanker)
- Time-series: time-based split CV + tree/linear/seq model (if applicable)

### 3.3 Optimization work
- Early stopping, class weight, calibration (for classification)
- Ensembling (see `docs/05_ensemble.md`; must use OOF design)

---

## 4) Experiment Loop (Autonomous)
Each iteration you must:

### Step A — Plan
- Find best run in `results.json`
- Pick one "most likely to improve" change (1–2 changes per run for attribution)
- In `runs/<run_id>/notes.md` write:
  - Hypothesis
  - Expected direction
  - What changed（file + key diff）

### Step B — Execute
- Generate run_id: `YYYYMMDD_HHMM_<shortdesc>`
- Run training + evaluation
- Save:
  - `runs/<run_id>/params.json` (effective config)
  - `runs/<run_id>/metrics.json` (per fold + aggregate)
  - `runs/<run_id>/artifacts/*` (model, feature list, OOF, key plots)
  - `runs/<run_id>/notes.md`

### Step C — Log to Ledger (`results.json`)
Write run result to `results.json` (format see `docs/06_experiment_log.md` Spec).

### Step D — Keep / Discard
- Keep conditions (default; override in `AGENT_RULES.md`):
  - primary_metric_mean improvement >= {{IMPROVE_THRESHOLD}}
  - or primary_metric_mean similar but secondary improves significantly (within limits)
- If discard: still record reason (overfit / high variance / too slow / leakage risk)

---

## 5) Output Contract (Must Produce)
Each run must produce:
- One `runs/<run_id>/` folder (params, metrics, notes)
- One new record in `results.json`
- Updated docs if CV/features/model design changed