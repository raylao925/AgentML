# Configs Template Guide (for OpenClaw Agents)

This directory provides **config templates** so OpenClaw / Auto-ML agents know:

- `baseline.yaml`: complete executable config for a single experiment
- `search_space.yaml`: explorable model / feature / hyperparam space and guardrails

Each project copies these to `projects/<project_slug>/configs/`; the agent then adapts them per task.

---

## 1. `baseline.yaml` — The "real config" for a single run

**Purpose**:

- Defines all details for **task / data / cv / features / model / training / output**
- `src/train.py` reads this YAML and runs one experiment
- Full config snapshot per run is stored in `runs/<run_id>/params.json`

**Key sections**:

- `project`:
  - `name`: project slug (e.g. `playground-series-s6e3`)
  - `seed`: global random seed (CV + model)

- `task`:
  - `family`: `classification_binary` / `classification_multiclass` / `regression` / `ranking` / `time_series`
  - `target`: target column name (e.g. `Churn`)
  - `target_positive_label` (optional): for binary, which label is 1 (e.g. `"Yes"`)
  - `primary_metric` / `secondary_metrics`: align with `doc/00_problem_statement.md`, `doc/06_experiment_log.md`

- `data`:
  - `train_path` / `test_path`: data paths (relative to project root)
  - `id_cols`: ID columns (not used in model; carried to submission at inference)
  - `drop_cols`: explicitly forbidden columns (incl. leakage)
  - `time_col` / `group_key`: if time-based / group-based CV, align with `03_cv_strategy.md`

- `cv`:
  - `cv_type`: e.g. `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`
  - `n_splits`, `shuffle`, `random_state`: fixed CV settings
  - `stratify_col`: usually target; if `null` let code use target
  - `time_col` / `group_key` / `gap_or_embargo`: match `doc/03_cv_strategy.md`; agent must not change silently; update doc first

- `features`:
  - `feature_set_id`: feature version ID (e.g. `fs_baseline_v1` / `fs_ft_v1`)
  - `flags`: toggles (e.g. `use_scaler`, `use_onehot`, `use_featuretools`) for agent to switch during search
  - `params`: parameters for feature modules (target encoding, lag/rolling, featuretools, etc.)

- `model`:
  - `name`: `LightGBM` / `XGBoost` / `CatBoost` / `LogisticRegression` / `ElasticNet` / `LGBMRanker` / `XGBRanker`
  - `objective`: `binary` / `multiclass` / `regression` / `ranking`
  - `params`: model hyperparams (num_leaves, learning_rate, reg_lambda, etc.)

- `training`:
  - Early stopping, num_boost_round, eval_at, etc.

- `output`:
  - `results_path`: experiment ledger (usually `results.json`)
  - `runs_dir`: run artifacts directory (e.g. `runs/`)
  - `save_oof` / `save_model`: whether to save OOF and model artifacts

**How the agent may safely modify `baseline.yaml`**:

- **Allowed**:
  - `features.flags` / `features.params` (toggles and params)
  - `model.name` / `model.params` (within `search_space.yaml` model bounds)
  - `training` (early stopping / num_boost_round, etc.; record in params.json)

- **Forbidden**:
  - Changing `task.primary_metric` definition (unless fixing a bug)
  - `cv` block (if changing, update `doc/03_cv_strategy.md` and document in notes)
  - Using test set for tuning (violates AGENT_RULES)

---

## 2. `search_space.yaml` — Explorable space and guardrails

**Purpose**:

- Tell the agent which dimensions are searchable and their ranges
- Define keep / discard thresholds (improve_threshold, tie_margin, etc.)

**Core sections**:

- `policy`:
  - `improve_threshold`: how much improvement counts as "significant" per task (classification AUC / ranking NDCG@10 / regression RMSE)
  - `tie_margin`: difference within this range vs best run = tie
  - `std_worsen_ratio_max`: max allowed CV std increase (default 1.2 = 20%)
  - `resources_limits`: train/infer time, memory, model size limits

- `allowed.models_by_task`:
  - List of model names per `task.family`
  - Agent must stick to this whitelist when changing `model.name`

- `allowed.feature_flags`:
  - Boolean ranges for each `features.flags.*`
  - Agent usually changes 1–2 flags per run for attribution

- `search.hyperparams`:
  - Hyperparam space and types per model (int / uniform / log_uniform)
  - Agent may do grid / random / Bayesian search within bounds but not beyond

- `guardrails`:
  - `forbidden_changes`: explicitly forbidden (change metric definition, use test for tuning, change CV silently)
  - `require_docs_update_if_changed`: fields (e.g. `cv`, `features.flags`, `model.name`) that require doc update when changed

**How the agent uses `search_space.yaml`**:

- When picking the next experiment:
  - Choose model family from `allowed.models_by_task`
  - Pick flags to flip from `allowed.feature_flags` (e.g. `use_featuretools` false→true)
  - Sample new hyperparam combo from `search.hyperparams.<ModelName>`

- When deciding KEEP / DISCARD:
  - Compare run primary metric vs best baseline with `policy.improve_threshold` / `tie_margin` / `std_worsen_ratio_max`
  - If `resources_limits` or `guardrails` violated, mark DISCARD even if score is better

---

## 3. Suggested usage for OpenClaw / Agent

1. **Read `configs/baseline.yaml` and `configs/search_space.yaml`**:
   - Parse task family, target, metric
   - Confirm `data.train_path` / `test_path` and id / drop columns

2. **Decide which dimensions to change this run**:
   - E.g.: only change `features.flags.use_featuretools` (baseline → add auto feature)
   - Or keep features fixed, only change `LightGBM` `learning_rate`, `num_leaves`

3. **Produce new config and run**:
   - Merge changes into baseline config (do not overwrite original YAML; combine in code)
   - Write snapshot to `runs/<run_id>/params.json`

4. **Decide KEEP / DISCARD per `search_space.policy`**:
   - Append one record to `results.json` with:
     - Config summary used
     - Primary / secondary metrics (incl. per-fold)
     - decision.status and decision.reason

This README serves as the "config spec" for OpenClaw / Auto-ML agents:

- Which fields may be safely changed and their ranges
- Which fields are hard protocol/metric/CV constraints and require doc update
- How to map a single experiment's changes to `runs/<run_id>/params.json` and `results.json`
