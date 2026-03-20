# AgentML Framework Skills

## 0) Project Environment Management (uv)
Because each project may choose different ML/deep-learning libraries (based on `task.target`, `metric`, and the model/feature search space),
AgentML uses `uv` to create a dedicated virtual environment per new project.

When a new folder `projects/<project_slug>/` is created (copy from `templates/`):
1) Ensure `uv` is installed on the host machine (one-time).
2) Create a project-local venv (recommended path: `projects/<project_slug>/.venv`):
   - Use a fixed python version if the runner supports it.
   - Example:
     - `uv venv --python 3.11 .venv`
     - `uv pip install -r <repo_root>/requirements.txt`
3) (Optional / recommended for smaller environments) Install only required extras by scanning configs/docs:
   - If `configs/baseline.yaml` or candidate `model.name` includes:
     - `LightGBM` => ensure `lightgbm` installed
     - `XGBoost`   => ensure `xgboost` installed
     - `CatBoost`  => ensure `catboost` installed
   - If feature flags include:
     - `use_featuretools: true` => ensure `featuretools` installed
     - `use_pycaret: true` or agent enabled pycaret => ensure `pycaret` installed
4) Activate the env when running any python command for that project.

Output: a usable interpreter for that project so `python src/train.py` / `evaluate.py` / `infer.py` runs inside the venv.

This document lists the core *agent skills* AgentML expects a runner / OpenClaw agent to perform.
Each skill is written in a way that can be mapped to an implementation module, prompt step,
or automated checklist.

---

## 1) Project Discovery & Contract Reading
- Read project folder structure from `structure.md`.
- Load authoritative research protocol from `program.md`.
- Parse constraints from `AGENT_RULES.md` (no leakage, CV authority, test restrictions, metric definition fixed).
- Load task specs:
  - `docs/00_problem_statement.md`
  - `docs/01_data_card.md`
  - `docs/03_cv_strategy.md` (if present/locked)

Output: a structured “run plan” object containing task family, target, primary metric, CV policy, and allowed change dimensions.

---

## 1.5) Data Wide Search (Minimal Context Bootstrapping)

When the user has **only dropped a dataset** into `projects/<project_slug>/data/` without filling `docs/00_problem_statement.md`, `docs/01_data_card.md`, or other docs, the agent should perform **wide search** to bootstrap context and strengthen feature engineering.

### Triggers
- `docs/00_problem_statement.md` or `01_data_card.md` is empty, placeholder-only, or missing.
- User explicitly asks: "help me understand this data" / "suggest features" / "what can I do with this dataset?"

### Wide Search Steps
1) **Infer minimal context from data**:
   - Run `df.info()`, schema scan, sample rows.
   - Identify likely target column(s), ID columns, datetime columns.
   - Guess task type (binary/multiclass/regression) from target distribution.

2) **Web / domain search**:
   - If dataset name, competition slug, or domain hints exist → search for similar problems, Kaggle notebooks, domain best practices.
   - Search for: "[dataset/domain] feature engineering", "[task type] tabular tips", "common features for [domain]".
   - Collect candidate feature ideas: ratios, aggregations, time-based transforms, encoding strategies.

3) **User–agent interaction**:
   - Summarize findings and propose: inferred target, task type, and a short list of feature engineering candidates.
   - Ask user to confirm or correct: target column, positive class (if binary), domain assumptions.
   - Iterate: refine `docs/01_data_card.md` and `04_modeling.md` based on user feedback.

4) **Document and implement**:
   - Write inferred/problem-statement and data-card content into `docs/*`.
   - Add a "Data Wide Search" subsection to `docs/04_modeling.md` (see template) with discovered feature ideas.
   - Implement fold-safe features in `src/features.py` per `AGENT_RULES` (no leakage).

### Output
- Updated `docs/00_problem_statement.md`, `01_data_card.md`, and `04_modeling.md` (Data Wide Search section).
- A set of feature engineering hypotheses ready for ablation runs.

---

## 2) CV Strategy Compliance (Lock-in Mode)
- If `docs/03_cv_strategy.md` is authoritative (P1), follow it exactly.
- If CV is missing/empty (P2 auto-infer), infer safely using minimal schema/EDA evidence:
  - detect time/group/id candidates
  - detect leakage risk signals
- Write the inferred rule back to `docs/03_cv_strategy.md` and treat it as locked.

Output: a CV object (cv_type, n_splits, keys, split rules) used consistently by training + any fold-safe preprocessing.

---

## 3) Leakage Checks (Hard Guardrail)
- Drop features explicitly flagged as leak in `docs/01_data_card.md`.
- Ensure preprocessing is fold-safe:
  - fit preprocessors only on the train fold
  - compute aggregations/encodings/rollups only on train fold window
- Ensure no calculation step uses valid/test to fit encoders, scalers, or aggregations.

Output: a “leakage risk report” for the planned run (pass/fail + reasons).

---

## 4) Config-Driven Training Orchestration
- Read `configs/baseline.yaml` as the executable starting point.
- Produce new runs by applying *bounded changes* (1–2 change families per run).
- Ensure CV/split, metric definition, and test-tuning rules are not silently violated.

Output: `runs/<run_id>/params.json` (effective config snapshot).

---

## 5) Fold-safe Feature Engineering
- Build a fold-safe feature pipeline in `src/features.py`.
- Support common feature families via config flags:
  - numeric: impute + optional scaling + transforms
  - categorical: impute + one-hot or tree-friendly handling
  - target encoding: fold-safe (train-fold only / OOF-safe)
  - auto feature: `featuretools` via fold-safe transformer
  - time series: lag/rolling strictly past-only and computed within fold windows

Output: trained preprocessors per fold and a reproducible `feature_list.json` artifact.

---

## 6) Model Training with OOF Logging
- Train using CV folds from `docs/03_cv_strategy.md`.
- Produce OOF predictions per fold and store them in `runs/<run_id>/artifacts/`.
- Save all required artifacts:
  - `artifacts/model.pkl`
  - `artifacts/oof_predictions.*`
  - `artifacts/feature_list.json`
  - `artifacts/dataset_profile_train.json` (recommended) / `dataset_profile_test.json` (optional)

Output: `runs/<run_id>/metrics.json` + `runs/<run_id>/notes.md`.

---

## 7) Metric Evaluation (Fixed Definition)
- Use `src/evaluate.py` for metric computation.
- Do not change the metric definition (only bug fixes / efficiency improvements allowed).
- For binary classification, ensure target is numeric 0/1 before sklearn metrics.

Output: primary metric mean/std and per-fold diagnostics inside `metrics.json`.

---

## 8) Ledger Update (Append-only)
- Append one record per run (keep or discard) to `results.json`.
- Ensure each record contains:
  - `run_id`, `datetime`
  - task + cv + model + metrics.primary.mean
  - artifact paths
  - decision status and reason

Output: updated `results.json` ledger entry for run traceability.

---

## 9) Keep/Discard Decision
- Compare candidate run vs best baseline KEEP using `search_space.yaml` policy.
- Enforce resource limits and std-worsen thresholds.
- If discard, still log a meaningful reason (variance, speed, leakage risk, constraint violation).

Output: `decision.status` and `decision.reason` written into the ledger and/or notes.

---

## 10) Inference / Submission Generation
- Load trained `artifacts/model.pkl` and its feature schema.
- Use `data.id_cols` from config to preserve identifier columns in output submission.
- Generate `submission.csv` (CSV) with proper encoding for Excel compatibility if needed.

Output: `runs/<run_id>/artifacts/submission.csv`.

---

## 11) Ensemble Over Runs (OOF-safe)
- Read multiple runs' OOF artifacts only.
- Train meta-model / choose weights using OOF design (no leakage from test).
- Produce ensemble artifacts into a new run directory.

Output: ensemble `submission` and ensemble metrics record in `results.json`.

## 11.5) Multi-Model Training
- Train multiple models in sequence for comparison using `train_multi_model.py`.
- Supports: LightGBM, XGBoost, CatBoost, LogisticRegression, ElasticNet.
- Each model gets its own run directory with OOF predictions and metrics.
- Summary table shows all models ranked by primary metric for easy comparison.

Usage:
```bash
python src/train_multi_model.py --config configs/baseline.yaml
python src/train_multi_model.py --config configs/baseline.yaml --models "LightGBM,XGBoost"
```

Output: Multiple run directories with individual model artifacts and a comparison summary.

## 11.6) Ensemble Inference with Hill Climbing
- Load multiple trained models from an ensemble run using `infer_ensemble.py`.
- Generate combined predictions using weighted average.
- Supports `--optimize_weights` flag for hill climbing algorithm.

**Hill Climbing Algorithm**:
- Iteratively adjusts model weights to maximize AUC on OOF predictions.
- Starts with uniform weights and perturbs them randomly.
- Accepts weight changes that improve AUC, rejects those that don't.
- Converges to optimal weight combination after multiple iterations.

Usage:
```bash
python src/infer_ensemble.py --ensemble_run_id <run_id>
python src/infer_ensemble.py --ensemble_run_id <run_id> --optimize_weights
```

Output: `submission.csv` with ensemble predictions and optimized weights metadata.

---

## 12) Reproducibility & Traceability
- Fix seeds for all stochastic operations.
- Record:
  - `data_version`
  - `feature_version` / feature flags
  - `code_hash` (if you implement it)
  - config snapshot (`params.json`)

Output: deterministic enough runs to reproduce the same artifacts under same data + config.

