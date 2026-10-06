# Project Structure (Agent-Friendly)

This project uses a "Markdown-driven + structured Ledger" approach to manage experiments.
Core concepts:
- `program.md` defines the research protocol (agent reads and auto-modifies config/feature/model)
- `doc/` defines the fixed research flow (Problem → Data → EDA → CV → Modeling → Ensemble → Deploy)
- `runs/<run_id>/` stores full artifacts for each experiment
- `results.json` serves as the experiment ledger (one record per run for ranking/best/keep-discard)

---

## Folder Layout

```text
projects/<project_slug>/
  program.md                 # Agent research protocol (automation core)
  AGENT_RULES.md             # Non-negotiable rules (leakage/CV/test/metric)
  README.md                  # Project overview (human-readable)

  memory/                    # Cross-session memory (schemas in the skill payload)
    MEMORY.md                # Context index (goals, decisions, env notes)
    ITERATIONS.md            # Append-only round log (one block per round)
    WINS.md                  # KEEP table + current best
    NEXT.md                  # Live queue — authoritative over doc/06 §G
    FAILURES.md              # DISCARD table + frozen families
    debugging.md             # Program errors only
    eda_report.md            # Generated EDA write-up

  doc/
    00_problem_statement.md  # Task definition (tabular/time-series/ranking/multiclass/binary)
    01_data_card.md          # Data schema + leakage checklist
    02_eda.md                # EDA conclusions (report artifact: memory/eda_report.md)
    03_features_engineering.md  # Feature registry + fold-safe policy
    04_cv_strategy.md        # CV authority (group key/time col user-defined)
    05_ensemble.md           # Ensemble design (OOF stacking/blending)
    06_experiment_log.md     # Ledger spec (results.json schema + keep/discard)
    07_modeling.md           # Modeling, hyperparams, CV procedure, ablation
    08_deployment_or_submission.md  # Inference/deploy/submission

  configs/
    baseline.yaml            # Runnable baseline (agent starting point)
    search_space.yaml        # Agent explorable boundary (models/params/feature flags)

  src/
    data.py                  # load/clean/split (must follow doc/04_cv_strategy.md)
    eda.py                   # EDA report generator (writes memory/eda_report.md)
    features.py              # Feature engineering (must be fold-safe; manual + auto feature)
    train.py                 # Training entry (reads configs, writes runs/<run_id> + results.json)
    evaluate.py              # Metric computation (fixed definition; recompute OOF metrics)
    infer.py                 # Inference/submission output (add prediction per data.id_cols)
    train_multi_model.py     # Multi-model training: train multiple models in sequence for comparison
    tune.py                  # Optuna HPO over the locked CV (writes runs/tune_* + ledger records)
    ensemble.py              # (Optional) Ensemble entry, per doc/05_ensemble.md OOF stacking/blending
    infer_ensemble.py        # Ensemble inference (weighted average / stacking meta; hill climbing)

  data/
    raw/                     # Raw data (usually gitignored)
    interim/
    processed/               # train/test parquet/csv etc.

  runs/
    <run_id>/
      params.json            # Effective config snapshot
      metrics.json           # per-fold + aggregate metrics
      notes.md               # hypothesis/change/outcome/decision
      artifacts/             # model, oof preds, feature list, etc.
      plots/                 # (Optional) plots

  results.json               # Experiment ledger (append-only; keep/discard decision basis)
