# Project Structure (Agent-Friendly)

This project uses a "Markdown-driven + structured Ledger" approach to manage experiments.
Core concepts:
- `program.md` defines the research protocol (agent reads and auto-modifies config/feature/model)
- `docs/` defines the fixed research flow (Problem → Data → EDA → CV → Modeling → Ensemble → Deploy)
- `runs/<run_id>/` stores full artifacts for each experiment
- `results.json` serves as the experiment ledger (one record per run for ranking/best/keep-discard)

---

## Folder Layout

```text
projects/<project_slug>/
  program.md                 # Agent research protocol (automation core)
  AGENT_RULES.md             # Non-negotiable rules (leakage/CV/test/metric)
  README.md                  # Project overview (human-readable)

  docs/
    00_problem_statement.md  # Task definition (tabular/time-series/ranking/multiclass/binary)
    01_data_card.md          # Data schema + leakage checklist
    02_eda.md                # EDA template (insights → actions)
    03_cv_strategy.md        # CV authority (group key/time col user-defined)
    04_modeling.md           # Modeling, hyperparams, KFold flow, ablation
    05_ensemble.md           # Ensemble design (OOF stacking/blending)
    06_experiment_log.md     # Ledger spec (results.json schema + keep/discard)
    07_deployment_or_submission.md  # Inference/deploy/submission

  configs/
    baseline.yaml            # Runnable baseline (agent starting point)
    search_space.yaml        # Agent explorable boundary (models/params/feature flags)

  src/
    data.py                  # load/clean/split (must follow docs/03_cv_strategy.md)
    features.py              # Feature engineering (must be fold-safe; manual + auto feature)
    train.py                 # Training entry (reads configs, writes runs/<run_id> + results.json)
    evaluate.py              # Metric computation (fixed definition; recompute OOF metrics)
    infer.py                 # Inference/submission output (add prediction per data.id_cols)
    ensemble.py              # (Optional) Ensemble entry, per docs/05_ensemble.md OOF stacking/blending

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
