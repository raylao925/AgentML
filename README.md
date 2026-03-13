# ClawML

ClawML is an **OpenClaw-powered** framework for running AutoML as a structured, reproducible workflow.
Each dataset/problem lives in an isolated project folder, governed by Markdown documents and logged via a machine-readable experiment ledger.

---

## What ClawML Provides

- **Per-dataset Project Isolation**: each dataset/problem becomes a standalone project folder with its own docs, configs, runs, and ledger.
- **Markdown-driven Workflow**: consistent research structure across projects:
  - Problem Statement → Data Card → EDA → CV Strategy → Modeling → Ensemble → Deployment/Submission
- **Autonomous Execution (Automation Level C)**:
  - the agent reads `program.md` and can automatically modify **config / feature / model**
- **Reproducible Experiments**:
  - CV strategy is treated as authoritative and **locked** once established
  - all runs produce consistent artifacts and structured logs
- **Structured Ledger**:
  - all runs append to `results.json` for easy filtering, ranking, and automation

---

## Project Structure (Agent-Friendly)

Each dataset/problem lives in its own project folder:

```text
projects/<project_slug>/
  program.md                 # Agent protocol (autonomous loop)
  AGENT_RULES.md             # Non-negotiable rules (leakage/CV/test/logging)
  README.md                  # Project overview (human-friendly)

  docs/
    00_problem_statement.md  # Task definition (tabular/time-series/ranking/multiclass/binary)
    01_data_card.md          # Data schema + leakage checklist
    02_eda.md                # EDA template (insights → actions)
    03_cv_strategy.md        # CV authority (lock-in after established)
    04_modeling.md           # Modeling + params + CV procedure + ablations
    05_ensemble.md           # Ensemble design (OOF-safe stacking/blending)
    06_experiment_log.md     # Ledger spec (results.json schema + keep/discard)
    07_deployment_or_submission.md  # Inference/deploy/submission

  configs/
    baseline.yaml            # Runnable baseline (starting point)
    search_space.yaml        # Allowed search boundary (models/params/feature flags)

  src/
    data.py                  # load/clean/split (must follow docs/03_cv_strategy.md)
    features.py              # feature engineering (must be fold-safe)
    train.py                 # training entry (reads configs)
    evaluate.py              # metrics computation (definition fixed)
    infer.py                 # inference/submission

  data/
    raw/                     # raw data (usually gitignored)
    interim/
    processed/

  runs/
    <run_id>/
      params.json            # effective config snapshot
      metrics.json           # per-fold + aggregate metrics
      notes.md               # hypothesis/change/outcome/decision/next step
      artifacts/             # model, oof preds, feature list, etc.
      plots/                 # optional plots

  results.json               # append-only experiment ledger
```

---

## Quick Start

### 1) Create a New Project
Copy the project template into a new folder:

```bash
cp -r templates/. projects/<project_slug>/
```

### 2) Fill in Authoritative Docs
At minimum, complete:

- `docs/00_problem_statement.md`
- `docs/01_data_card.md`

For CV, you have two options:
- **User-specified**: fill `docs/03_cv_strategy.md` explicitly (recommended for strict control)
- **Auto-infer + Lock-in**: leave it missing/empty and let the agent infer from EDA/`df.info()` and write it once (see “CV Bootstrapping”)

### 3) Run Baseline (Manual)
Run the baseline config once to generate the first ledger entry:

```bash
python projects/<project_slug>/src/train.py --config projects/<project_slug>/configs/baseline.yaml
```

This produces:
- `projects/<project_slug>/runs/<run_id>/...`
- `projects/<project_slug>/results.json` (appended)

### 4) Run Autonomous Loop (OpenClaw)
Use OpenClaw to execute the project protocol:

```bash
# placeholder: adjust to your OpenClaw runner syntax
openclaw run projects/<project_slug>/program.md
```

The agent will:
1) read `program.md` + `docs/*` + `configs/*`
2) propose and apply a bounded change (config/feature/model) within `search_space.yaml`
3) train + evaluate using the locked CV strategy
4) append a record to `results.json`
5) mark keep/discard and iterate

---

## CV Bootstrapping (Auto-infer + Lock-in)

If the user did not specify CV and `docs/03_cv_strategy.md` is missing/empty:

1) the agent performs minimal data understanding (EDA / `df.info()` / schema scan)
2) infers a safe CV strategy (time-based vs group-based vs stratified, etc.)
3) **writes it to `docs/03_cv_strategy.md` and locks it**
4) all subsequent runs must follow it exactly

Once locked, the agent must not silently change:
- Group-based CV → random CV
- time-based split → random split
- ranking query integrity rules

See `AGENT_RULES.md` for the full authority and lock-in policy.

---

## Experiment Tracking

### Ledger
- `projects/<project_slug>/results.json` (append-only)

### Artifacts per Run
- `projects/<project_slug>/runs/<run_id>/`
  - `params.json` (effective config snapshot)
  - `metrics.json` (per-fold + aggregate)
  - `notes.md` (hypothesis/change/outcome/decision)
  - `artifacts/` (model, OOF predictions, feature list)
  - `plots/` (optional)

The ledger enables:
- ranking best `decision == "keep"` runs
- auditing changes (notes + params snapshot)
- automated keep/discard based on defined rules

---

## Conventions & Guardrails

- **No test set for tuning**: test is for final reporting/submission only.
- **Fold-safe preprocessing**: fit on train fold, transform valid fold.
- **Attribution-friendly iteration**: 1–2 changes per run (feature family OR param tweak OR model swap).
- **Append-only logs**: all runs (including failures) must write to `results.json`.

---

## Project Index

See `PROJECTS.md` for a list of all projects.