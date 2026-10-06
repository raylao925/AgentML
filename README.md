# AgentML

> [English](README.md) | [中文](README.zh-CN.md)

AgentML is an **agent-skill-driven** framework (Skill + Template) for running AutoML as a structured, reproducible workflow.
Each dataset/problem lives in an isolated project folder, governed by Markdown documents and logged via a machine-readable experiment ledger.

---

## Repo Overview

```text
AgentML/
  README.md           # This file (English)
  README.zh-CN.md     # Chinese version
  SKILL.md            # Skill index: project setup, CV, leakage, Data Wide Search, training, etc.
  structure.md        # Project folder layout (agent-friendly)
  projects.md         # Project index
  requirements.txt    # Base dependencies (numpy, pandas, sklearn, lightgbm, etc.)

  .agents/skills/agentml/          # THE skill — single source of truth
    SKILL.md                       #   skill entry: outcome / when-to-use / decision points
    references/                    #   00_contract .. 09_guardrails (+ sources/kaggle.md)
    scripts/                       #   agentml.py (unified CLI) | guardrails.py | check_ssot.py
                                   #   sync_project.py | init_agentml_project.py | .ps1
    assets/project-template/       #   THE template payload (copied to projects/<slug>/)
      project.yaml                 #   skill/policy version + project mode (kaggle | customer)
      AGENT_RULES.md  program.md  README.md  results.json
      configs/         # baseline.yaml, search_space.yaml
      data_sources/    # ingestion specs (local_files / kaggle_competition / database ...)
      doc/             # 00-08 markdown templates
      src/             # data.py, eda.py, features.py, models.py, train.py,
                       # train_multi_model.py, evaluate.py, tune.py, ensemble.py,
                       # infer.py, infer_ensemble.py, ingest.py, deliver.py
      memory/          # MEMORY.md, ITERATIONS.md, WINS.md, NEXT.md, FAILURES.md,
                       # debugging.md, eda_report.md

  tests/              # smoke test (train -> evaluate -> ensemble on synthetic data)
  .github/workflows/  # CI: check-ssot + guardrails --payload + unittest

  projects/           # Per-dataset project folders (see structure.md; gitignored)
    <project_slug>/
```

---

## What AgentML Provides

- **Per-dataset Project Isolation**: each dataset/problem becomes a standalone project folder with its own docs, configs, runs, and ledger.
- **Markdown-driven Workflow**: consistent research structure across projects:
  - Problem Statement → Data Card → EDA → CV Strategy → Modeling → Ensemble → Deployment/Submission
- **Autonomous Execution (Agent Automation)**: the agent reads `program.md` and can automatically modify **config / feature / model** within `search_space.yaml`.
- **Data Wide Search**: when the user **only drops a dataset** without docs, the agent can perform web/domain search to bootstrap context and strengthen feature engineering (see below).
- **Reproducible Experiments**: CV strategy is treated as authoritative and **locked** once established; all runs produce consistent artifacts and structured logs.
- **Structured Ledger**: all runs append to `results.json` for easy filtering, ranking, and automation.
- **Skill + Template Ownership**: the skill (`.agents/skills/agentml/`) owns all policy and template code; each project folder holds only its own state (docs, data, runs). Policy changes are made once, in the skill, and never re-copied by hand.

---

## Project Structure (Agent-Friendly)

Each dataset/problem lives in its own project folder:

```text
projects/<project_slug>/
  program.md                 # Agent protocol (autonomous loop)
  AGENT_RULES.md             # Non-negotiable rules (leakage/CV/test/logging)
  README.md                  # Project overview (human-friendly)
  project.yaml               # skill/policy version + mode + created timestamp

  memory/                    # Cross-session memory (state + context)
    MEMORY.md                # Context index (goals, decisions, env notes)
    ITERATIONS.md            # Append-only round log
    WINS.md                  # KEEP table + current best
    NEXT.md                  # Live queue (authoritative over doc/06 §G)
    FAILURES.md              # DISCARD table + frozen families
    debugging.md             # Program errors only
    eda_report.md            # Generated EDA write-up

  doc/
    00_problem_statement.md  # Task definition (tabular/time-series/ranking/multiclass/binary)
    01_data_card.md          # Data schema + leakage checklist
    02_eda.md                # EDA conclusions (report artifact: memory/eda_report.md)
    03_features_engineering.md  # Feature registry + fold-safe policy
    04_cv_strategy.md        # CV authority (lock-in after established)
    05_ensemble.md           # Ensemble design (OOF-safe stacking/blending)
    06_experiment_log.md     # Ledger spec (results.json schema + keep/discard)
    07_modeling.md           # Modeling + params + CV procedure + ablations
    08_deployment_or_submission.md  # Inference/deploy/submission

  configs/
    baseline.yaml            # Runnable baseline (starting point)
    search_space.yaml        # Allowed search boundary (models/params/feature flags)

  data_sources/
    <spec>.yaml              # ingestion spec (source_type + target + pii_cols); see references/04_ingestion.md

  src/
    data.py                  # load/clean/split (must follow doc/04_cv_strategy.md)
    ingest.py                # customer data ingestion: raw -> data/processed + reports/data_manifest.json
    eda.py                   # EDA report generator (writes memory/eda_report.md)
    features.py              # feature engineering (must be fold-safe)
    models.py                # model factory (LightGBM/XGBoost/CatBoost/LogReg/... presets)
    train.py                 # single-model CV training entry (reads configs)
    train_multi_model.py     # multi-model CV training + comparison
    evaluate.py              # metrics computation (definition fixed)
    ensemble.py              # ensemble over runs (OOF-safe)
    infer.py                 # inference/submission
    infer_ensemble.py        # ensemble inference (hill-climbing weights)
    deliver.py               # delivery artifacts: submission_csv / batch_scoring / api_contract

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
Scaffold from the skill's template payload (preferred), or copy it manually:

```bash
# preferred: unified CLI (creates data/ + runs/, fills {{PROJECT_NAME}}, stamps project.yaml)
python .agents/skills/agentml/scripts/agentml.py new <project_slug> --mode kaggle

# equivalent: skill script directly
python ./.agents/skills/agentml/scripts/init_agentml_project.py <project_slug>

# manual equivalent
mkdir -p projects/<project_slug>/
cp -r .agents/skills/agentml/assets/project-template/. projects/<project_slug>/
```

### 2) Fill in Authoritative Docs (or Use Data Wide Search)

**Option A — Full control**: At minimum, complete `doc/00_problem_statement.md` and `doc/01_data_card.md`.

**Option B — Minimal setup**: Drop your train/test data into `data/raw/` or `data/processed/` and run the agent. It will perform Data Wide Search to infer context, propose features, and interact with you to confirm (see "Data Wide Search" below).

For CV, you have two options:
- **User-specified**: fill `doc/04_cv_strategy.md` explicitly (recommended for strict control)
- **Auto-infer + Lock-in**: leave it missing/empty and let the agent infer from EDA/`df.info()` and write it once (see “CV Bootstrapping”)

### 3) Run Baseline (Manual)
Run the baseline config once to generate the first ledger entry:

```bash
python projects/<project_slug>/src/train.py --config projects/<project_slug>/configs/baseline.yaml
```

This produces:
- `projects/<project_slug>/runs/<run_id>/...`
- `projects/<project_slug>/results.json` (appended)

### 4) Run the Autonomous Loop
Hand the project to the agent (skill-driven). The protocol the agent follows is
`program.md` + `AGENT_RULES.md`, with mechanism details in
`.agents/skills/agentml/references/`.

```text
"run projects/<project_slug>/program.md"
```

The agent will:
1) read `program.md` + `doc/*` + `configs/*`
2) propose and apply a bounded change (config/feature/model) within `search_space.yaml`
3) train + evaluate using the locked CV strategy
4) append a record to `results.json`
5) mark keep/discard and iterate

---

## Unified CLI

Every skill script is reachable through one entry point (the individual scripts stay callable):

```bash
python .agents/skills/agentml/scripts/agentml.py <command> [options]
```

| Command | Purpose |
|---|---|
| `new <slug> --mode kaggle\|customer` | scaffold `projects/<slug>/` from the payload and stamp `project.yaml` |
| `sync --project <slug> [--dry-run]` | re-sync templates from the payload (never overwrites project state) |
| `doctor [--project <slug>]` | preflight: paths, config, deps, label/dtype sanity |
| `ingest --source data_sources/<spec>.yaml` | raw → `data/processed/*` (PII hashing, target binarization, `reports/data_manifest.json`) |
| `eda --project <slug>` | write `memory/eda_report.md` |
| `cv-lock --project <slug>` | freeze the CV strategy and record `locked_hash` in `doc/04_cv_strategy.md` |
| `run` / `ensemble` / `infer` | train (single / multi-model), ensemble, inference |
| `deliver --run_id <id> --mode submission_csv\|batch_scoring\|api_contract` | customer-facing delivery artifacts |
| `ledger` | inspect / verify `results.json` |
| `guardrails [--project <slug>] [--payload]` | executable G1–G9 policy checks (the CI gate) |
| `check-ssot` | verify payload/document numbering consistency |

`agentml.py <command> --help` prints the underlying script flags. CI
(`.github/workflows/ci.yml`) runs `check-ssot` + `guardrails --payload` + `unittest`
on Python 3.11 and 3.12.

---

## CV Bootstrapping (Auto-infer + Lock-in)

If the user did not specify CV and `doc/04_cv_strategy.md` is missing/empty:

1) the agent performs minimal data understanding (EDA / `df.info()` / schema scan)
2) infers a safe CV strategy (time-based vs group-based vs stratified, etc.)
3) **writes it to `doc/04_cv_strategy.md` and locks it**
4) all subsequent runs must follow it exactly

Once locked, the agent must not silently change:
- Group-based CV → random CV
- time-based split → random split
- ranking query integrity rules

See `AGENT_RULES.md` for the full authority and lock-in policy.

---

## Data Wide Search (Minimal Context Bootstrapping)

When the user **only drops a dataset** into `projects/<project_slug>/data/` without filling `doc/00_problem_statement.md` or `01_data_card.md`, the agent can bootstrap context via **wide search**:

1. **Infer from data**: Run `df.info()`, schema scan, sample rows; identify likely target, ID, datetime columns; guess task type.
2. **Web / domain search**: Search for similar problems, Kaggle notebooks, domain best practices; collect feature ideas (ratios, aggregations, time transforms, encodings).
3. **User–agent interaction**: Summarize findings; propose inferred target, task type, and feature candidates; ask user to confirm or correct.
4. **Document and implement**: Write inferred content into `doc/*`; add the "Data Wide Search" subsection to `doc/07_modeling.md`; implement fold-safe features in `src/features.py`.

See `SKILL.md` §1.5 and `.agents/skills/agentml/assets/project-template/doc/07_modeling.md` §2.3 for the full protocol.

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
- **Enforced, not just documented**: `agentml guardrails` (G1–G9) mechanically checks the rules above (target leakage, CV lock hash, test usage in tuning, ledger schema, placeholders, fold integrity); CI fails the build if any check fails.

---

## Project Index

See `projects.md` for a list of all projects.