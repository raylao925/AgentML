# 00 — Contract (Folders, Files, CLI)

> **SSoT**: this file is the single owner of the *directory / file / CLI* contract.
> Root `structure.md` and `projects.md` only link here; project `doc/*` never re-states it.

## 1) Directory contract (project instance)

```text
projects/<project_slug>/
  project.yaml                    # skill/policy contract (mode, versions, locks)
  program.md                      # agent autonomous loop (rendered from skill)
  AGENT_RULES.md                  # hard policy (CV authority, leakage, test, metric)
  README.md                       # human overview (task placeholders)

  doc/                            # FIXED numbering 00..08 (see below)
  configs/
    baseline.yaml                 # runnable starting config
    search_space.yaml             # agent-explorable boundary + keep/discard policy
  src/                            # AGENT-WRITABLE zone (see 02_policy §4)
    data.py  eda.py  features.py  models.py
    train.py  train_multi_model.py  evaluate.py
    ensemble.py  infer.py  infer_ensemble.py
    ingest.py  deliver.py        # data entry / delivery adapters
  memory/  (MEMORY.md, debugging.md, eda_report.md)
  data/    (raw/ interim/ processed/)      # raw is gitignored
  runs/<run_id>/ (params.json metrics.json notes.md artifacts/ plots/)
  results.json                    # append-only ledger (JSON array)
  reports/                        # data_manifest.json, eda outputs
  deliverables/                   # scored output (batch_scoring / submission)
```

## 2) `doc/` numbering is frozen (R3/R4)

| # | file | role |
|---|---|---|
| 00 | `00_problem_statement.md` | task definition + success criteria |
| 01 | `01_data_card.md` | schema, leakage checklist, PII |
| 02 | `02_eda.md` | EDA conclusions |
| 03 | `03_features_engineering.md` | **feature registry** + fold-safe policy |
| 04 | `04_cv_strategy.md` | **CV authority** (lock-in, `locked_hash`) |
| 05 | `05_ensemble.md` | ensemble design (OOF stacking/blending) |
| 06 | `06_experiment_log.md` | ledger spec (results.json schema, keep/discard) |
| 07 | `07_modeling.md` | models, hyperparams, CV procedure, ablation |
| 08 | `08_deployment_or_submission.md` | delivery: submission_csv / batch_scoring / api_contract |

Each file's H1 **must** start with its two-digit number. `check_ssot.py` enforces this.

## 3) CLI contract (single entrypoint: `agentml`)

```bash
python .agents/skills/agentml/scripts/agentml.py new <slug> [--mode customer|kaggle] [--setup-venv]
python .agents/skills/agentml/scripts/agentml.py sync --project <slug> [--all] [--include-docs]
python .agents/skills/agentml/scripts/agentml.py doctor [--project <slug>]
python .agents/skills/agentml/scripts/agentml.py ingest --source data_sources/<name>.yaml [--project <slug>]
python .agents/skills/agentml/scripts/agentml.py eda [--project <slug>]
python .agents/skills/agentml/scripts/agentml.py cv-lock [--project <slug>]
python .agents/skills/agentml/scripts/agentml.py run [--project <slug>] [--config configs/baseline.yaml] [--models ...] [--run-id ...]
python .agents/skills/agentml/scripts/agentml.py ensemble --run_ids a,b [--method weighted_average] [--project <slug>]
python .agents/skills/agentml/scripts/agentml.py infer --run_id <id> [--ensemble] [--out path.csv] [--project <slug>]
python .agents/skills/agentml/scripts/agentml.py deliver --run_id <id> [--mode batch_scoring] [--threshold 0.5]
python .agents/skills/agentml/scripts/agentml.py ledger verify|best|report [--project <slug>]
python .agents/skills/agentml/scripts/agentml.py guardrails [--project <slug>]
python .agents/skills/agentml/scripts/agentml.py check-ssot [--project <slug>]
```

Underlying scripts (`src/*.py`, `scripts/*.py`) stay runnable directly; `agentml` only
dispatches, so older command snippets keep working.

## 4) Evidence contract (`runs/`)

Each run **must** produce `params.json`, `metrics.json`, `notes.md`, and
`artifacts/` (model, OOF, feature list). Every run appends one record to `results.json`.