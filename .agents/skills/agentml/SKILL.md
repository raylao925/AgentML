---
name: agentml
description: Build a standardized AgentML project folder from templates and the SSoT skill policy. Use when creating projects/<project_slug>, bootstrapping doc/config/src/memory layout, running the agentml CLI (new/ingest/eda/cv-lock/run/ensemble/infer/deliver/guardrails), and setting up uv virtual environments for Kaggle OR customer tabular ML workflows.
argument-hint: project slug and optional setup mode (scaffold-only or scaffold+venv)
user-invocable: true
---

# AgentML Standard Folder Builder

## Outcome
Create a consistent `projects/<project_slug>/` workspace that follows:
- the template payload owned by this skill (`assets/project-template/`)
- this skill's `references/` workflow rules (contract, lifecycle, policy, discovery, ingestion, features, modeling, ensemble, delivery, guardrails)
- optional `uv` virtual environment bootstrap

## Reference index (read on demand)
`references/00_contract.md` · `01_lifecycle.md` · `02_policy.md` · `03_task_discovery.md` ·
`04_ingestion.md` · `05_features.md` · `06_modeling.md` · `07_ensemble.md` ·
`08_delivery.md` · `09_guardrails.md` · `sources/kaggle.md` · `workflow.md`.

Root `SKILL.md` is the human-readable index; if it disagrees with this skill set, this wins.

## Modes
- `mode: kaggle` — competition dataset, metric/target given → follow `references/sources/kaggle.md`.
- `mode: customer` — customer dataset, target/metric must be **discovered** → `references/03_task_discovery.md` (Lifecycle Gate A).

## Payload, SSoT & Verification
- **Payload (single owner)**: `assets/project-template/` — this is the only copy of the project template.
  Older clones may still carry a legacy `templates/` directory; the scripts fall back to it if the payload is missing.
- **SSoT rule**: policy/template code lives in this skill; a project folder only holds its own state
  (`doc/`, `data/`, `runs/`, `results.json`).
- **Verifier**: run `python ./.agents/skills/agentml/scripts/check_ssot.py` after any doc/path change
  (checks doc numbering, stale documentation paths, payload presence, placeholder leakage).

## When To Use
Use this skill when the user asks to:
- create a new AgentML project quickly
- standardize project folder structure
- initialize doc/config/src/memory scaffolding
- bootstrap a per-project Python environment for training

## Required Inputs
- `project_slug`: folder name under `projects/`
- `mode`: `scaffold-only` or `scaffold+venv`
- Optional: python version (default `3.11`)

## Procedure
1. Validate repository prerequisites.
2. Resolve the template payload (`assets/project-template/`, falling back to a legacy `templates/`) and copy it into the target folder.
3. Ensure required runtime folders exist (`data/interim`, `runs`).
4. Apply minimal placeholder replacement (project name markers) and stamp `project.yaml`
   with `project.mode` and a creation date.
5. If requested, create `.venv` and install base dependencies.
6. Run completion checks and report generated artifacts.

Run the unified CLI (recommended) or a specific script:

```bash
# unified entrypoint (see references/00_contract.md §3)
python ./.agents/skills/agentml/scripts/agentml.py new <project_slug> --mode customer
python ./.agents/skills/agentml/scripts/agentml.py new <project_slug> --setup-venv

# direct scaffold script
python ./.agents/skills/agentml/scripts/init_agentml_project.py <project_slug> --setup-venv --python-version 3.11
```

Verify after any change:
```bash
python ./.agents/skills/agentml/scripts/check_ssot.py     # structure/anti-drift
python ./.agents/skills/agentml/scripts/agentml.py guardrails --project <slug>
```

Use supporting docs:
- [Workflow Reference](./references/workflow.md)
- [Checklist](./assets/checklist.md)

## Decision Points
- If `projects/<project_slug>/` already exists:
  - default behavior: stop and ask user before overwrite
  - optional behavior: overwrite only with explicit force flag
- If `uv` is unavailable:
  - keep scaffold
  - report environment setup command for manual execution
- If doc files are missing task specifics:
  - keep template placeholders and continue
  - later run Data Wide Search workflow from root `SKILL.md` section 1.5

## Completion Criteria
A run is complete only when:
- target folder exists with expected `doc/`, `configs/`, `src/`, `memory/`
- `data/raw`, `data/interim`, `data/processed`, and `runs/` are present
- `README.md`, `program.md`, `AGENT_RULES.md`, and `project.yaml` exist
- `project.yaml` declares `project.mode` + `skill.version` + `policy.version`
- if mode is `scaffold+venv`, `.venv` exists and requirements installation was attempted
- `python ./scripts/check_ssot.py` passes for the new project

## Notes
- This skill creates project scaffolding only; it does not train models.
- Follow the generated project's `program.md` and `AGENT_RULES.md` for all subsequent runs.
