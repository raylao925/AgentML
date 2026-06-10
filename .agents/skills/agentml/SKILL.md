---
name: agentml
description: Build a standardized AgentML project folder from templates and root SKILL.md policy. Use when creating projects/<project_slug>, bootstrapping doc/config/src/memory layout, and setting up uv virtual environments for Kaggle or tabular ML workflows.
argument-hint: project slug and optional setup mode (scaffold-only or scaffold+venv)
user-invocable: true
---

# AgentML Standard Folder Builder

## Outcome
Create a consistent `projects/<project_slug>/` workspace that follows:
- `templates/` folder structure and starter files
- root `SKILL.md` workflow rules (CV lock-in, leakage guardrails, logging, reproducibility)
- optional `uv` virtual environment bootstrap

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
2. Create target folder from `templates/`.
3. Ensure required runtime folders exist (`data/interim`, `runs`).
4. Apply minimal placeholder replacement (project name markers).
5. If requested, create `.venv` and install base dependencies.
6. Run completion checks and report generated artifacts.

Run one of the reference scripts:
- [Python Reference Script](./scripts/init_agentml_project.py) (recommended, cross-platform)
- [PowerShell Script](./scripts/init-agentml-project.ps1) (Windows option)

Suggested commands:
```bash
# scaffold only
python ./.agents/skills/agentml/scripts/init_agentml_project.py <project_slug>

# scaffold + venv
python ./.agents/skills/agentml/scripts/init_agentml_project.py <project_slug> --setup-venv --python-version 3.11
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
- `README.md`, `program.md`, and `AGENT_RULES.md` exist
- if mode is `scaffold+venv`, `.venv` exists and requirements installation was attempted

## Notes
- This skill creates project scaffolding only; it does not train models.
- Follow the generated project's `program.md` and `AGENT_RULES.md` for all subsequent runs.
