# AgentML Project Bootstrap Workflow

This reference maps the repository's root `SKILL.md` and `templates/` into a concrete project creation workflow.

## Stage 1: Scaffold Creation
- Copy `templates/` into `projects/<project_slug>/`.
- Preserve key files:
  - `program.md`
  - `AGENT_RULES.md`
  - `configs/baseline.yaml`
  - `configs/search_space.yaml`
  - `src/*.py`
  - `doc/*.md`
  - `memory/MEMORY.md`
  - `memory/debugging.md`

## Stage 2: Required Folder Checks
Ensure these folders exist after copy:
- `data/raw`
- `data/interim`
- `data/processed`
- `runs`

## Stage 3: Environment Setup (Optional)
From root `SKILL.md` section 0:
1. Create project-local environment under `.venv`.
2. Install root `requirements.txt`.
3. Optionally install model-specific extras based on config flags.

## Stage 4: Governance Alignment
Before first training run:
- Read `program.md`.
- Enforce `AGENT_RULES.md`.
- Confirm CV authority in `docs/03_cv_strategy.md` (lock-in mode).

## Stage 5: Readiness Gate
Project is ready when:
- scaffold exists,
- runtime folders exist,
- docs placeholders are minimally customized,
- optional environment setup completed or explicitly skipped.

## Branching Logic
- Existing project slug:
  - stop unless force overwrite is approved.
- Missing `uv`:
  - scaffold only and provide manual environment steps.
- Empty problem/data docs:
  - continue with template defaults and schedule Data Wide Search in the first modeling session.
