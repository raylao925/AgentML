# AgentML Project Bootstrap Workflow

This reference maps the repository's root `SKILL.md` and this skill's template payload (`assets/project-template/`) into a concrete project creation workflow.

## Stage 1: Scaffold Creation
- Copy `assets/project-template/` into `projects/<project_slug>/`.
- Preserve key files:
  - `program.md`
  - `AGENT_RULES.md`
  - `configs/baseline.yaml`
  - `configs/search_space.yaml`
  - `src/*.py`
  - `doc/*.md`
  - `memory/MEMORY.md`
  - `memory/debugging.md`
  - `memory/ITERATIONS.md` · `memory/WINS.md` · `memory/NEXT.md` · `memory/FAILURES.md`
    (cross-session state schemas — an existing project copy of these is never overwritten;
    see `scripts/sync_project.py`)

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
- Restore session memory (`01_lifecycle.md` Step 0.5): read `memory/NEXT.md` → `WINS.md` →
  `FAILURES.md` → `ITERATIONS.md` → `MEMORY.md`.
- Read `program.md`.
- Enforce `AGENT_RULES.md`.
- Confirm CV authority in `doc/04_cv_strategy.md` (lock-in mode).

## Stage 5: Readiness Gate
Project is ready when:
- scaffold exists,
- runtime folders exist,
- docs placeholders are minimally customized,
- memory files exist and the session-start restore has been performed,
- optional environment setup completed or explicitly skipped.

## Branching Logic
- Existing project slug:
  - stop unless force overwrite is approved.
- Missing `uv`:
  - scaffold only and provide manual environment steps.
- Empty problem/data docs:
  - continue with template defaults and schedule Data Wide Search in the first modeling session.
