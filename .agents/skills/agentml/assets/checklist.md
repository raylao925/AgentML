# Completion Checklist

Use this checklist at the end of scaffold creation.

- [ ] `projects/<project_slug>/` was created.
- [ ] Core files exist: `README.md`, `program.md`, `AGENT_RULES.md`, `results.json`.
- [ ] Core folders exist: `configs/`, `doc/`, `src/`, `memory/`, `data/`, `runs/`.
- [ ] Runtime data folders exist: `data/raw`, `data/interim`, `data/processed`.
- [ ] `memory/MEMORY.md` and `memory/debugging.md` exist.
- [ ] Memory state files exist: `memory/ITERATIONS.md`, `memory/WINS.md`, `memory/NEXT.md`,
      `memory/FAILURES.md` (schemas from the payload — G10 checks them once a run is logged).
- [ ] `program.md` §1 restore step and `AGENT_RULES.md` §7 are present (cross-session contract).
- [ ] Placeholder replacement applied for project name markers.
- [ ] If requested, `.venv` was created.
- [ ] If requested, dependency installation was attempted.
- [ ] User received next-step commands for first train/eval/infer run.
- [ ] Verification green: `python ./.agents/skills/agentml/scripts/check_ssot.py` and
      `python ./.agents/skills/agentml/scripts/guardrails.py --project <slug>`.
