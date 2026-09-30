# templates/ — MOVED (Phase 0)

**This directory is no longer the template.** The project template payload is now owned by the
skill, so there is exactly one copy:

```
.agents/skills/agentml/assets/project-template/
```

Scaffold a new project with:

```bash
python ./.agents/skills/agentml/scripts/init_agentml_project.py <project_slug>
# or, PowerShell:
./.agents/skills/agentml/scripts/init-agentml-project.ps1 <project_slug>
```

The skill scripts still fall back to a root-level `templates/` for older clones, but do not
recreate this folder — putting payload files back here re-introduces the dual-source drift
that this change removed.

See `.agents/skills/agentml/SKILL.md` and `.agents/skills/agentml/references/workflow.md`.
