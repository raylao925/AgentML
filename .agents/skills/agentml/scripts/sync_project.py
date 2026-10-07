#!/usr/bin/env python3
"""sync_project.py — flow skill-payload fixes OUT to existing projects (SSoT rule R0).

Why: policy/template code lives in the skill, state lives in the project. Fixing the
payload must not require re-scaffolding (that would wipe data/ and runs/), and anything
validated inside a project must be able to flow back in.

Ownership classes (payload/state manifest — Phase 1):
  PAYLOAD — skill-owned policy/framework: AGENT_RULES.md, program.md, data_sources/*,
            src/* except STATE_SRC_FILES. An existing file is overwritten
            whenever the payload differs ({{PROJECT_NAME}} is substituted while copying).
  STATE   — project-owned content: README.md, project.yaml (add-only), configs/*, doc/*,
            memory/* and the project-owned
            src modules (features.py, tune.py). An existing STATE file that already has real
            body (anything beyond headings/quotes/rule lines) is NEVER overwritten; a body-less
            file (pure template) is still refreshed from the payload, and a missing one is added.
Never touched : data/, runs/, plots/, deliverables/, reports/, results.json (unless --reset-ledger)
Opt-in        : --include-docs (doc/*.md, STATE-guarded), --include-memory (memory/*.md, STATE-guarded)
NOTE          : the base sync above ALWAYS runs — flags only ADD the optional dirs. Run with
                --dry-run first to see the classified actions (state skips/updates are listed).

Usage:
    python ./.agents/skills/agentml/scripts/sync_project.py --project <slug> [--include-docs] [--dry-run]
    python ./.agents/skills/agentml/scripts/sync_project.py --all --include-docs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PAYLOAD = REPO_ROOT / ".agents" / "skills" / "agentml" / "assets" / "project-template"
PROJECTS_DIR = REPO_ROOT / "projects"

BASE_FILES = ["AGENT_RULES.md", "program.md", "README.md", "project.yaml"]
BASE_DIRS = ["src", "configs", "data_sources"]
OPTIONAL_DIRS = {"doc": "include_docs", "memory": "include_memory"}
NEVER = {"data", "runs", "plots", "deliverables", "reports", "results.json", ".venv"}

# --- Ownership manifest (payload/state classification, Phase 1) -----------------------
# PAYLOAD class — flowed outward unconditionally (see module docstring).
# STATE class   — project-owned: an existing file with real body is never overwritten.
STATE_TOP_FILES = {"README.md", "project.yaml"}
STATE_DIRS = ("doc", "memory", "configs")
STATE_SRC_FILES = {"features.py", "tune.py"}  # project-owned src modules (X class)


def is_state(rel: Path) -> bool:
    """True when `rel` (payload-relative) belongs to the project STATE class."""
    if rel.parts and rel.parts[0] in STATE_DIRS:
        return True
    if len(rel.parts) == 1 and rel.name in STATE_TOP_FILES:
        return True
    if rel.parts[:1] == ("src",) and rel.name in STATE_SRC_FILES:
        return True
    return False


def transform(src: Path, project_slug: str) -> bytes:
    """Text payload files get {{PROJECT_NAME}} substituted with the project slug."""
    data = src.read_bytes()
    if src.suffix.lower() not in {".md", ".py", ".yaml", ".yml", ".txt", ".cfg", ".json", ".ps1"}:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    if "{{PROJECT_NAME}}" in text:
        return text.replace("{{PROJECT_NAME}}", project_slug).encode("utf-8")
    return data


def _has_body(path: Path) -> bool:
    """True when an existing file carries real content beyond headings/quotes/rule lines."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(">") or s.startswith("---"):
            continue
        return True
    return False


def copy_one(src: Path, dst: Path, project_slug: str, dry_run: bool) -> str:
    payload = transform(src, project_slug)
    if dst.exists() and dst.is_file():
        if dst.read_bytes() == payload:
            return "same"
        action = "update"
    else:
        action = "add"
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(payload)
    return action


def sync_project(project: Path, include_docs: bool, include_memory: bool, dry_run: bool, reset_ledger: bool) -> tuple[dict, list[str]]:
    stats = {"add": 0, "update": 0, "same": 0, "skipped": 0}
    state_actions: list[str] = []
    targets: list[Path] = [PAYLOAD / f for f in BASE_FILES]
    for d in BASE_DIRS:
        targets.extend(sorted(p for p in (PAYLOAD / d).rglob("*") if p.is_file()))
    if include_docs:
        targets.extend(sorted(p for p in (PAYLOAD / "doc").rglob("*.md")))
    if include_memory:
        targets.extend(sorted(p for p in (PAYLOAD / "memory").rglob("*.md")))

    for src in targets:
        rel = src.relative_to(PAYLOAD)
        if any(part in NEVER for part in rel.parts):
            stats["skipped"] += 1
            continue
        # project.yaml holds per-project state (mode/created) -> add only if missing.
        if rel.name == "project.yaml" and (project / rel).exists():
            stats["same"] += 1
            continue
        # STATE class (README, configs/*, doc/*, memory/*, project-owned src): an existing file
        # with content is NEVER overwritten — the payload only fills gaps and refreshes
        # body-less templates, so project research/config state cannot be clobbered.
        if is_state(rel) and (project / rel).exists() and _has_body(project / rel):
            stats["skipped"] += 1
            state_actions.append(f"skip(state): {rel.as_posix()}")
            continue
        action = copy_one(src, project / rel, project.name, dry_run)
        stats[action] += 1
        if is_state(rel) and action != "same":
            state_actions.append(f"{action}(state): {rel.as_posix()}")

    if reset_ledger:
        ledger = project / "results.json"
        if not dry_run:
            ledger.write_text("[]\n", encoding="utf-8")
        print(f"  ledger reset -> {ledger}")

    return stats, state_actions


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync project scaffolding from the skill payload.")
    ap.add_argument("--project", default=None, help="Project slug under projects/")
    ap.add_argument("--all", action="store_true", help="Sync every project under projects/")
    ap.add_argument("--include-docs", action="store_true", help="Also sync doc/*.md (STATE-guarded: an existing doc with content is never overwritten)")
    ap.add_argument("--include-memory", action="store_true", help="Also sync memory/*.md (STATE-guarded/add-only: a populated memory file is never overwritten; note the base payload/state sync always runs regardless of this flag)")
    ap.add_argument("--reset-ledger", action="store_true", help="Replace results.json with []")
    ap.add_argument("--dry-run", action="store_true", help="Report only, change nothing")
    args = ap.parse_args()

    if not PAYLOAD.is_dir():
        print(f"Error: payload not found: {PAYLOAD}", file=sys.stderr)
        return 1

    if args.all:
        projects = sorted(p for p in PROJECTS_DIR.glob("*") if p.is_dir())
    elif args.project:
        projects = [PROJECTS_DIR / args.project]
    else:
        print("Error: pass --project <slug> or --all", file=sys.stderr)
        return 1

    if not projects:
        print("No projects to sync.")
        return 0

    for project in projects:
        if not project.is_dir():
            print(f"Error: project not found: {project}", file=sys.stderr)
            return 1
        print(f"== {project.name}{' (dry-run)' if args.dry_run else ''}")
        stats, state_actions = sync_project(project, args.include_docs, args.include_memory, args.dry_run, args.reset_ledger)
        print("   " + "  ".join(f"{k}={v}" for k, v in stats.items()))
        for line in state_actions:
            print(f"   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
