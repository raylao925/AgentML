#!/usr/bin/env python3
"""sync_project.py — flow skill-payload fixes OUT to existing projects (SSoT rule R0).

Why: policy/template code lives in the skill, state lives in the project. Fixing the
payload must not require re-scaffolding (that would wipe data/ and runs/), and anything
validated inside a project must be able to flow back in.

Always overwritten: AGENT_RULES.md, program.md, README.md, src/*.py, configs/*
   ({{PROJECT_NAME}} is substituted with the project slug while copying)
Never touched    : data/, runs/, plots/, deliverables/, results.json (unless --reset-ledger)
Opt-in           : --include-docs (doc/*.md), --include-memory (memory/*.md)

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


def sync_project(project: Path, include_docs: bool, include_memory: bool, dry_run: bool, reset_ledger: bool) -> dict:
    stats = {"add": 0, "update": 0, "same": 0, "skipped": 0}
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
        stats[copy_one(src, project / rel, project.name, dry_run)] += 1

    if reset_ledger:
        ledger = project / "results.json"
        if not dry_run:
            ledger.write_text("[]\n", encoding="utf-8")
        print(f"  ledger reset -> {ledger}")

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync project scaffolding from the skill payload.")
    ap.add_argument("--project", default=None, help="Project slug under projects/")
    ap.add_argument("--all", action="store_true", help="Sync every project under projects/")
    ap.add_argument("--include-docs", action="store_true", help="Also overwrite doc/*.md")
    ap.add_argument("--include-memory", action="store_true", help="Also overwrite memory/*.md")
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
        stats = sync_project(project, args.include_docs, args.include_memory, args.dry_run, args.reset_ledger)
        print("   " + "  ".join(f"{k}={v}" for k, v in stats.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
