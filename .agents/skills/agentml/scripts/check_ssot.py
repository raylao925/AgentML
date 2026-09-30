#!/usr/bin/env python3
"""check_ssot.py — AgentML SSoT verifier (Phase 0).

Single Source Of Truth rules enforced here:

  R1  The project-template payload lives inside this skill: assets/project-template/.
  R2  `doc/` is the canonical documentation directory name (a legacy "docs" directory must not be referenced).
  R3  Doc numbering is fixed: doc/00..08, and each file's H1 must carry the same number.
  R4  Every project under projects/ mirrors the payload doc set + numbering.
  R5  No live file points at the legacy `templates/` payload path.
  R6  `results.json` is a JSON array with no placeholder/example run records.
  R7  Projects contain no unresolved {{PLACEHOLDER}} tokens.

Usage:
    python ./.agents/skills/agentml/scripts/check_ssot.py [--project SLUG] [--quiet]
Exit code: 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# .agents/skills/agentml/scripts/check_ssot.py -> parents[4] == repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "agentml"
PAYLOAD = SKILL_DIR / "assets" / "project-template"
PROJECTS_DIR = REPO_ROOT / "projects"

EXPECTED_DOCS = [
    "00_problem_statement",
    "01_data_card",
    "02_eda",
    "03_features_engineering",
    "04_cv_strategy",
    "05_ensemble",
    "06_experiment_log",
    "07_modeling",
    "08_deployment_or_submission",
]

DOC_RE = re.compile(r"^\s*#\s*(\d{2})\s*[\u2014\-\u2013]")
PLACEHOLDER_RE = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}")
# R2: a reference to a documentation *file* under the legacy "docs" directory is a hard
# error; a bare mention of that directory name is only a warning.
STALE_DOCS_PATH_RE = re.compile(r"\bdocs/[A-Za-z0-9_*]")
STALE_DOCS_BARE_RE = re.compile(r"\bdocs/")
SKIP_DIRS = {".git", "__pycache__", "catboost_info", ".venv", "node_modules", ".ipynb_checkpoints"}
TEXT_EXT = {".md", ".py", ".yaml", ".yml", ".ps1", ".json", ".ipynb", ".txt", ".cfg", ".toml"}

problems: list[str] = []
notes: list[str] = []


def add(msg: str) -> None:
    problems.append(msg)


def iter_text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in TEXT_EXT:
            yield path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def check_r1_payload() -> None:
    if not PAYLOAD.is_dir():
        add(f"R1: template payload missing -> {rel(PAYLOAD)}")
        return
    for required in ["AGENT_RULES.md", "program.md", "README.md", "results.json", "doc", "configs", "src"]:
        if not (PAYLOAD / required).exists():
            add(f"R1: payload missing '{required}'")
    if (REPO_ROOT / "templates").exists():
        notes.append("legacy root 'templates/' still exists (allowed as a shim; remove once all clones migrate)")


def check_doc_set(doc_dir: Path, label: str) -> None:
    if not doc_dir.is_dir():
        add(f"R3/R4: {label}: doc/ directory missing")
        return
    present = sorted(p.stem for p in doc_dir.glob("*.md"))
    expected = sorted(EXPECTED_DOCS)
    if present != expected:
        missing = [d for d in expected if d not in present]
        extra = [d for d in present if d not in expected]
        if missing:
            add(f"R3/R4: {label}: missing docs {missing}")
        if extra:
            add(f"R3/R4: {label}: unexpected docs {extra}")
    for path in sorted(doc_dir.glob("*.md")):
        prefix = re.match(r"^(\d{2})_", path.name)
        h1 = next((ln for ln in read(path).splitlines() if ln.startswith("#")), None)
        if prefix is None:
            add(f"R3: {rel(path)}: filename has no NN_ prefix")
            continue
        if h1 is None:
            add(f"R3: {rel(path)}: no H1 heading")
            continue
        m = DOC_RE.match(h1)
        if not m:
            add(f"R3: {rel(path)}: H1 lacks a 'NN -' prefix -> {h1[:60]}")
        elif m.group(1) != prefix.group(1):
            add(f"R3: {rel(path)}: H1 number {m.group(1)} != filename prefix {prefix.group(1)}")


def scan_paths(paths: list[Path]) -> None:
    """R2/R5/R7 scans over a concrete file list."""
    for path in paths:
        try:
            text = read(path)
        except OSError as exc:  # pragma: no cover
            add(f"read error: {rel(path)}: {exc}")
            continue
        in_project = PROJECTS_DIR in path.parents
        for lineno, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            if STALE_DOCS_PATH_RE.search(line):
                add(f"R2: {rel(path)}:{lineno}: stale 'docs/<file>' reference -> {line.strip()[:90]}")
            elif STALE_DOCS_BARE_RE.search(line):
                notes.append(f"R2(warn): {rel(path)}:{lineno} mentions 'docs/' -> {line.strip()[:80]}")
            if "templates/" in low and not any(k in low for k in ("legacy", "fallback", "venue-templates")):
                add(f"R5: {rel(path)}:{lineno}: legacy payload path -> {line.strip()[:90]}")
            if in_project:
                for token in PLACEHOLDER_RE.findall(line):
                    if token == "PROJECT_NAME":
                        # scaffold-time token: must always be replaced in a real project
                        add(f"R7: {rel(path)}:{lineno}: unresolved {{{{PROJECT_NAME}}}} -> {line.strip()[:80]}")
                    else:
                        # task token (TARGET_COLUMN, ID_COL, ...): expected until the task is configured
                        notes.append(f"R7(warn): {rel(path)}:{lineno}: task placeholder {{{{{token}}}}} not filled yet")


def check_ledger(path: Path, label: str) -> None:
    if not path.exists():
        add(f"R6: {label}: results.json missing")
        return
    try:
        data = json.loads(read(path) or "[]")
    except json.JSONDecodeError as exc:
        add(f"R6: {label}: results.json is not valid JSON ({exc})")
        return
    if not isinstance(data, list):
        add(f"R6: {label}: results.json must be a JSON array")
        return
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            add(f"R6: {label}: record {i} is not an object")
            continue
        run_id = str(rec.get("run_id", ""))
        if not run_id or re.search(r"^YYYY|example|placeholder", run_id, re.I):
            add(f"R6: {label}: record {i} looks like a placeholder (run_id={run_id!r})")


def project_paths(project: Path) -> list[Path]:
    paths: list[Path] = []
    for name in ["AGENT_RULES.md", "README.md", "program.md", "results.json"]:
        p = project / name
        if p.exists():
            paths.append(p)
    for sub in ["doc", "src", "configs", "memory"]:
        d = project / sub
        if d.is_dir():
            paths.extend(sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXT))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify AgentML SSoT consistency.")
    ap.add_argument("--project", default=None, help="Only check this project slug")
    ap.add_argument("--quiet", action="store_true", help="Only print the verdict")
    args = ap.parse_args()

    check_r1_payload()

    live = [REPO_ROOT / n for n in ["README.md", "README.zh-CN.md", "SKILL.md", "structure.md", "projects.md"]]
    live = [p for p in live if p.exists()]
    live.extend(iter_text_files(SKILL_DIR))
    live.extend(iter_text_files(PAYLOAD))
    live = [p for p in live if p.suffix.lower() in TEXT_EXT]
    scan_paths(live)

    check_doc_set(PAYLOAD / "doc", "payload")
    check_ledger(PAYLOAD / "results.json", "payload")

    projects = sorted(p for p in PROJECTS_DIR.glob("*") if p.is_dir()) if PROJECTS_DIR.is_dir() else []
    if args.project:
        projects = [p for p in projects if p.name == args.project]
    if not projects:
        notes.append("no projects/ found (nothing to verify for R4/R6/R7)")
    for project in projects:
        label = f"projects/{project.name}"
        check_doc_set(project / "doc", label)
        check_ledger(project / "results.json", label)
        scan_paths(project_paths(project))
        if not (project / "src").is_dir():
            notes.append(f"{label}: no src/ (scaffold-only copy?)")

    if not args.quiet:
        print("AgentML SSoT check")
        print(f"  repo root : {REPO_ROOT}")
        print(f"  payload   : {rel(PAYLOAD)}")
        print(f"  projects  : {[p.name for p in projects] or '-'}")
        for n in notes:
            print(f"  NOTE  {n}")
        if problems:
            print(f"\n  {len(problems)} problem(s):")
            for p in problems:
                print(f"   - {p}")
        else:
            print("\n  no problems found")

    verdict = "PASS" if not problems else "FAIL"
    print(f"\n[{verdict}] {len(problems)} problem(s) / {len(notes)} note(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
