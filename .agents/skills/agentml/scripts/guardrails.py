#!/usr/bin/env python3
"""guardrails.py — executable hard-rule checks (references/09_guardrails.md).

Two modes:
  --payload            run template-level checks against assets/project-template
  --project <slug>     run full checks against projects/<slug> (default: cwd if it is a project)

Exit code: 0 = no FAIL, 1 = at least one FAIL. WARN never fails the run.

Checks: G1 target-not-in-features, G2 CV locked_hash, G3 no test-in-tuning, G4 ledger schema,
G5 no placeholder residue, G6 no group/query across folds, G7 fold-safe AST heuristic,
G8 project.yaml versions, G9 frozen-file hashes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "agentml"
PAYLOAD = SKILL_DIR / "assets" / "project-template"
PROJECTS_DIR = REPO_ROOT / "projects"

PLACEHOLDER_RE = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}")
FROZEN_RE = re.compile(r"#\s*AGENTML:FROZEN\s+v\d+\s+([0-9a-fA-F]{64})")

results: list[tuple[str, str, str]] = []  # (check_id, status, message)


def record(check_id: str, status: str, msg: str) -> None:
    results.append((check_id, status, msg))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_yaml(path: Path) -> dict:
    import yaml

    if not path.exists():
        return {}
    try:
        return yaml.safe_load(read(path)) or {}
    except yaml.YAMLError:
        # Template files may contain {{PLACEHOLDER}} tokens that are not valid YAML
        # (e.g. seed_list: [{{SEED1}}]). Treat unparsable configs as empty.
        return {}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_project(args) -> Path | None:
    if args.payload:
        return None
    if args.project:
        return PROJECTS_DIR / args.project
    cwd = Path.cwd()
    if (cwd / "project.yaml").exists() or (cwd / "configs").exists():
        return cwd
    return None


def _src_files(root: Path) -> list[Path]:
    return sorted((root / "src").glob("*.py")) if (root / "src").is_dir() else []


def _config_files(root: Path) -> list[Path]:
    return sorted((root / "configs").glob("*.yaml")) if (root / "configs").is_dir() else []


def normalize_py(text: str) -> str:
    """Stable normal form for hashing: drop blank/comment lines, strip trailing ws."""
    keep = []
    for line in text.splitlines():
        s = line.rstrip()
        if s and not s.lstrip().startswith("#"):
            keep.append(s)
    return "\n".join(keep)


def frozen_expected_hash(path: Path) -> str:
    """Hash of the file with its FROZEN marker line removed (so the hash is not self-referential)."""
    lines = [ln for ln in read(path).splitlines() if not FROZEN_RE.search(ln)]
    return sha256_text(normalize_py("\n".join(lines)))


# ---------------------------------------------------------------- checks

def check_g1(root: Path) -> None:
    cfg = load_yaml(root / "configs" / "baseline.yaml")
    target = (cfg.get("task") or {}).get("target")
    if not target or target in ("{{TARGET_COLUMN}}", "target"):
        record("G1", "SKIP", "no concrete task.target configured yet")
        return
    for cpath in _config_files(root):
        c = load_yaml(cpath)
        for key in ("feature_cols", "include_cols"):
            vals = (c.get("features") or {}).get(key) or c.get(key)
            if isinstance(vals, list) and target in vals:
                record("G1", "FAIL", f"{cpath.name}: target '{target}' present in {key}")
                return
    fp = root / "src" / "features.py"
    if fp.exists() and "target_col" not in read(fp):
        record("G1", "FAIL", "src/features.py does not reference target_col (target may leak into features)")
        return
    record("G1", "PASS", f"target '{target}' excluded from features")


def check_g2(root: Path, allow_unlocked: bool) -> None:
    doc = root / "doc" / "04_cv_strategy.md"
    data_py = root / "src" / "data.py"
    if not doc.exists():
        record("G2", "FAIL", "doc/04_cv_strategy.md missing")
        return
    m = re.search(r"locked_hash:\s*([0-9a-fA-F]{8,})", read(doc))
    if not m:
        record("G2", "WARN" if allow_unlocked else "FAIL",
               "doc/04_cv_strategy.md has no locked_hash (run `agentml cv-lock`)")
        return
    if not data_py.exists():
        record("G2", "FAIL", "src/data.py missing")
        return
    actual = sha256_text(normalize_py(read(data_py)))
    if actual.startswith(m.group(1)) or m.group(1).startswith(actual[: len(m.group(1))]):
        record("G2", "PASS", f"locked_hash matches ({actual[:12]}...)")
    else:
        record("G2", "FAIL",
               f"locked_hash {m.group(1)[:12]}... != current src/data.py {actual[:12]}...")


def check_g3(root: Path) -> None:
    hints = ("tune", "search", "optuna", "hyperopt")
    hits = []
    for f in _src_files(root):
        txt = read(f)
        is_tune = any(h in f.name.lower() for h in hints) or "import optuna" in txt or "import hyperopt" in txt
        if is_tune and ("test_path" in txt or "test_df" in txt or "test.parquet" in txt):
            hits.append(f.name)
    record("G3", "FAIL" if hits else "PASS",
           f"test data referenced in tuning files: {hits}" if hits else "no test usage in tuning paths")


def check_g4(root: Path) -> None:
    p = root / "results.json"
    if not p.exists():
        record("G4", "FAIL", "results.json missing")
        return
    try:
        ledger = json.loads(read(p) or "[]")
    except Exception as exc:
        record("G4", "FAIL", f"results.json not valid JSON: {exc}")
        return
    if not isinstance(ledger, list):
        record("G4", "FAIL", "results.json is not a JSON array")
        return
    bad = []
    for i, rec in enumerate(ledger):
        if not isinstance(rec, dict) or "run_id" not in rec:
            bad.append(i)
            continue
        rid = str(rec.get("run_id", "")).lower()
        if rid in {"example", "placeholder", "example_run"} or rid.startswith("example"):
            bad.append(i)
    record("G4", "FAIL" if bad else "PASS",
           f"invalid/placeholder ledger records at {bad}" if bad else f"ledger OK ({len(ledger)} records)")


def check_g5(root: Path, allow: bool) -> None:
    if allow:
        record("G5", "WARN", "placeholder check skipped (--allow-placeholders)")
        return
    found = []
    targets = list((root / "doc").glob("*.md")) + _config_files(root) + [root / "project.yaml", root / "README.md"]
    for p in targets:
        if not p.exists():
            continue
        for m in PLACEHOLDER_RE.finditer(read(p)):
            found.append(f"{p.relative_to(root).as_posix()}:{m.group(0)}")
    record("G5", "FAIL" if found else "PASS",
           f"unresolved placeholders: {found[:6]}{' ...' if len(found) > 6 else ''}" if found else "no placeholder residue")


def check_g6(root: Path) -> None:
    cfg = load_yaml(root / "configs" / "baseline.yaml")
    cv = cfg.get("cv") or {}
    cv_type = str(cv.get("cv_type", ""))
    group = cv.get("group_key") or (cfg.get("data") or {}).get("group_key")
    if "Group" not in cv_type or not group:
        record("G6", "SKIP", f"cv_type={cv_type or 'n/a'} (no group constraint)")
        return
    try:
        import pandas as pd
        from sklearn.model_selection import GroupKFold
    except Exception as exc:  # pragma: no cover
        record("G6", "SKIP", f"deps unavailable: {exc}")
        return
    train_path = Path((cfg.get("data") or {}).get("train_path", "data/processed/train.parquet"))
    if not train_path.is_absolute():
        train_path = root / train_path
    if not train_path.exists():
        record("G6", "SKIP", f"train data not found: {train_path.name}")
        return
    df = pd.read_parquet(train_path) if train_path.suffix != ".csv" else pd.read_csv(train_path)
    if group not in df.columns:
        record("G6", "FAIL", f"group_key '{group}' not in training data")
        return
    groups = df[group].to_numpy()
    n = int(cv.get("n_splits", 5))
    leaks = 0
    for tr, va in GroupKFold(n_splits=n).split(df, groups=groups):
        if set(groups[tr]) & set(groups[va]):
            leaks += 1
    record("G6", "FAIL" if leaks else "PASS",
           f"{leaks} fold(s) leak groups across train/valid" if leaks else f"GroupKFold({n}) clean")


def check_g7(root: Path) -> None:
    warns = []
    for f in _src_files(root):
        try:
            tree = ast.parse(read(f))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in {"fit", "fit_transform"} and node.args:
                a = node.args[0]
                name = getattr(a, "id", None) or getattr(a, "attr", None) or ""
                if re.search(r"(^|_)(va|valid|test)", str(name), re.IGNORECASE):
                    warns.append(f"{f.name}:{node.lineno} .{node.func.attr}({name})")
    record("G7", "WARN" if warns else "PASS",
           f"possible fit on valid/test: {warns[:4]}" if warns else "no obvious fit() on valid/test")


def check_g8(root: Path) -> None:
    p = root / "project.yaml"
    if not p.exists():
        record("G8", "FAIL", "project.yaml missing")
        return
    cfg = load_yaml(p)
    sv = (cfg.get("skill") or {}).get("version")
    pv = (cfg.get("policy") or {}).get("version")
    if not sv or not pv:
        record("G8", "FAIL", "project.yaml missing skill.version and/or policy.version")
    else:
        record("G8", "PASS", f"project.yaml skill={sv} policy={pv}")


def check_g9(root: Path) -> None:
    bad, checked = [], 0
    for f in _src_files(root):
        m = FROZEN_RE.search(read(f))
        if not m:
            continue
        checked += 1
        if frozen_expected_hash(f) != m.group(1).lower():
            bad.append(f.name)
    if bad:
        record("G9", "FAIL", f"frozen files modified: {bad}")
    elif checked:
        record("G9", "PASS", f"{checked} frozen file(s) intact")
    else:
        record("G9", "SKIP", "no AGENTML:FROZEN markers present")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AgentML executable guardrails (references/09_guardrails.md)")
    ap.add_argument("--payload", action="store_true", help="Check the template payload instead of a project")
    ap.add_argument("--project", default=None, help="Project slug under projects/")
    ap.add_argument("--allow-unlocked", action="store_true", help="Downgrade a missing locked_hash to WARN")
    ap.add_argument("--allow-placeholders", action="store_true", help="Skip the placeholder residue check")
    ap.add_argument("--quiet", action="store_true", help="Only print failures/warnings")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_project(args)
    if args.payload:
        root = PAYLOAD
    if root is None or not root.exists():
        print("guardrails: no project resolved (pass --project <slug> or --payload)", file=sys.stderr)
        return 1

    label = "payload" if args.payload else root.name
    print(f"== guardrails [{label}] {root}")
    # The template legitimately carries placeholders / an unlocked CV -> be lenient in payload mode.
    allow_unlocked = args.allow_unlocked or args.payload
    allow_placeholders = args.allow_placeholders or args.payload
    check_g1(root)
    check_g2(root, allow_unlocked)
    check_g3(root)
    check_g4(root)
    check_g5(root, allow_placeholders)
    check_g6(root)
    check_g7(root)
    check_g8(root)
    check_g9(root)

    fails = 0
    for cid, status, msg in results:
        if status == "FAIL":
            fails += 1
        if args.quiet and status in ("PASS", "SKIP"):
            continue
        print(f"  [{status}] {cid}: {msg}")
    print(f"-- {len(results)} checks, {fails} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())