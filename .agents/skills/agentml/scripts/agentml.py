#!/usr/bin/env python3
"""agentml.py — unified entrypoint for the AgentML skill (references/00_contract.md §3).

    python .agents/skills/agentml/scripts/agentml.py <command> [options]

Commands:
  new        Scaffold projects/<slug> from the template payload
  sync       Flow skill-payload fixes out to existing projects
  doctor     Environment + contract sanity checks
  ingest     raw -> data/processed via data_sources/<name>.yaml
  eda        Run src/eda.py for a project
  cv-lock    Write/refresh locked_hash in doc/04_cv_strategy.md
  run        Train (single via train.py, or --models via train_multi_model.py)
  ensemble   Build an ensemble from existing runs
  tune       Optuna HPO over the locked CV (src/tune.py)
  infer      Score test data (single run or ensemble)
  deliver    Produce deliverables (submission_csv | batch_scoring | api_contract)
  ledger     verify | best | report over results.json
  guardrails Executable hard-rule checks (references/09_guardrails.md)
  check-ssot Structure / anti-drift checks

Everything is dispatch: the underlying scripts stay runnable on their own, so older
command snippets keep working.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "agentml"
SCRIPTS = SKILL_DIR / "scripts"
PAYLOAD = SKILL_DIR / "assets" / "project-template"
PROJECTS_DIR = REPO_ROOT / "projects"


def _py(path: Path, extra: list[str], cwd: Path | None = None) -> int:
    cmd = [sys.executable, str(path), *extra]
    print(f"$ {' '.join(_q(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None).returncode


def _q(s: str) -> str:
    return f'"{s}"' if " " in s else s


def _project_dir(slug: str | None) -> Path:
    if slug:
        return PROJECTS_DIR / slug
    cwd = Path.cwd()
    if (cwd / "project.yaml").exists() or (cwd / "configs").is_dir():
        return cwd
    raise SystemExit("Error: pass --project <slug> (or run from inside a project folder)")


def _slug_of(project_dir: Path) -> str:
    return project_dir.name


def _passthrough(opt: str, value) -> list[str]:
    return [opt, str(value)] if value not in (None, "") else []


# ---------------------------------------------------------------- commands

def cmd_new(args) -> int:
    extra = [args.slug, "--mode", args.mode]
    if args.setup_venv:
        extra.append("--setup-venv")
    if args.python_version:
        extra += ["--python-version", args.python_version]
    if args.force:
        extra.append("--force")
    return _py(SCRIPTS / "init_agentml_project.py", extra, cwd=REPO_ROOT)


def cmd_sync(args) -> int:
    extra: list[str] = []
    if args.all:
        extra.append("--all")
    else:
        extra += ["--project", args.project]
    if args.include_docs:
        extra.append("--include-docs")
    if args.include_memory:
        extra.append("--include-memory")
    if args.reset_ledger:
        extra.append("--reset-ledger")
    if args.dry_run:
        extra.append("--dry-run")
    return _py(SCRIPTS / "sync_project.py", extra, cwd=REPO_ROOT)


def cmd_doctor(args) -> int:
    import shutil

    problems = 0
    print("== agentml doctor ==")
    print(f"python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"repo root: {REPO_ROOT}")

    core = ["numpy", "pandas", "sklearn", "lightgbm", "pyarrow", "yaml", "joblib", "scipy"]
    optional = ["xgboost", "catboost", "featuretools", "kaggle", "sqlalchemy", "openpyxl"]
    for mods, tag in ((core, "core"), (optional, "optional")):
        for m in mods:
            try:
                __import__(m)
                print(f"  [OK] {m} ({tag})")
            except Exception as exc:  # noqa: BLE001
                flag = "FAIL" if tag == "core" else "--"
                print(f"  [{flag}] {m} ({tag}): {exc}")
                if tag == "core":
                    problems += 1
    uv = shutil.which("uv")
    print(f"  [{'OK' if uv else '--'}] uv {'found' if uv else 'not installed (venv setup skipped)'}")

    try:
        pdir = _project_dir(args.project)
    except SystemExit:
        pdir = None
    if pdir and pdir.exists():
        print(f"-- project: {pdir.name}")
        checks = [
            ("project.yaml", (pdir / "project.yaml").exists()),
            ("configs/baseline.yaml", (pdir / "configs" / "baseline.yaml").exists()),
            ("src/data.py", (pdir / "src" / "data.py").exists()),
            ("reports/data_manifest.json", (pdir / "reports" / "data_manifest.json").exists()),
        ]
        for name, present in checks:
            print(f"  [{'OK' if present else '--'}] {name}")
        doc4 = pdir / "doc" / "04_cv_strategy.md"
        locked = doc4.exists() and "locked_hash" in doc4.read_text(encoding="utf-8", errors="replace")
        print(f"  [{'OK' if locked else '--'}] CV locked (doc/04_cv_strategy.md)")
        pj = pdir / "project.yaml"
        if pj.exists():
            import yaml

            cfg = yaml.safe_load(pj.read_text(encoding="utf-8")) or {}
            print(f"  mode={ (cfg.get('project') or {}).get('mode') } "
                  f"skill={(cfg.get('skill') or {}).get('version')} "
                  f"policy={(cfg.get('policy') or {}).get('version')}")

    print(f"doctor: {problems} core problem(s)")
    return 1 if problems else 0


def cmd_ingest(args) -> int:
    pdir = _project_dir(args.project)
    return _py(pdir / "src" / "ingest.py", ["--source", args.source], cwd=pdir)


def cmd_eda(args) -> int:
    pdir = _project_dir(args.project)
    extra = ["--project", _slug_of(pdir), "--config", args.config]
    if args.no_plots:
        extra.append("--no-plots")
    return _py(pdir / "src" / "eda.py", extra, cwd=pdir)


def cmd_cv_lock(args) -> int:
    sys.path.insert(0, str(SCRIPTS))
    import guardrails as gr  # reuse the exact hash normalisation

    pdir = _project_dir(args.project)
    data_py = pdir / "src" / "data.py"
    doc = pdir / "doc" / "04_cv_strategy.md"
    if not data_py.exists():
        print(f"Error: {data_py} not found", file=sys.stderr)
        return 1
    if not doc.exists():
        print(f"Error: {doc} not found", file=sys.stderr)
        return 1
    digest = gr.sha256_text(gr.normalize_py(data_py.read_text(encoding="utf-8", errors="replace")))
    text = doc.read_text(encoding="utf-8", errors="replace")
    if "locked_hash:" in text:
        import re

        text = re.sub(r"(?m)^locked_hash:.*$", f"locked_hash: {digest}", text)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += f"\n## Lock\n\nlocked_hash: {digest}\n"
    doc.write_text(text, encoding="utf-8")
    print(f"cv-lock: {doc.name} -> locked_hash={digest[:12]}...")
    return 0


def cmd_run(args) -> int:
    pdir = _project_dir(args.project)
    if args.models:
        extra = ["--config", args.config, "--models", args.models]
        if args.run_id_prefix is not None:
            extra += ["--run_id_prefix", args.run_id_prefix]
        return _py(pdir / "src" / "train_multi_model.py", extra, cwd=pdir)
    extra = ["--config", args.config]
    if args.run_id:
        extra += ["--run_id", args.run_id]
    return _py(pdir / "src" / "train.py", extra, cwd=pdir)


def cmd_ensemble(args) -> int:
    pdir = _project_dir(args.project)
    extra = ["--run_ids", args.run_ids, "--method", args.method]
    if args.output_run_id:
        extra += ["--output_run_id", args.output_run_id]
    return _py(pdir / "src" / "ensemble.py", extra, cwd=pdir)


def cmd_tune(args) -> int:
    pdir = _project_dir(args.project)
    extra = ["--config", args.config, "--model", args.model,
             "--n-trials", str(args.n_trials), "--timeout-s", str(args.timeout_s)]
    if args.study_id:
        extra += ["--study-id", args.study_id]
    if args.no_final_run:
        extra.append("--no-final-run")
    return _py(pdir / "src" / "tune.py", extra, cwd=pdir)


def cmd_infer(args) -> int:
    pdir = _project_dir(args.project)
    if args.ensemble:
        if not args.ensemble_run_id:
            print("Error: --ensemble requires --ensemble_run_id", file=sys.stderr)
            return 1
        extra = ["--ensemble_run_id", args.ensemble_run_id]
        if args.optimize_weights:
            extra.append("--optimize_weights")
        extra += _passthrough("--input", args.input) + _passthrough("--output", args.output)
        return _py(pdir / "src" / "infer_ensemble.py", extra, cwd=pdir)
    if not args.run_id:
        print("Error: pass --run_id (or --ensemble --ensemble_run_id)", file=sys.stderr)
        return 1
    extra = ["--run_id", args.run_id]
    extra += _passthrough("--input", args.input) + _passthrough("--output", args.output)
    return _py(pdir / "src" / "infer.py", extra, cwd=pdir)


def cmd_deliver(args) -> int:
    pdir = _project_dir(args.project)
    extra: list[str] = []
    if args.ensemble_run_id:
        extra += ["--ensemble_run_id", args.ensemble_run_id]
    elif args.run_id:
        extra += ["--run_id", args.run_id]
    else:
        print("Error: pass --run_id or --ensemble_run_id", file=sys.stderr)
        return 1
    extra += _passthrough("--mode", args.mode)
    extra += ["--threshold", str(args.threshold)]
    extra += _passthrough("--input", args.input)
    extra += _passthrough("--output", args.output)
    extra += _passthrough("--out-dir", args.out_dir)
    return _py(pdir / "src" / "deliver.py", extra, cwd=pdir)


def _ledger(pdir: Path) -> list:
    p = pdir / "results.json"
    if not p.exists():
        print("Error: results.json not found", file=sys.stderr)
        raise SystemExit(1)
    try:
        return json.loads(p.read_text(encoding="utf-8") or "[]")
    except json.JSONDecodeError as exc:
        print(f"Error: results.json invalid JSON: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _primary(rec: dict):
    metrics = rec.get("metrics") or {}
    for path in (("primary", "mean"), ("primary_mean",)):
        cur = metrics
        for key in path:
            cur = cur.get(key) if isinstance(cur, dict) else None
        if isinstance(cur, (int, float)):
            return float(cur)
    return None


def cmd_ledger(args) -> int:
    pdir = _project_dir(args.project)
    ledger = _ledger(pdir)
    if args.action == "verify":
        bad = []
        for i, rec in enumerate(ledger):
            if not isinstance(rec, dict) or "run_id" not in rec or "metrics" not in rec:
                bad.append(i)
                continue
            if str(rec["run_id"]).lower().startswith("example"):
                bad.append(i)
        if bad:
            print(f"ledger verify: FAIL ({len(bad)} bad records at {bad})")
            return 1
        print(f"ledger verify: PASS ({len(ledger)} records)")
        return 0

    if args.action == "report":
        print(f"{'run_id':40} {'primary':>10} {'decision':>10}")
        for rec in ledger:
            val = _primary(rec)
            dec = (rec.get("decision") or {}).get("status", "")
            shown = "" if val is None else f"{val:.5f}"
            print(f"{str(rec.get('run_id', '?'))[:40]:40} {shown:>10} {dec:>10}")
        return 0

    # best
    import yaml

    cfg_path = pdir / "configs" / "baseline.yaml"
    higher = True
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        higher = bool((cfg.get("task") or {}).get("metric_higher_is_better", True))
    scored = [(r, _primary(r)) for r in ledger if _primary(r) is not None]
    if not scored:
        print("ledger best: no records with a primary metric")
        return 1
    best = (max if higher else min)(scored, key=lambda t: t[1])
    print(f"best run: {best[0].get('run_id')}  primary={best[1]:.5f}  (higher_is_better={higher})")
    return 0


def cmd_guardrails(args) -> int:
    extra: list[str] = []
    if args.payload:
        extra = ["--payload"]
    else:
        pdir = _project_dir(args.project)
        extra = ["--project", _slug_of(pdir)]
    if args.quiet:
        extra.append("--quiet")
    return _py(SCRIPTS / "guardrails.py", extra, cwd=REPO_ROOT)


def cmd_check_ssot(args) -> int:
    extra = _passthrough("--project", args.project)
    if args.quiet:
        extra.append("--quiet")
    return _py(SCRIPTS / "check_ssot.py", extra, cwd=REPO_ROOT)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentml", description="AgentML unified CLI (references/00_contract.md §3)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("new", help="Scaffold projects/<slug>")
    s.add_argument("slug")
    s.add_argument("--mode", choices=["customer", "kaggle"], default="customer")
    s.add_argument("--setup-venv", action="store_true")
    s.add_argument("--python-version", default="3.11")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("sync", help="Sync payload fixes to projects")
    s.add_argument("--project", default=None)
    s.add_argument("--all", action="store_true")
    s.add_argument("--include-docs", action="store_true")
    s.add_argument("--include-memory", action="store_true")
    s.add_argument("--reset-ledger", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("doctor", help="Environment + contract checks")
    s.add_argument("--project", default=None)
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("ingest", help="raw -> data/processed")
    s.add_argument("--source", required=True)
    s.add_argument("--project", default=None)
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("eda", help="Run EDA for a project")
    s.add_argument("--project", default=None)
    s.add_argument("--config", default="configs/baseline.yaml")
    s.add_argument("--no-plots", action="store_true")
    s.set_defaults(func=cmd_eda)

    s = sub.add_parser("cv-lock", help="Write locked_hash into doc/04_cv_strategy.md")
    s.add_argument("--project", default=None)
    s.set_defaults(func=cmd_cv_lock)

    s = sub.add_parser("run", help="Train a model / models")
    s.add_argument("--project", default=None)
    s.add_argument("--config", default="configs/baseline.yaml")
    s.add_argument("--models", default=None)
    s.add_argument("--run-id", dest="run_id", default=None)
    s.add_argument("--run_id_prefix", dest="run_id_prefix", default=None)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("ensemble", help="Build an ensemble")
    s.add_argument("--run_ids", required=True)
    s.add_argument("--method", default="weighted_average")
    s.add_argument("--output_run_id", default=None)
    s.add_argument("--project", default=None)
    s.set_defaults(func=cmd_ensemble)

    s = sub.add_parser("tune", help="Optuna HPO over the locked CV (src/tune.py)")
    s.add_argument("--model", default="LightGBM")
    s.add_argument("--config", default="configs/baseline.yaml")
    s.add_argument("--n-trials", dest="n_trials", type=int, default=60)
    s.add_argument("--timeout-s", dest="timeout_s", type=int, default=7200)
    s.add_argument("--study-id", dest="study_id", default=None)
    s.add_argument("--no-final-run", dest="no_final_run", action="store_true")
    s.add_argument("--project", default=None)
    s.set_defaults(func=cmd_tune)

    s = sub.add_parser("infer", help="Score test data")
    s.add_argument("--run_id", default=None)
    s.add_argument("--ensemble", action="store_true")
    s.add_argument("--ensemble_run_id", default=None)
    s.add_argument("--optimize_weights", action="store_true")
    s.add_argument("--input", default=None)
    s.add_argument("--output", default=None)
    s.add_argument("--project", default=None)
    s.set_defaults(func=cmd_infer)

    s = sub.add_parser("deliver", help="Produce deliverables")
    s.add_argument("--run_id", default=None)
    s.add_argument("--ensemble_run_id", default=None)
    s.add_argument("--mode", choices=["submission_csv", "batch_scoring", "api_contract"], default=None)
    s.add_argument("--threshold", default="auto")
    s.add_argument("--input", default=None)
    s.add_argument("--output", default=None)
    s.add_argument("--out-dir", dest="out_dir", default=None)
    s.add_argument("--project", default=None)
    s.set_defaults(func=cmd_deliver)

    s = sub.add_parser("ledger", help="results.json operations")
    s.add_argument("action", choices=["verify", "best", "report"])
    s.add_argument("--project", default=None)
    s.set_defaults(func=cmd_ledger)

    s = sub.add_parser("guardrails", help="Executable hard-rule checks")
    s.add_argument("--project", default=None)
    s.add_argument("--payload", action="store_true")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_guardrails)

    s = sub.add_parser("check-ssot", help="Structure / anti-drift checks")
    s.add_argument("--project", default=None)
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_check_ssot)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())