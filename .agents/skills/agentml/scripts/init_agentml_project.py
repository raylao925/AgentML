from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold an AgentML project from templates/."
    )
    parser.add_argument("project_slug", help="Folder name under projects/")
    parser.add_argument(
        "--python-version",
        default="3.11",
        help="Python version used for uv venv (default: 3.11)",
    )
    parser.add_argument(
        "--setup-venv",
        action="store_true",
        help="Create .venv and install root requirements.txt with uv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing projects/<project_slug>",
    )
    return parser.parse_args()


def find_repo_root(script_path: Path) -> Path:
    # Expected location: <repo>/.agents/skills/agentml/scripts/init_agentml_project.py
    expected = script_path.resolve().parents[4]
    if (expected / "templates").exists() and (expected / "projects").exists():
        return expected

    for parent in script_path.resolve().parents:
        if (parent / "templates").exists() and (parent / "projects").exists():
            return parent

    raise FileNotFoundError("Cannot locate repository root containing templates/ and projects/.")


def replace_placeholder(file_path: Path, project_slug: str) -> None:
    if not file_path.exists():
        return
    content = file_path.read_text(encoding="utf-8")
    content = content.replace("{{PROJECT_NAME}}", project_slug)
    file_path.write_text(content, encoding="utf-8")


def setup_venv(project_dir: Path, requirements_file: Path, python_version: str) -> None:
    uv_cmd = shutil.which("uv")
    if not uv_cmd:
        print("Warning: uv is not installed. Scaffold completed, environment setup skipped.")
        return

    subprocess.run([uv_cmd, "venv", "--python", python_version, ".venv"], cwd=project_dir, check=True)

    if requirements_file.exists():
        subprocess.run([uv_cmd, "pip", "install", "-r", str(requirements_file)], cwd=project_dir, check=True)
    else:
        print(f"Warning: requirements.txt not found at {requirements_file}")


def main() -> int:
    args = parse_args()
    script_path = Path(__file__)
    repo_root = find_repo_root(script_path)

    template_dir = repo_root / "templates"
    projects_dir = repo_root / "projects"
    project_dir = projects_dir / args.project_slug
    requirements_file = repo_root / "requirements.txt"

    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    if project_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Project already exists: {project_dir}. Re-run with --force to overwrite."
            )
        shutil.rmtree(project_dir)

    shutil.copytree(template_dir, project_dir)

    required_dirs = [
        project_dir / "data" / "raw",
        project_dir / "data" / "interim",
        project_dir / "data" / "processed",
        project_dir / "runs",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    replace_placeholder(project_dir / "README.md", args.project_slug)
    replace_placeholder(project_dir / "program.md", args.project_slug)

    if args.setup_venv:
        setup_venv(project_dir, requirements_file, args.python_version)

    print(f"Scaffold completed: {project_dir}")
    print("Next steps:")
    print("  1) Fill doc/00_problem_statement.md and doc/01_data_card.md")
    print("  2) Confirm doc/03_cv_strategy.md")
    print("  3) Run: python src/train.py --config configs/baseline.yaml")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
