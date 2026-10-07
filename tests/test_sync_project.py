"""
Unit tests for sync_project.py payload/state ownership (Phase 1).

Verifies the core safety property introduced by the ownership manifest:
  - PAYLOAD class (AGENT_RULES.md, program.md, framework src, data_sources) is
    flowed outward (overwritten) when the payload differs.
  - STATE class (README.md, project.yaml, configs/*, doc/*, memory/*,
    src/{features,tune}.py) is never overwritten when the existing file already
    has real body; missing or body-less state files are still (re)filled.

Run from repo root:
  python tests/test_sync_project.py
  py tests/test_sync_project.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / ".agents" / "skills" / "agentml" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import sync_project as sp  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class SyncOwnershipTest(unittest.TestCase):
    """Payload flows out; state is protected."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.payload = root / "payload"
        self.projects = root / "projects"
        self.project = self.projects / "proj"

        # Remember and patch the module globals sync_project resolves against.
        self._old = (sp.PAYLOAD, sp.PROJECTS_DIR)
        sp.PAYLOAD = self.payload
        sp.PROJECTS_DIR = self.projects
        self.addCleanup(self._restore)

        # --- payload (skill-owned) -------------------------------------------- #
        _write(self.payload / "AGENT_RULES.md", "# AGENT_RULES\n\npayload rules v2\n")
        _write(self.payload / "program.md", "# program\n\npayload program v2\n")
        _write(self.payload / "README.md", "# readme\n\npayload readme template\n")
        _write(self.payload / "project.yaml", 'project:\n  name: "{{PROJECT_NAME}}"\n')
        _write(self.payload / "src" / "train.py", "# payload train v2\n")
        _write(self.payload / "src" / "features.py", "# payload features v2\n")
        _write(self.payload / "src" / "tune.py", "# payload tune v2\n")
        _write(self.payload / "configs" / "baseline.yaml", "project:\n  name: template\n")
        _write(self.payload / "data_sources" / "example.yaml", "source: example v2\n")
        _write(self.payload / "doc" / "06_experiment_log.md", "# 06\n\npayload doc v2\n")
        _write(self.payload / "memory" / "ITERATIONS.md", "# ITERATIONS\n\npayload memory v2\n")

        # --- existing project with populated state ---------------------------- #
        _write(self.project / "AGENT_RULES.md", "# AGENT_RULES\n\nproject rules v1\n")
        _write(self.project / "program.md", "# program\n\nproject program v1\n")
        _write(self.project / "README.md", "# readme\n\nproject research notes\n")
        _write(self.project / "project.yaml", 'project:\n  name: "proj"\n  mode: "kaggle"\n')
        _write(self.project / "src" / "train.py", "# project train v1\n")
        _write(self.project / "src" / "features.py", "# project features\n\nproject state fs_v2 v1\n")
        _write(self.project / "src" / "tune.py", "# project tune\n\ntune state v1\n")
        _write(self.project / "configs" / "baseline.yaml", "project:\n  name: proj-real\n")
        _write(self.project / "doc" / "06_experiment_log.md", "# 06\n\nproject rounds log\n")
        _write(self.project / "memory" / "ITERATIONS.md", "# ITERATIONS\n\nproject round state\n")

    def _restore(self) -> None:
        sp.PAYLOAD, sp.PROJECTS_DIR = self._old

    def _sync(self, include_docs: bool = True, include_memory: bool = True, dry_run: bool = False):
        return sp.sync_project(self.project, include_docs, include_memory, dry_run, False)

    def test_payload_class_is_overwritten(self) -> None:
        stats, _ = self._sync()
        self.assertGreaterEqual(stats["update"], 3)
        self.assertIn("payload rules v2", (self.project / "AGENT_RULES.md").read_text(encoding="utf-8"))
        self.assertIn("payload program v2", (self.project / "program.md").read_text(encoding="utf-8"))
        self.assertIn("payload train v2", (self.project / "src" / "train.py").read_text(encoding="utf-8"))
        self.assertIn("example v2", (self.project / "data_sources" / "example.yaml").read_text(encoding="utf-8"))

    def test_state_class_is_never_overwritten(self) -> None:
        stats, actions = self._sync()
        # Existing files with real body keep their content.
        self.assertIn("project research notes", (self.project / "README.md").read_text(encoding="utf-8"))
        self.assertIn("project rounds log", (self.project / "doc" / "06_experiment_log.md").read_text(encoding="utf-8"))
        self.assertIn("project round state", (self.project / "memory" / "ITERATIONS.md").read_text(encoding="utf-8"))
        self.assertIn("name: proj-real", (self.project / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
        self.assertIn("fs_v2", (self.project / "src" / "features.py").read_text(encoding="utf-8"))
        self.assertIn("tune state v1", (self.project / "src" / "tune.py").read_text(encoding="utf-8"))
        # project.yaml (per-project mode/created) always stays add-only.
        self.assertIn('mode: "kaggle"', (self.project / "project.yaml").read_text(encoding="utf-8"))
        # The report names the protected files so a dry-run makes the guard visible.
        self.assertTrue(any(a.startswith("skip(state): configs/baseline.yaml") for a in actions))
        self.assertTrue(any(a.startswith("skip(state): src/features.py") for a in actions))

    def test_missing_and_bodyless_state_files_are_filled(self) -> None:
        # README exists but carries only headings (pure template) -> refreshed.
        _write(self.project / "README.md", "# readme\n\n> template quote only\n---\n")
        # doc/ file missing entirely -> added.
        (self.project / "doc" / "06_experiment_log.md").unlink()
        stats, actions = self._sync()
        self.assertIn("payload readme template", (self.project / "README.md").read_text(encoding="utf-8"))
        self.assertIn("payload doc v2", (self.project / "doc" / "06_experiment_log.md").read_text(encoding="utf-8"))
        self.assertTrue(any(a.startswith("update(state): README.md") for a in actions))
        self.assertTrue(any(a.startswith("add(state): doc/06_experiment_log.md") for a in actions))

    def test_optional_dirs_are_flag_gated(self) -> None:
        # Without the flags, doc/ and memory/ are not targeted at all.
        (self.project / "doc" / "06_experiment_log.md").unlink()
        (self.project / "memory" / "ITERATIONS.md").unlink()
        self._sync(include_docs=False, include_memory=False)
        self.assertFalse((self.project / "doc" / "06_experiment_log.md").exists())
        self.assertFalse((self.project / "memory" / "ITERATIONS.md").exists())

    def test_dry_run_changes_nothing(self) -> None:
        before = (self.project / "AGENT_RULES.md").read_text(encoding="utf-8")
        stats, _ = self._sync(dry_run=True)
        self.assertEqual(before, (self.project / "AGENT_RULES.md").read_text(encoding="utf-8"))
        self.assertGreaterEqual(stats["update"], 3)


class IsStateClassificationTest(unittest.TestCase):
    def test_payload_paths(self) -> None:
        for rel in (
            Path("AGENT_RULES.md"),
            Path("program.md"),
            Path("src/train.py"),
            Path("src/train_multi_model.py"),
            Path("src/build_features.py"),
            Path("data_sources/example.yaml"),
        ):
            self.assertFalse(sp.is_state(rel), rel.as_posix())

    def test_state_paths(self) -> None:
        for rel in (
            Path("README.md"),
            Path("configs/baseline.yaml"),
            Path("configs/search_space.yaml"),
            Path("doc/06_experiment_log.md"),
            Path("memory/NEXT.md"),
            Path("src/features.py"),
            Path("src/tune.py"),
            Path("project.yaml"),
        ):
            self.assertTrue(sp.is_state(rel), rel.as_posix())


if __name__ == "__main__":
    unittest.main()
