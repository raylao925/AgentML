"""
Smoke test: synthetic 200-row project, 2-fold CV, train → evaluate → ensemble.

Run from repo root:
  python tests/test_smoke.py
  py tests/test_smoke.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / ".agents" / "skills" / "agentml" / "assets" / "project-template"


def _python() -> str:
    return sys.executable


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


class AgentMLSmokeTest(unittest.TestCase):
    def test_end_to_end_smoke(self):
        if not TEMPLATE.is_dir():
            self.skipTest(f"Template payload missing: {TEMPLATE}")

        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=(n, 5))
        logits = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
        y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(int)

        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / "smoke_project"
            shutil.copytree(TEMPLATE, proj, dirs_exist_ok=True)
            raw = proj / "data" / "raw"
            raw.mkdir(parents=True, exist_ok=True)

            train_df = pd.DataFrame(
                {
                    "row_id": np.arange(n),
                    "f0": X[:, 0],
                    "f1": X[:, 1],
                    "f2": X[:, 2],
                    "f3": X[:, 3],
                    "f4": X[:, 4],
                    "target": y,
                }
            )
            test_df = train_df.drop(columns=["target"]).head(20)
            train_df.to_csv(raw / "train.csv", index=False)
            test_df.to_csv(raw / "test.csv", index=False)

            smoke_cfg = {
                "project": {"name": "smoke", "run_tag": "smoke", "seed": 42, "seed_list": [42]},
                "task": {
                    "family": "classification_binary",
                    "target": "target",
                    "primary_metric": "AUC",
                    "metric_higher_is_better": True,
                    "secondary_metrics": ["LogLoss", "Brier"],
                },
                "data": {
                    "data_version": "smoke_v1",
                    "train_path": "data/processed/train.parquet",
                    "test_path": "data/processed/test.parquet",
                    "id_cols": ["row_id"],
                    "drop_cols": [],
                },
                "cv": {
                    "cv_type": "StratifiedKFold",
                    "n_splits": 2,
                    "shuffle": True,
                    "random_state": 42,
                    "stratify_col": "target",
                },
                "features": {
                    "feature_set_id": "smoke_fs",
                    "flags": {
                        "use_scaler": True,
                        "use_onehot": False,
                        "use_featuretools": False,
                    },
                },
                "model": {
                    "name": "LogisticRegression",
                    "objective": "binary",
                    "params": {"C": 1.0, "max_iter": 500},
                },
                "training": {"early_stopping": False},
                "output": {
                    "results_path": "results.json",
                    "runs_dir": "runs",
                    "save_oof": True,
                    "save_model": True,
                },
            }
            cfg_path = proj / "configs" / "smoke.yaml"
            cfg_path.write_text(yaml.safe_dump(smoke_cfg, sort_keys=False), encoding="utf-8")

            sys.path.insert(0, str(proj / "src"))
            try:
                import data as data_mod

                resolved = data_mod.resolve_data_path("data/processed/train.parquet")
                self.assertEqual(resolved.name, "train.csv")
            finally:
                if str(proj / "src") in sys.path:
                    sys.path.remove(str(proj / "src"))

            py = _python()
            run_id = "smoke_run_a"
            _run([py, "src/train.py", "--config", "configs/smoke.yaml", "--run_id", run_id], proj)

            run_dir = proj / "runs" / run_id
            self.assertTrue((run_dir / "artifacts" / "model.pkl").exists())
            self.assertTrue((run_dir / "artifacts" / "oof_predictions.parquet").exists())

            oof = pd.read_parquet(run_dir / "artifacts" / "oof_predictions.parquet")
            expected_auc = roc_auc_score(oof["target"], oof["oof_pred"])
            self.assertGreater(expected_auc, 0.5)

            ev = _run([py, "src/evaluate.py", "--run_id", run_id], proj)
            metrics = json.loads(ev.stdout)
            self.assertAlmostEqual(metrics["primary"], expected_auc, places=6)

            ledger = json.loads((proj / "results.json").read_text(encoding="utf-8"))
            self.assertIsInstance(ledger, list)
            self.assertGreaterEqual(len(ledger), 1)
            self.assertEqual(ledger[-1]["run_id"], run_id)
            self.assertIn("metrics", ledger[-1])

            run_id_b = "smoke_run_b"
            _run([py, "src/train.py", "--config", "configs/smoke.yaml", "--run_id", run_id_b], proj)

            ens_id = "smoke_ensemble"
            _run(
                [
                    py,
                    "src/ensemble.py",
                    "--run_ids",
                    f"{run_id},{run_id_b}",
                    "--output_run_id",
                    ens_id,
                ],
                proj,
            )
            meta_path = proj / "runs" / ens_id / "ensemble_metadata.json"
            self.assertTrue(meta_path.exists())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["run_ids"], [run_id, run_id_b])
            self.assertEqual(len(meta["weights"]), 2)
            self.assertAlmostEqual(sum(meta["weights"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
