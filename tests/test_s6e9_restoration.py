"""
Restoration checks for projects/playground-series-s6e9 after the 2026-10-06 sync clobbered
configs/baseline.yaml, configs/search_space.yaml and src/features.py with the bare template.

What is verified:
1. baseline.yaml  == runs/run_00_lgbm_baseline/params.json (verbatim config dump of
   src/train.py) with seed_list + features.feature_set_id upgraded to the mm03 snapshot.
2. search_space.yaml carries the documented project values (improve_threshold 0.0005,
   infer_seconds_max 120, no use_target_encoding in allowed.feature_flags, pinned
   LightGBM ranges, max_changes_per_run 2).
3. src/features.py::create_derived_features reproduces the materialised
   data/processed/{train,test}_features.parquet byte-for-byte (values, dtypes, column order),
   and its registry constants mirror reports/feature_manifest.json.

Skips cleanly when the (gitignored) project or its data is absent.

Run from repo root:
  python tests/test_s6e9_restoration.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT = REPO_ROOT / "projects" / "playground-series-s6e9"


def _load_features_module():
    sys.path.insert(0, str(PROJECT / "src"))
    import features as feat  # noqa: PLC0415

    return feat


@unittest.skipUnless(PROJECT.is_dir(), "s6e9 project not present")
class BaselineRestorationTest(unittest.TestCase):
    def test_baseline_matches_params_snapshots(self) -> None:
        cfg = yaml.safe_load((PROJECT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
        run00 = json.loads((PROJECT / "runs" / "run_00_lgbm_baseline" / "params.json").read_text(encoding="utf-8"))
        mm03 = json.loads((PROJECT / "runs" / "mm03__seed42_LightGBM_cv5" / "params.json").read_text(encoding="utf-8"))
        # Expected = earliest verbatim dump, upgraded with the two documented later changes.
        expected = run00
        expected["project"]["seed_list"] = mm03["project"]["seed_list"]
        expected["features"]["feature_set_id"] = mm03["features"]["feature_set_id"]
        self.assertEqual(cfg, expected)

    def test_search_space_documented_values(self) -> None:
        space = yaml.safe_load((PROJECT / "configs" / "search_space.yaml").read_text(encoding="utf-8"))
        policy = space["policy"]
        self.assertEqual(policy["improve_threshold"]["classification_auc"], 0.0005)
        self.assertEqual(policy["tie_margin"], 0.0002)
        self.assertEqual(policy["resources_limits"]["infer_seconds_max"], 120)
        self.assertNotIn("use_target_encoding", space["allowed"]["feature_flags"])
        self.assertEqual(space["search"]["max_changes_per_run"], 2)
        lgbm = space["search"]["hyperparams"]["LightGBM"]
        self.assertEqual(lgbm["learning_rate"], {"type": "log_uniform", "min": 0.02, "max": 0.1})
        self.assertEqual(lgbm["num_leaves"], {"type": "int", "min": 32, "max": 256})
        self.assertEqual(lgbm["min_data_in_leaf"], {"type": "int", "min": 20, "max": 300})


def _diff_frames(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """Fast exact frame diff (per-column .equals; avoids assert_frame_equal's slow object path)."""
    problems: list[str] = []
    if list(a.columns) != list(b.columns):
        problems.append(f"column order differs: {list(a.columns)} != {list(b.columns)}")
        return problems
    for c in a.columns:
        if a[c].dtype != b[c].dtype:
            problems.append(f"{c}: dtype {a[c].dtype} != {b[c].dtype}")
        elif not a[c].equals(b[c]):
            problems.append(f"{c}: values differ")
    return problems


@unittest.skipUnless(
    PROJECT.is_dir() and (PROJECT / "data" / "processed" / "train_features.parquet").is_file(),
    "s6e9 materialised features not present",
)
class FeaturesRestorationTest(unittest.TestCase):
    def test_registry_mirrors_feature_manifest(self) -> None:
        feat = _load_features_module()
        manifest = json.loads((PROJECT / "reports" / "feature_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(feat.DERIVED_COLS, manifest["columns"]["derived_total"])
        self.assertEqual(feat.DERIVED_NUMERIC_COLS, manifest["columns"]["derived_numeric"])
        self.assertEqual(feat.DERIVED_CATEGORICAL_COLS, manifest["columns"]["derived_categorical"])

    def test_recompute_matches_materialised_parquet(self) -> None:
        feat = _load_features_module()
        for split in ("train", "test"):
            with self.subTest(split=split):
                raw = pd.read_parquet(PROJECT / "data" / "processed" / f"{split}.parquet")
                stored = pd.read_parquet(PROJECT / "data" / "processed" / f"{split}_features.parquet")
                rebuilt = feat.create_derived_features(raw)
                self.assertEqual(_diff_frames(rebuilt, stored), [])

    def test_idempotent_on_enriched_frame(self) -> None:
        feat = _load_features_module()
        stored = pd.read_parquet(PROJECT / "data" / "processed" / "train_features.parquet")
        again = feat.create_derived_features(stored)
        self.assertEqual(_diff_frames(again, stored), [])


if __name__ == "__main__":
    unittest.main()
