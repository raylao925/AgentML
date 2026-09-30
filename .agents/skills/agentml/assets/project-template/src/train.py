"""
src/train.py — CV training with fold-safe preprocessing and run artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import data as data_mod
import evaluate as eval_mod
import features as feat_mod
import models as model_mod


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args():
    p = argparse.ArgumentParser(description="AgentML train")
    p.add_argument("--config", type=str, default="configs/baseline.yaml")
    p.add_argument("--run_id", type=str, default=None)
    return p.parse_args()


def _append_to_results_json(results_path: Path, record: dict):
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        ledger = json.loads(content) if content else []
    else:
        ledger = []
    ledger.append(record)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)


def main():
    args = _parse_args()
    config = data_mod.load_config(args.config)
    seed = int(config.get("project", {}).get("seed", 42))
    np.random.seed(seed)

    task_cfg = config.get("task", {})
    task_family = task_cfg.get("family", "classification_binary")
    target_col = task_cfg.get("target", "target")
    primary_metric = task_cfg.get("primary_metric", "AUC")
    secondary_metrics = task_cfg.get("secondary_metrics", [])

    data_cfg = config.get("data", {})
    train_path = data_cfg.get("train_path", "data/raw/train.csv")
    test_path = data_cfg.get("test_path")

    out_cfg = config.get("output", {})
    runs_dir = Path(out_cfg.get("runs_dir", "runs"))
    results_path = PROJECT_ROOT / out_cfg.get("results_path", "results.json")
    save_oof = bool(out_cfg.get("save_oof", True))
    save_model = bool(out_cfg.get("save_model", True))

    train_df, _test_df = data_mod.load_data(train_path, test_path)
    if target_col not in train_df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    features_cfg = config.get("features", {})
    flags = features_cfg.get("flags", {})
    if bool(flags.get("use_derived_features", False)):
        train_df = feat_mod.create_derived_features(train_df)

    id_cols = data_cfg.get("id_cols", []) or []
    drop_cols = data_cfg.get("drop_cols", []) or []
    feature_cols = feat_mod.get_feature_columns(
        train_df, target_col, id_cols=id_cols, drop_cols=drop_cols
    )

    folds = data_mod.get_cv_folds(config, train_df, target_col)
    n_splits = len(folds)

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "LightGBM")
    model_params = dict(model_cfg.get("params", {}))
    training_cfg = config.get("training", {})

    run_id = args.run_id
    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        run_id = f"{ts}_{model_name}_cv{n_splits}"

    run_dir = PROJECT_ROOT / runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    oof_preds = np.full(len(train_df), np.nan)
    oof_fold_idx = np.full(len(train_df), -1, dtype=np.int32)
    per_fold_primary: list[float] = []
    models: list = []
    pipeline = None
    feature_names: list[str] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        pipeline, feature_names = feat_mod.build_feature_pipeline(
            config, feature_cols, train_df.iloc[train_idx]
        )
        X_tr, X_va, y_tr, y_va, _, _ = feat_mod.prepare_fold(
            config,
            train_idx=train_idx,
            valid_idx=valid_idx,
            train_df=train_df,
            target_col=target_col,
            feature_cols=feature_cols,
            pipeline_and_names=(pipeline, feature_names),
        )

        model = model_mod.create_model(
            model_name,
            model_params,
            seed,
            use_gpu=False,
            task_family=task_family,
        )
        model_mod.fit_model(
            model, model_name, X_tr, y_tr, X_va, y_va, training_cfg
        )
        preds = model_mod.predict_scores(model, X_va)

        oof_preds[valid_idx] = preds
        oof_fold_idx[valid_idx] = fold_idx
        models.append(model)

        fold_metrics = eval_mod.compute_metrics(
            y_va,
            preds,
            task_family=task_family,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
        )
        per_fold_primary.append(float(fold_metrics["primary"]))

    primary_mean = float(np.mean(per_fold_primary)) if per_fold_primary else 0.0
    primary_std = float(np.std(per_fold_primary)) if per_fold_primary else 0.0

    oof_metrics = eval_mod.compute_metrics(
        train_df[target_col].values,
        oof_preds,
        task_family=task_family,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
    )

    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    metrics_payload = {
        "primary": {
            "name": primary_metric,
            "mean": primary_mean,
            "std": primary_std,
            "per_fold": per_fold_primary,
            "oof": float(oof_metrics["primary"]),
        },
        "secondary": oof_metrics.get("secondary", {}),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    if save_oof:
        oof_df = pd.DataFrame(
            {
                "oof_pred": oof_preds,
                "fold": oof_fold_idx,
                target_col: train_df[target_col].values,
            }
        )
        oof_df.to_parquet(run_dir / "artifacts" / "oof_predictions.parquet", index=False)

    if save_model and models:
        joblib.dump(
            {
                "pipeline": pipeline,
                "model": models[-1],
                "feature_cols": feature_cols,
                "feature_names": feature_names,
            },
            run_dir / "artifacts" / "model.pkl",
        )
        with open(run_dir / "artifacts" / "feature_list.json", "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2, ensure_ascii=False)

    record = {
        "run_id": run_id,
        "datetime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "task": {
            "family": task_family,
            "target": target_col,
            "primary_metric": primary_metric,
            "secondary_metrics": secondary_metrics,
        },
        "data": {
            "data_version": data_cfg.get("data_version", ""),
            "row_count_train": int(len(train_df)),
        },
        "cv": {
            "cv_type": config.get("cv", {}).get("cv_type", ""),
            "n_splits": n_splits,
            "random_state": seed,
        },
        "features": {
            "feature_set_id": config.get("features", {}).get("feature_set_id", ""),
            "dropped_columns": drop_cols,
        },
        "model": {
            "name": model_name,
            "objective": model_cfg.get("objective", ""),
        },
        "metrics": {
            "primary": {"name": primary_metric, "mean": primary_mean, "std": primary_std}
        },
        "resources": {},
        "artifacts": {
            "run_dir": str(run_dir).replace("\\", "/"),
            "params_path": str(run_dir / "params.json").replace("\\", "/"),
            "metrics_path": str(run_dir / "metrics.json").replace("\\", "/"),
            "oof_path": str(run_dir / "artifacts" / "oof_predictions.parquet").replace(
                "\\", "/"
            ),
        },
        "decision": {"status": "keep", "reason": "Completed CV training run."},
        "notes_short": f"CV run {model_name} n_splits={n_splits}",
    }
    _append_to_results_json(results_path, record)


if __name__ == "__main__":
    main()
