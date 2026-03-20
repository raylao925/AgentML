"""
templates/src/train_multi_model.py

Multi-model training template for AgentML projects.

Goal:
- Train multiple models (LightGBM, XGBoost, CatBoost, etc.) on the same data
- Compare performance across models
- Save all model artifacts for later ensemble
- Support configurable model selection via config

Agent development steps (commentary):
1) Parse args and load config
2) Load train/test data
3) Get list of models to train from config (or use defaults)
4) For each model:
   - Build fold-safe preprocessing pipeline
   - Train model with CV
   - Save OOF predictions and model artifacts
   - Record metrics
5) Compare all models and output summary
6) Append records to results.json

Usage:
    python src/train_multi_model.py --config configs/baseline.yaml
    python src/train_multi_model.py --config configs/baseline.yaml --models "LightGBM,XGBoost"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import data as data_mod
import evaluate as eval_mod
import features as feat_mod


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args():
    p = argparse.ArgumentParser(description="AgentML multi-model training")
    p.add_argument("--config", type=str, default="configs/baseline.yaml")
    p.add_argument("--models", type=str, default=None, help="Comma-separated model names to train (e.g., 'LightGBM,XGBoost')")
    p.add_argument("--run_id_prefix", type=str, default=None, help="Prefix for run IDs")
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


def get_available_models(task_family: str) -> list[str]:
    """Get list of available models for a given task family."""
    model_map = {
        "classification_binary": ["LightGBM", "XGBoost", "CatBoost", "LogisticRegression"],
        "classification_multiclass": ["LightGBM", "XGBoost", "CatBoost"],
        "regression": ["LightGBM", "XGBoost", "CatBoost", "ElasticNet"],
        "ranking": ["LGBMRanker", "XGBRanker"],
        "time_series": ["LightGBM", "XGBoost"],
    }
    return model_map.get(task_family, ["LightGBM"])


def create_model(model_name: str, model_params: dict, seed: int = 42):
    """Create a model instance based on model name and parameters."""
    
    if model_name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=model_params.get("n_estimators", 2000),
            learning_rate=model_params.get("learning_rate", 0.05),
            max_depth=model_params.get("max_depth", -1),
            num_leaves=model_params.get("num_leaves", 64),
            min_data_in_leaf=model_params.get("min_data_in_leaf", 50),
            subsample=model_params.get("subsample", 0.8),
            colsample_bytree=model_params.get("colsample_bytree", 0.8),
            reg_lambda=model_params.get("reg_lambda", 1.0),
            random_state=seed,
            verbose=-1,
        )
    
    elif model_name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=model_params.get("n_estimators", 2000),
            learning_rate=model_params.get("learning_rate", 0.05),
            max_depth=model_params.get("max_depth", 6),
            min_child_weight=model_params.get("min_child_weight", 1.0),
            subsample=model_params.get("subsample", 0.8),
            colsample_bytree=model_params.get("colsample_bytree", 0.8),
            reg_lambda=model_params.get("reg_lambda", 1.0),
            random_state=seed,
            verbose=0,
        )
    
    elif model_name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=model_params.get("n_estimators", 2000),
            learning_rate=model_params.get("learning_rate", 0.05),
            depth=model_params.get("max_depth", 6),
            l2_leaf_reg=model_params.get("reg_lambda", 1.0),
            random_seed=seed,
            verbose=0,
        )
    
    elif model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=model_params.get("C", 1.0),
            max_iter=model_params.get("max_iter", 1000),
            random_state=seed,
        )
    
    elif model_name == "ElasticNet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(
            alpha=model_params.get("alpha", 1.0),
            l1_ratio=model_params.get("l1_ratio", 0.5),
            max_iter=model_params.get("max_iter", 1000),
            random_state=seed,
        )
    
    else:
        # Default to LightGBM
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=model_params.get("n_estimators", 2000),
            learning_rate=model_params.get("learning_rate", 0.05),
            max_depth=model_params.get("max_depth", -1),
            num_leaves=model_params.get("num_leaves", 64),
            min_data_in_leaf=model_params.get("min_data_in_leaf", 50),
            subsample=model_params.get("subsample", 0.8),
            colsample_bytree=model_params.get("colsample_bytree", 0.8),
            reg_lambda=model_params.get("reg_lambda", 1.0),
            random_state=seed,
            verbose=-1,
        )


def train_single_model(
    model_name: str,
    config: dict,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    folds: list,
    seed: int,
    run_id_prefix: str | None = None,
) -> dict:
    """Train a single model and return results."""
    
    task_cfg = config.get("task", {})
    task_family = task_cfg.get("family", "classification_binary")
    primary_metric = task_cfg.get("primary_metric", "AUC")
    secondary_metrics = task_cfg.get("secondary_metrics", [])
    
    data_cfg = config.get("data", {})
    out_cfg = config.get("output", {})
    runs_dir = Path(out_cfg.get("runs_dir", "runs"))
    results_path = PROJECT_ROOT / out_cfg.get("results_path", "results.json")
    save_oof = bool(out_cfg.get("save_oof", True))
    save_model = bool(out_cfg.get("save_model", True))
    
    # Get model-specific params
    search_space_path = PROJECT_ROOT / "configs" / "search_space.yaml"
    if search_space_path.exists():
        import yaml
        with open(search_space_path, "r", encoding="utf-8") as f:
            search_space = yaml.safe_load(f)
        hyperparams = search_space.get("hyperparams", {}).get(model_name, {})
    else:
        hyperparams = {}
    
    model_cfg = config.get("model", {})
    model_params = model_cfg.get("params", {})
    # Merge with search space params
    for key, value in hyperparams.items():
        if key not in model_params:
            if isinstance(value, dict) and "type" in value:
                # Use default value from range
                if value["type"] == "log_uniform":
                    model_params[key] = np.exp((np.log(value["min"]) + np.log(value["max"])) / 2)
                elif value["type"] == "uniform":
                    model_params[key] = (value["min"] + value["max"]) / 2
                elif value["type"] == "int":
                    model_params[key] = (value["min"] + value["max"]) // 2
            else:
                model_params[key] = value
    
    n_splits = len(folds)
    
    # Generate run ID
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    if run_id_prefix:
        run_id = f"{run_id_prefix}_{model_name}_cv{n_splits}"
    else:
        run_id = f"{ts}_{model_name}_cv{n_splits}"
    
    run_dir = PROJECT_ROOT / runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training model: {model_name}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}")
    
    oof_preds = np.full(len(train_df), np.nan)
    oof_fold_idx = np.full(len(train_df), -1, dtype=np.int32)
    per_fold_primary: list[float] = []
    models = []
    
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        print(f"\n  Fold {fold_idx + 1}/{n_splits}")
        
        # Fit preprocessors on train fold only
        pipeline, feature_names = feat_mod.build_feature_pipeline(config, feature_cols, train_df.iloc[train_idx])
        X_tr, X_va, y_tr, y_va, _, _ = feat_mod.prepare_fold(
            config,
            train_idx=train_idx,
            valid_idx=valid_idx,
            train_df=train_df,
            target_col=target_col,
            feature_cols=feature_cols,
            pipeline_and_names=(pipeline, feature_names),
        )
        
        # Convert target to binary if needed
        if y_tr.dtype == object or y_tr.dtype == str:
            y_tr_binary = (y_tr == "Yes").astype(int)
            y_va_binary = (y_va == "Yes").astype(int)
        else:
            y_tr_binary = y_tr.astype(int)
            y_va_binary = y_va.astype(int)
        
        # Create and train model
        model = create_model(model_name, model_params, seed)
        
        training_cfg = config.get("training", {})
        early_stopping = training_cfg.get("early_stopping", True)
        early_stopping_rounds = training_cfg.get("early_stopping_rounds", 200)
        verbose = training_cfg.get("verbose", 50)
        
        if early_stopping and hasattr(model, "fit"):
            try:
                if model_name in ["LightGBM", "LGBMRanker"]:
                    import lightgbm as lgb
                    model.fit(
                        X_tr, y_tr_binary,
                        eval_set=[(X_va, y_va_binary)],
                        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
                    )
                elif model_name in ["XGBoost", "XGBRanker"]:
                    model.fit(
                        X_tr, y_tr_binary,
                        eval_set=[(X_va, y_va_binary)],
                        verbose=verbose,
                    )
                elif model_name == "CatBoost":
                    model.fit(
                        X_tr, y_tr_binary,
                        eval_set=(X_va, y_va_binary),
                        verbose=verbose,
                    )
                else:
                    model.fit(X_tr, y_tr_binary)
            except Exception as e:
                print(f"    Warning: Early stopping failed: {e}")
                model.fit(X_tr, y_tr_binary)
        else:
            model.fit(X_tr, y_tr_binary)
        
        # Predict valid fold
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_va)
            if preds.ndim == 2 and preds.shape[1] == 2:
                preds = preds[:, 1]
        else:
            preds = model.predict(X_va)
        
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
        print(f"    Fold {fold_idx + 1} {primary_metric}: {fold_metrics['primary']:.6f}")
    
    primary_mean = float(np.mean(per_fold_primary)) if per_fold_primary else 0.0
    primary_std = float(np.std(per_fold_primary)) if per_fold_primary else 0.0
    
    print(f"\n  {model_name} CV {primary_metric}: {primary_mean:.6f} (+/- {primary_std:.6f})")
    
    # Save params/metrics
    model_config = config.copy()
    model_config["model"] = {
        "name": model_name,
        "objective": model_cfg.get("objective", "binary"),
        "params": model_params,
    }
    
    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
    
    metrics_payload = {
        "primary": {"name": primary_metric, "mean": primary_mean, "std": primary_std, "per_fold": per_fold_primary},
        "secondary": {},
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    
    # Save OOF
    if save_oof:
        oof_df = pd.DataFrame({"oof_pred": oof_preds, "fold": oof_fold_idx, target_col: train_df[target_col].values})
        oof_df.to_parquet(run_dir / "artifacts" / "oof_predictions.parquet", index=False)
    
    # Save model
    if save_model and models:
        import joblib
        model_path = run_dir / "artifacts" / "model.pkl"
        model_obj = {
            "pipeline": pipeline,
            "model": models[0],
            "feature_cols": feature_cols,
            "feature_names": feature_names,
        }
        joblib.dump(model_obj, model_path)
        
        # Save feature list
        feature_list_path = run_dir / "artifacts" / "feature_list.json"
        with open(feature_list_path, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2, ensure_ascii=False)
    
    # Append to results.json
    record = {
        "run_id": run_id,
        "datetime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "task": {
            "family": task_family,
            "target": target_col,
            "primary_metric": primary_metric,
            "secondary_metrics": secondary_metrics,
        },
        "data": {"data_version": data_cfg.get("data_version", ""), "row_count_train": int(len(train_df))},
        "cv": {"cv_type": config.get("cv", {}).get("cv_type", ""), "n_splits": n_splits, "random_state": seed},
        "features": {"feature_set_id": config.get("features", {}).get("feature_set_id", ""), "dropped_columns": data_cfg.get("drop_cols", [])},
        "model": {"name": model_name, "objective": model_cfg.get("objective", "")},
        "metrics": {"primary": {"name": primary_metric, "mean": primary_mean, "std": primary_std}},
        "resources": {},
        "artifacts": {
            "run_dir": str(run_dir).replace("\\", "/"),
            "params_path": str(run_dir / "params.json").replace("\\", "/"),
            "metrics_path": str(run_dir / "metrics.json").replace("\\", "/"),
            "oof_path": str(run_dir / "artifacts" / "oof_predictions.parquet").replace("\\", "/"),
        },
        "decision": {"status": "keep", "reason": f"Multi-model training: {model_name}"},
        "notes_short": f"Multi-model training: {model_name}",
    }
    _append_to_results_json(results_path, record)
    
    return {
        "run_id": run_id,
        "model_name": model_name,
        "primary_mean": primary_mean,
        "primary_std": primary_std,
        "per_fold": per_fold_primary,
    }


def main():
    args = _parse_args()
    config = data_mod.load_config(args.config)
    seed = int(config.get("project", {}).get("seed", 42))
    np.random.seed(seed)
    
    task_cfg = config.get("task", {})
    task_family = task_cfg.get("family", "classification_binary")
    target_col = task_cfg.get("target", "target")
    
    data_cfg = config.get("data", {})
    train_path = data_cfg.get("train_path", "data/raw/train.csv")
    test_path = data_cfg.get("test_path")
    
    train_df, test_df = data_mod.load_data(train_path, test_path)
    if target_col not in train_df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    
    # Apply derived features if enabled
    features_cfg = config.get("features", {})
    flags = features_cfg.get("flags", {})
    use_derived_features = bool(flags.get("use_derived_features", True))
    
    if use_derived_features:
        train_df = feat_mod.create_derived_features(train_df)
        if test_df is not None:
            test_df = feat_mod.create_derived_features(test_df)
    
    id_cols = data_cfg.get("id_cols", []) or []
    drop_cols = data_cfg.get("drop_cols", []) or []
    feature_cols = feat_mod.get_feature_columns(train_df, target_col, id_cols=id_cols, drop_cols=drop_cols)
    
    folds = data_mod.get_cv_folds(config, train_df, target_col)
    
    # Determine which models to train
    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        # Get models from search_space.yaml or use defaults
        search_space_path = PROJECT_ROOT / "configs" / "search_space.yaml"
        if search_space_path.exists():
            import yaml
            with open(search_space_path, "r", encoding="utf-8") as f:
                search_space = yaml.safe_load(f)
            model_names = search_space.get("allowed", {}).get("models_by_task", {}).get(task_family, ["LightGBM"])
        else:
            model_names = get_available_models(task_family)
    
    print(f"Training models: {model_names}")
    print(f"Task family: {task_family}")
    print(f"Target column: {target_col}")
    print(f"Number of folds: {len(folds)}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Train all models
    results = []
    for model_name in model_names:
        try:
            result = train_single_model(
                model_name=model_name,
                config=config,
                train_df=train_df,
                feature_cols=feature_cols,
                target_col=target_col,
                folds=folds,
                seed=seed,
                run_id_prefix=args.run_id_prefix,
            )
            results.append(result)
        except Exception as e:
            print(f"\nError training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*60}")
    print("MULTI-MODEL TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'AUC (mean)':<15} {'AUC (std)':<15}")
    print("-" * 50)
    
    # Sort by performance
    results.sort(key=lambda x: x["primary_mean"], reverse=True)
    
    for result in results:
        print(f"{result['model_name']:<20} {result['primary_mean']:<15.6f} {result['primary_std']:<15.6f}")
    
    print(f"\nBest model: {results[0]['model_name']} (AUC: {results[0]['primary_mean']:.6f})")
    print(f"\nAll runs saved. Use ensemble.py to combine predictions.")


if __name__ == "__main__":
    main()