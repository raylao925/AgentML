"""
templates/src/infer_ensemble.py

Ensemble inference template for AgentML projects.

Goal:
- Load multiple models from ensemble runs
- Apply optimized weights to combine predictions
- Generate submission.csv for ensemble predictions

Agent development steps (commentary):
1) Load ensemble metadata from runs/<ensemble_run_id>/ensemble_metadata.json
2) Load config from first input run
3) Load test data from config `data.test_path`
4) For each run_id in ensemble:
   - Load model.pkl (pipeline + trained model + feature_cols)
   - Apply preprocessing and predict
5) Combine predictions using optimized weights (hill climbing or pre-computed)
6) Build output DataFrame:
   - if `id_cols` exist in test, prepend them
   - then append prediction column (named after target)
7) Write output as CSV for Excel compatibility

Usage:
    python src/infer_ensemble.py --ensemble_run_id ensemble_v1_weighted
    python src/infer_ensemble.py --ensemble_run_id ensemble_v1_weighted --input path/to/test.csv
    python src/infer_ensemble.py --ensemble_run_id ensemble_v1_weighted --optimize_weights
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import features as feat_mod


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_table(path: Path) -> pd.DataFrame:
    """Read table by extension; supports .parquet, .csv, .csv.gz."""
    path = Path(path)
    suf = path.suffix.lower()
    if path.suffix.lower() == ".gz" and path.stem.endswith(".csv"):
        return pd.read_csv(path, compression="gzip")
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported data format: {path}")


def _parse_args():
    p = argparse.ArgumentParser(description="AgentML ensemble inference")
    p.add_argument("--ensemble_run_id", type=str, required=True, help="Ensemble run ID")
    p.add_argument("--input", type=str, default=None, help="Optional override test data path")
    p.add_argument("--output", type=str, default=None, help="Optional output file path")
    p.add_argument("--optimize_weights", action="store_true", help="Use hill climbing to optimize weights")
    return p.parse_args()


def hill_climbing_weights(
    pred_matrix: np.ndarray,
    y_true: np.ndarray | None = None,
    n_iterations: int = 1000,
    step_size: float = 0.05,
    random_state: int = 42,
) -> np.ndarray:
    """
    Optimize ensemble weights using hill climbing algorithm.
    
    Args:
        pred_matrix: Matrix of predictions from multiple models (n_samples, n_models)
        y_true: True labels for optimization (if None, returns uniform weights)
        n_iterations: Number of hill climbing iterations
        step_size: Size of weight perturbation at each step
        random_state: Random seed for reproducibility
    
    Returns:
        Optimized weights array
    """
    from sklearn.metrics import roc_auc_score
    
    n_models = pred_matrix.shape[1]
    
    if y_true is None:
        print("Warning: No target provided for hill climbing, using uniform weights")
        return np.ones(n_models) / n_models
    
    np.random.seed(random_state)
    
    # Initialize with uniform weights
    current_weights = np.ones(n_models) / n_models
    current_score = roc_auc_score(y_true, np.average(pred_matrix, axis=1, weights=current_weights))
    
    best_weights = current_weights.copy()
    best_score = current_score
    
    print(f"Hill climbing initial AUC: {current_score:.6f}")
    
    for iteration in range(n_iterations):
        # Create neighbor by perturbing weights
        perturbation = np.random.randn(n_models) * step_size
        new_weights = current_weights + perturbation
        
        # Ensure non-negative and normalize
        new_weights = np.maximum(new_weights, 0)
        if new_weights.sum() == 0:
            new_weights = np.ones(n_models)
        new_weights = new_weights / new_weights.sum()
        
        # Evaluate new weights
        new_score = roc_auc_score(y_true, np.average(pred_matrix, axis=1, weights=new_weights))
        
        # Accept if better (hill climbing)
        if new_score > current_score:
            current_weights = new_weights
            current_score = new_score
            
            if new_score > best_score:
                best_weights = new_weights.copy()
                best_score = new_score
        
        # Print progress every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration + 1}/{n_iterations}, Best AUC: {best_score:.6f}")
    
    print(f"Hill climbing final AUC: {best_score:.6f}")
    print(f"Optimized weights: {best_weights}")
    
    return best_weights


def load_model(model_path: Path):
    import joblib

    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        return obj.get("pipeline"), obj.get("model"), obj.get("feature_cols", [])
    return None, obj, []


def main():
    args = _parse_args()
    ensemble_dir = PROJECT_ROOT / "runs" / args.ensemble_run_id
    
    # Load ensemble metadata
    metadata_path = ensemble_dir / "ensemble_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Ensemble metadata not found: {metadata_path}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        ensemble_metadata = json.load(f)
    
    run_ids = ensemble_metadata["run_ids"]
    weights = np.array(ensemble_metadata["weights"])
    
    print(f"Ensemble run IDs: {run_ids}")
    print(f"Weights: {weights}")
    
    # Load config from first run
    first_run_dir = PROJECT_ROOT / "runs" / run_ids[0]
    params_path = first_run_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    
    with open(params_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    task_cfg = config.get("task", {})
    target_col = task_cfg.get("target", "prediction")
    data_cfg = config.get("data", {})
    
    input_path = args.input or data_cfg.get("test_path")
    if input_path is None:
        raise ValueError("No test_path provided in config and no --input given.")
    
    input_path = Path(input_path)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    
    test_df = _read_table(input_path)
    
    # Apply derived features if enabled
    features_cfg = config.get("features", {})
    flags = features_cfg.get("flags", {})
    use_derived_features = bool(flags.get("use_derived_features", True))
    
    if use_derived_features:
        test_df = feat_mod.create_derived_features(test_df)
    
    # Load all models and make predictions
    all_preds = []
    
    for i, run_id in enumerate(run_ids):
        run_dir = PROJECT_ROOT / "runs" / run_id
        model_path = run_dir / "artifacts" / "model.pkl"
        
        if not model_path.exists():
            print(f"Warning: Model not found for run {run_id}, skipping")
            continue
        
        print(f"Loading model from run: {run_id}")
        pipeline, model, feature_cols = load_model(model_path)
        
        # Check if all feature columns exist
        missing = [c for c in feature_cols if c not in test_df.columns]
        if missing:
            raise ValueError(f"Test missing feature columns for run {run_id}: {missing}")
        
        X = test_df[feature_cols]
        if pipeline is not None:
            X = pipeline.transform(X)
        
        # Get predictions
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)
            if preds.ndim == 2 and preds.shape[1] == 2:
                preds = preds[:, 1]
        else:
            preds = model.predict(X)
        
        all_preds.append(preds)
        print(f"  Model {run_id} predictions shape: {preds.shape}")
    
    if not all_preds:
        raise ValueError("No valid models found for ensemble")
    
    # Combine predictions using weights
    pred_matrix = np.column_stack(all_preds)
    
    # Ensure weights match number of models
    if len(weights) != len(all_preds):
        print(f"Warning: Adjusting weights from {len(weights)} to {len(all_preds)} models")
        weights = weights[:len(all_preds)]
        weights = weights / weights.sum()
    
    # Use hill climbing to optimize weights if requested
    if args.optimize_weights:
        print("\nOptimizing weights using hill climbing algorithm...")
        
        # Try to load OOF predictions for optimization
        oof_preds_list = []
        for run_id in run_ids:
            run_dir = PROJECT_ROOT / "runs" / run_id
            oof_parquet = run_dir / "artifacts" / "oof_predictions.parquet"
            oof_csv = run_dir / "artifacts" / "oof_predictions.csv"
            
            if oof_parquet.exists():
                oof_df = pd.read_parquet(oof_parquet)
                oof_preds_list.append(oof_df["oof_pred"].values)
            elif oof_csv.exists():
                oof_df = pd.read_csv(oof_csv)
                oof_preds_list.append(oof_df["oof_pred"].values)
        
        if len(oof_preds_list) == len(all_preds):
            # Load target from first OOF file
            first_run_dir = PROJECT_ROOT / "runs" / run_ids[0]
            oof_parquet = first_run_dir / "artifacts" / "oof_predictions.parquet"
            oof_csv = first_run_dir / "artifacts" / "oof_predictions.csv"
            
            if oof_parquet.exists():
                oof_df = pd.read_parquet(oof_parquet)
            else:
                oof_df = pd.read_csv(oof_csv)
            
            target_col_name = config.get("task", {}).get("target", "target")
            if target_col_name in oof_df.columns:
                y_oof = oof_df[target_col_name].values
                if y_oof.dtype == object or y_oof.dtype == str:
                    y_oof = (y_oof == "Yes").astype(int)
                
                oof_matrix = np.column_stack(oof_preds_list)
                weights = hill_climbing_weights(oof_matrix, y_oof)
            else:
                print(f"Warning: Target column '{target_col_name}' not found in OOF, using pre-computed weights")
        else:
            print(f"Warning: Could not load all OOF predictions ({len(oof_preds_list)}/{len(all_preds)}), using pre-computed weights")
    
    ensemble_preds = np.average(pred_matrix, axis=1, weights=weights)
    
    print(f"Ensemble predictions shape: {ensemble_preds.shape}")
    print(f"Ensemble predictions range: [{ensemble_preds.min():.4f}, {ensemble_preds.max():.4f}]")
    
    # Generate output
    out_path = Path(args.output) if args.output else (ensemble_dir / "artifacts" / "submission.csv")
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    id_cols = data_cfg.get("id_cols", []) or []
    id_cols_present = [c for c in id_cols if c in test_df.columns]
    
    out_df = test_df[id_cols_present].copy() if id_cols_present else pd.DataFrame()
    out_df[target_col] = ensemble_preds
    
    # Save as CSV
    if out_path.suffix.lower() == ".csv":
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        out_df.to_parquet(out_path, index=False)
    
    print(f"Submission saved to: {out_path}")
    print(f"Submission shape: {out_df.shape}")
    print(f"Submission columns: {list(out_df.columns)}")


if __name__ == "__main__":
    main()