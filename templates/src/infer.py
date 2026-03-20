"""
templates/src/infer.py

Inference/submission template for AgentML projects.

Goal:
- Load `runs/<run_id>/artifacts/model.pkl` (pipeline + trained model + feature_cols schema)
- Load test data from config `data.test_path`
- Run preprocessing + prediction
- Create submission output:
  - preserve `data.id_cols` in the output (if provided)
  - include prediction column

Agent development steps (commentary):
1) Determine `input_path` from:
   - CLI `--input` if provided
   - else from saved run `params.json` -> config `data.test_path`
2) Read saved model artifact and extract:
   - preprocessing pipeline (if any)
   - trained model
   - feature_cols used during training
3) Transform test with pipeline and predict probabilities/values.
4) Build output DataFrame:
   - if `id_cols` exist in test, prepend them
   - then append `prediction`
5) Write output:
   - default `submission.csv` should be real CSV (use to_csv) for Excel compatibility
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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
    p = argparse.ArgumentParser(description="AgentML inference")
    p.add_argument("--run_id", type=str, required=True)
    p.add_argument("--model", type=str, default=None, help="Optional override model.pkl path")
    p.add_argument("--input", type=str, default=None, help="Optional override test data path")
    p.add_argument("--output", type=str, default=None, help="Optional output file path")
    return p.parse_args()


def load_model(model_path: Path):
    import joblib

    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        return obj.get("pipeline"), obj.get("model"), obj.get("feature_cols", [])
    return None, obj, []


def main():
    args = _parse_args()
    run_dir = PROJECT_ROOT / "runs" / args.run_id

    model_path = Path(args.model) if args.model else (run_dir / "artifacts" / "model.pkl")
    pipeline, model, feature_cols = load_model(model_path)

    params_path = run_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    with open(params_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    task_family = config.get("task", {}).get("family", "classification_binary")
    data_cfg = config.get("data", {})
    input_path = args.input or data_cfg.get("test_path")
    if input_path is None:
        raise ValueError("No test_path provided in config and no --input given.")
    input_path = Path(input_path)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    test_df = _read_table(input_path)
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"Test missing feature columns: {missing}")

    X = test_df[feature_cols]
    if pipeline is not None:
        X = pipeline.transform(X)

    if hasattr(model, "predict_proba") and task_family in ("classification_binary", "classification_multiclass"):
        preds = model.predict_proba(X)
        if preds.shape[1] == 2:
            preds = preds[:, 1]
    else:
        preds = model.predict(X)

    out_path = Path(args.output) if args.output else (run_dir / "artifacts" / "submission.csv")
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    id_cols = data_cfg.get("id_cols", []) or []
    id_cols_present = [c for c in id_cols if c in test_df.columns]

    out_df = test_df[id_cols_present].copy() if id_cols_present else pd.DataFrame()
    out_df["prediction"] = preds

    # Important: CSV output should be real CSV.
    if out_path.suffix.lower() == ".csv":
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        out_df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    main()

