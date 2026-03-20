"""
templates/src/ensemble.py

Ensemble template for AgentML projects.

Goal:
- Combine multiple runs using OOF predictions (OOF-safe).
- Support weighted average and/or stacking/blending according to `docs/05_ensemble.md`.

Agent development steps (commentary):
1) Decide ensemble inputs:
   - list of run_ids whose OOF predictions are available:
     - runs/<run_id>/artifacts/oof_predictions.parquet (or .csv)
2) Ensure no leakage:
   - ensemble training must use OOF predictions only
   - never fit weights/meta-model on test data
3) Weighted averaging:
   - choose weights using a constrained search on OOF (e.g. grid search)
4) Stacking:
   - train a meta-model on OOF features (with clear CV for meta-model if needed)
5) Save ensemble artifacts into a new run directory:
   - runs/<run_id_ensemble>/artifacts/ensemble_oof_predictions.*
   - runs/<run_id_ensemble>/artifacts/submission.*
6) Append a record to results.json for ensemble decision traceability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args():
    p = argparse.ArgumentParser(description="AgentML ensemble")
    p.add_argument("--run_ids", type=str, required=True, help="Comma-separated run IDs to ensemble")
    p.add_argument("--output_run_id", type=str, default=None, help="Optional ensemble run id")
    p.add_argument("--method", type=str, default="weighted_average", help="weighted_average|stacking")
    return p.parse_args()


def _load_oof(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    oof_parquet = run_dir / "artifacts" / "oof_predictions.parquet"
    oof_csv = run_dir / "artifacts" / "oof_predictions.csv"
    if oof_parquet.exists():
        df = pd.read_parquet(oof_parquet)
    elif oof_csv.exists():
        df = pd.read_csv(oof_csv)
    else:
        raise FileNotFoundError(f"OOF not found in: {run_dir}")

    # Template assumes oof_pred column + target column exist.
    preds = df["oof_pred"].values
    fold = df.get("fold", None)
    return preds, fold.values if fold is not None else np.full_like(preds, -1)


def main():
    args = _parse_args()
    run_ids = [x.strip() for x in args.run_ids.split(",") if x.strip()]
    if not run_ids:
        raise ValueError("No run_ids provided.")

    if args.output_run_id:
        out_run_id = args.output_run_id
    else:
        out_run_id = f"ensemble_{args.method}_{len(run_ids)}runs"

    out_dir = PROJECT_ROOT / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # Step 1: load OOF predictions matrix
    oof_list = []
    for rid in run_ids:
        run_dir = PROJECT_ROOT / "runs" / rid
        preds, _ = _load_oof(run_dir)
        oof_list.append(preds)
    oof_mat = np.vstack(oof_list)  # shape: [n_models, n_samples]

    # Step 2: choose ensemble method
    if args.method == "weighted_average":
        # Template default weights: uniform.
        # In production, search weights on OOF using primary metric.
        weights = np.ones(oof_mat.shape[0], dtype=float)
        weights = weights / weights.sum()
        ensemble_oof = (weights[:, None] * oof_mat).sum(axis=0)
    else:
        raise NotImplementedError("Stacking template needs meta-model implementation.")

    # Step 3: save ensemble OOF predictions
    import pandas as pd

    oof_df = pd.DataFrame({"ensemble_oof_pred": ensemble_oof})
    oof_df.to_parquet(out_dir / "artifacts" / "ensemble_oof_predictions.parquet", index=False)

    # Step 4: (optional) compute ensemble metrics and append to results.json
    # TODO: read target, compute metrics via src/evaluate.py
    # TODO: append record to results.json


if __name__ == "__main__":
    main()

