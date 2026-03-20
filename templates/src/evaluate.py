"""
templates/src/evaluate.py

Metric evaluation template for AgentML projects.

Goal:
- Keep metric definitions stable.
- Compute primary metric + secondary metrics from y_true and y_pred.
- Provide CLI to re-evaluate a run from stored OOF artifacts.

Agent development steps (commentary):
1) Implement metric functions (AUC/LogLoss/Brier/RMSE/MAE/etc.) according to `docs/00_problem_statement.md`
   and task family expectations.
2) Do NOT change the meaning of PRIMARY_METRIC. Only fix bugs or efficiency issues.
3) Handle target type:
   - binary: ensure y_true is numeric 0/1 before sklearn classification metrics.
   - multiclass: ensure y_pred is probability or label encoding consistently.
4) Provide `evaluate_run(run_id)`:
   - read `runs/<run_id>/artifacts/oof_predictions.*`
   - compute metrics
   - print or save result
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task_family: str,
    primary_metric: str,
    secondary_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute primary/secondary metrics.

    Template suggestion:
    - For binary classification:
      - primary AUC uses roc_auc_score on y_true and y_pred probabilities
      - LogLoss and Brier use probabilities
    - For regression:
      - primary RMSE uses sqrt(mean((y-yhat)^2))
    """
    secondary_metrics = secondary_metrics or []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out: dict[str, Any] = {"primary": 0.0, "secondary": {}}

    # TODO: Implement metrics per template. Keep definitions stable.
    raise NotImplementedError("Fill in metric definitions according to docs/00_problem_statement.md")


def evaluate_run(run_dir: str):
    """
    Load OOF predictions from a run folder and compute metrics.
    """
    from pathlib import Path
    import json
    import pandas as pd

    run_dir = Path(run_dir)
    params_path = run_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    with open(params_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    task_cfg = config.get("task", {})
    task_family = task_cfg.get("family", "classification_binary")
    primary_metric = task_cfg.get("primary_metric", "AUC")
    secondary_metrics = task_cfg.get("secondary_metrics", [])
    target_col = task_cfg.get("target", "target")

    oof_parquet = run_dir / "artifacts" / "oof_predictions.parquet"
    oof_csv = run_dir / "artifacts" / "oof_predictions.csv"
    if oof_parquet.exists():
        df = pd.read_parquet(oof_parquet)
    elif oof_csv.exists():
        df = pd.read_csv(oof_csv)
    else:
        raise FileNotFoundError("OOF predictions not found.")

    y_true = df[target_col].values
    y_pred = df["oof_pred"].values
    metrics = compute_metrics(
        y_true,
        y_pred,
        task_family=task_family,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
    )
    return metrics

