"""
src/evaluate.py

Metric evaluation template for AgentML projects.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

import models as model_mod

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _coerce_binary(y_true: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true)
    if y.dtype == object or y.dtype.kind in ("U", "S"):
        return model_mod.coerce_binary_target(y)
    if len(np.unique(y)) > 2 and set(np.unique(y)).issubset({0, 1}):
        return y.astype(int)
    uniq = np.unique(y)
    if len(uniq) == 2 and not np.issubdtype(y.dtype, np.number):
        return (y == uniq[1]).astype(int)
    return y.astype(float)


def _parse_at_k(metric_name: str) -> int | None:
    m = re.match(r"^(NDCG|MAP)@(\d+)$", metric_name.strip(), re.I)
    if m:
        return int(m.group(2))
    return None


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(-y_score)
    y_true = np.asarray(y_true)[order[:k]]
    dcg = 0.0
    for i, rel in enumerate(y_true):
        dcg += (2**rel - 1) / np.log2(i + 2)
    ideal = np.sort(np.asarray(y_true))[::-1][:k]
    idcg = 0.0
    for i, rel in enumerate(ideal):
        idcg += (2**rel - 1) / np.log2(i + 2)
    return float(dcg / idcg) if idcg > 0 else 0.0


def _map_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(-y_score)
    y_true = np.asarray(y_true)[order[:k]]
    hits = 0
    prec_sum = 0.0
    for i, rel in enumerate(y_true, start=1):
        if rel > 0:
            hits += 1
            prec_sum += hits / i
    return float(prec_sum / max(hits, 1)) if hits else 0.0


def _compute_single_metric(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task_family: str,
) -> float:
    metric = name.strip()
    metric_upper = metric.upper()

    if metric_upper == "AUC":
        y_bin = _coerce_binary(y_true)
        return float(roc_auc_score(y_bin, y_pred))

    if metric_upper == "LOGLOSS":
        y_bin = _coerce_binary(y_true)
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(log_loss(y_bin, p))

    if metric_upper == "BRIER":
        y_bin = _coerce_binary(y_true).astype(float)
        p = np.clip(y_pred, 0.0, 1.0)
        return float(np.mean((y_bin - p) ** 2))

    if metric_upper in ("ACC", "ACCURACY"):
        y_bin = _coerce_binary(y_true)
        labels = (y_pred >= 0.5).astype(int)
        return float(accuracy_score(y_bin, labels))

    if metric_upper == "F1":
        y_bin = _coerce_binary(y_true)
        labels = (y_pred >= 0.5).astype(int)
        return float(f1_score(y_bin, labels, zero_division=0))

    if metric_upper == "RMSE":
        return float(np.sqrt(mean_squared_error(y_true.astype(float), y_pred.astype(float))))

    if metric_upper == "MAE":
        return float(mean_absolute_error(y_true.astype(float), y_pred.astype(float)))

    if metric_upper in ("R2", "R²"):
        return float(r2_score(y_true.astype(float), y_pred.astype(float)))

    at_k = _parse_at_k(metric)
    if at_k is not None:
        if metric.upper().startswith("NDCG"):
            return _ndcg_at_k(y_true, y_pred, at_k)
        return _map_at_k(y_true, y_pred, at_k)

    if task_family == "classification_multiclass" and metric_upper.startswith("AUC"):
        return float(roc_auc_score(y_true, y_pred, multi_class="ovr", average="weighted"))

    raise ValueError(f"Unsupported metric: {name}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task_family: str,
    primary_metric: str,
    secondary_metrics: list[str] | None = None,
) -> dict[str, Any]:
    secondary_metrics = secondary_metrics or []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    primary = _compute_single_metric(
        primary_metric, y_true, y_pred, task_family=task_family
    )
    secondary: dict[str, float] = {}
    for name in secondary_metrics:
        if name == primary_metric:
            continue
        try:
            secondary[name] = _compute_single_metric(
                name, y_true, y_pred, task_family=task_family
            )
        except ValueError:
            continue

    return {"primary": primary, "secondary": secondary}


def evaluate_run(run_dir: str | Path) -> dict[str, Any]:
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
    return compute_metrics(
        y_true,
        y_pred,
        task_family=task_family,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
    )


def _parse_args():
    p = argparse.ArgumentParser(description="AgentML evaluate run from OOF artifacts")
    p.add_argument("--run_id", type=str, required=True, help="Run ID under runs/")
    p.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    return p.parse_args()


def main():
    args = _parse_args()
    run_dir = PROJECT_ROOT / "runs" / args.run_id
    metrics = evaluate_run(run_dir)
    payload = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
