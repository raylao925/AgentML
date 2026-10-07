"""
src/ensemble.py — Combine multiple runs using OOF predictions (OOF-safe).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as eval_mod

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args():
    p = argparse.ArgumentParser(description="AgentML ensemble")
    p.add_argument("--run_ids", type=str, required=True, help="Comma-separated run IDs to ensemble")
    p.add_argument("--output_run_id", type=str, default=None, help="Optional ensemble run id")
    p.add_argument("--method", type=str, default="weighted_average", help="weighted_average|stacking")
    return p.parse_args()


def _load_oof(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    oof_parquet = run_dir / "artifacts" / "oof_predictions.parquet"
    oof_csv = run_dir / "artifacts" / "oof_predictions.csv"
    if oof_parquet.exists():
        df = pd.read_parquet(oof_parquet)
    elif oof_csv.exists():
        df = pd.read_csv(oof_csv)
    else:
        raise FileNotFoundError(f"OOF not found in: {run_dir}")

    preds = df["oof_pred"].values
    return df, preds


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

    oof_list: list[np.ndarray] = []
    base_df: pd.DataFrame | None = None
    target_col = "target"

    for rid in run_ids:
        run_dir = PROJECT_ROOT / "runs" / rid
        df, preds = _load_oof(run_dir)
        if base_df is None:
            base_df = df
            params_path = run_dir / "params.json"
            if params_path.exists():
                with open(params_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                target_col = cfg.get("task", {}).get("target", "target")
        oof_list.append(preds)

    oof_mat = np.column_stack(oof_list)
    meta_block: dict | None = None

    if args.method == "weighted_average":
        weights = np.ones(oof_mat.shape[1], dtype=float)
        weights = weights / weights.sum()
        ensemble_oof = np.average(oof_mat, axis=1, weights=weights)
    elif args.method == "stacking":
        # Fold-safe stacking: base OOFs are already out-of-fold, and the meta model is
        # fit on the OTHER folds' OOF rows to predict each held-out fold (meta-OOF).
        # Supported families: classification_binary (logistic meta) and regression (ridge meta).
        from sklearn.linear_model import LogisticRegression, Ridge

        if base_df is None or target_col not in base_df.columns:
            raise ValueError("stacking requires the target column in the base OOF frame")
        y_meta = base_df[target_col].values
        task_family = "classification_binary"
        first_params = PROJECT_ROOT / "runs" / run_ids[0] / "params.json"
        if first_params.exists():
            with open(first_params, "r", encoding="utf-8") as f:
                task_family = json.load(f).get("task", {}).get("family", "classification_binary")
        if task_family not in ("classification_binary", "regression"):
            raise NotImplementedError(
                f"stacking currently supports classification_binary/regression, got {task_family}"
            )

        if "fold" in base_df.columns:
            fold_ids = base_df["fold"].values
        else:
            fold_ids = np.zeros(len(y_meta), dtype=int)
        unique_folds = np.unique(fold_ids)
        preds = np.zeros(len(y_meta), dtype=float)
        coef_sum = np.zeros(oof_mat.shape[1], dtype=float)
        intercept_sum = 0.0
        n_fitted = 0
        for fid in unique_folds:
            tr = np.where(fold_ids != fid)[0]
            va = np.where(fold_ids == fid)[0]
            if not len(tr) or not len(va):
                continue
            if task_family == "regression":
                meta = Ridge(alpha=1.0)
            else:
                meta = LogisticRegression(max_iter=1000)
            meta.fit(oof_mat[tr], y_meta[tr])
            preds[va] = meta.predict(oof_mat[va]) if task_family == "regression" else (
                meta.predict_proba(oof_mat[va])[:, 1]
            )
            coef_sum += np.asarray(meta.coef_, dtype=float).ravel()
            intercept_sum += float(np.asarray(meta.intercept_).ravel()[0])
            n_fitted += 1
        if n_fitted == 0:
            raise ValueError("stacking: no usable folds found in the base OOF frame")

        coefs = coef_sum / n_fitted
        intercept = intercept_sum / n_fitted
        abs_sum = float(np.abs(coefs).sum()) or 1.0
        weights = np.abs(coefs) / abs_sum  # legacy fallback blend (documented in metadata)
        ensemble_oof = preds
        meta_block = {
            "type": "logistic" if task_family == "classification_binary" else "ridge",
            "family": task_family,
            "coef": coefs.tolist(),
            "intercept": intercept,
            "fitted_folds": int(n_fitted),
            "note": "inference applies sigmoid(coef . p + intercept) to member probabilities",
        }
    else:
        raise ValueError(f"Unknown ensemble method: {args.method} (weighted_average|stacking)")

    oof_df = pd.DataFrame({"ensemble_oof_pred": ensemble_oof})
    if base_df is not None and target_col in base_df.columns:
        oof_df[target_col] = base_df[target_col].values
    if base_df is not None and "fold" in base_df.columns:
        oof_df["fold"] = base_df["fold"].values
    oof_df.to_parquet(out_dir / "artifacts" / "ensemble_oof_predictions.parquet", index=False)

    ensemble_metadata = {
        "run_ids": run_ids,
        "weights": weights.tolist(),
        "method": args.method,
        "output_run_id": out_run_id,
        "n_models": len(run_ids),
    }
    if meta_block is not None:
        ensemble_metadata["meta"] = meta_block
    with open(out_dir / "ensemble_metadata.json", "w", encoding="utf-8") as f:
        json.dump(ensemble_metadata, f, indent=2, ensure_ascii=False)

    metrics_payload: dict = {}
    if base_df is not None and target_col in base_df.columns:
        first_run_dir = PROJECT_ROOT / "runs" / run_ids[0]
        with open(first_run_dir / "params.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        task_cfg = config.get("task", {})
        metrics_payload = eval_mod.compute_metrics(
            base_df[target_col].values,
            ensemble_oof,
            task_family=task_cfg.get("family", "classification_binary"),
            primary_metric=task_cfg.get("primary_metric", "AUC"),
            secondary_metrics=task_cfg.get("secondary_metrics", []),
        )
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "primary": {
                        "name": task_cfg.get("primary_metric", "AUC"),
                        "value": metrics_payload["primary"],
                    },
                    "secondary": metrics_payload.get("secondary", {}),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    out_cfg_path = PROJECT_ROOT / "results.json"
    if run_ids:
        with open(PROJECT_ROOT / "runs" / run_ids[0] / "params.json", "r", encoding="utf-8") as f:
            ref_cfg = json.load(f)
        out_cfg = ref_cfg.get("output", {})
        out_cfg_path = PROJECT_ROOT / out_cfg.get("results_path", "results.json")

    record = {
        "run_id": out_run_id,
        "datetime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "task": ref_cfg.get("task", {}) if run_ids else {},
        "ensemble": {
            "method": args.method,
            "input_run_ids": run_ids,
            "weights": weights.tolist(),
        },
        "metrics": metrics_payload,
        "artifacts": {
            "ensemble_metadata": str(out_dir / "ensemble_metadata.json").replace("\\", "/"),
            "ensemble_oof": str(
                out_dir / "artifacts" / "ensemble_oof_predictions.parquet"
            ).replace("\\", "/"),
        },
        "decision": {"status": "keep", "reason": "Ensemble OOF combination."},
        "notes_short": f"Ensemble {args.method} over {len(run_ids)} runs",
    }
    _append_to_results_json(out_cfg_path, record)


if __name__ == "__main__":
    main()
