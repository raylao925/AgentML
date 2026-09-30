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

    if args.method == "weighted_average":
        weights = np.ones(oof_mat.shape[1], dtype=float)
        weights = weights / weights.sum()
        ensemble_oof = np.average(oof_mat, axis=1, weights=weights)
    else:
        raise NotImplementedError("Stacking template needs meta-model implementation.")

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
