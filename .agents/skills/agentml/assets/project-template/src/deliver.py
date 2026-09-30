"""
src/deliver.py — generalized delivery (references/08_delivery.md).

Modes:
  submission_csv  kaggle-style id+prediction CSV (no threshold)
  batch_scoring   customer-style score + decision column (classification threshold)
  api_contract    feature/dtype/missing-policy spec for a serving layer

Examples (from the project root):
  python src/deliver.py --run_id 20260101_1200_lgbm --mode batch_scoring --threshold auto
  python src/deliver.py --ensemble_run_id v1_ensemble --mode submission_csv

Preprocessing is loaded from the run's saved pipeline, so train/serve preprocess identically.
Thresholds (mode=batch_scoring, --threshold auto) are chosen on OOF predictions, never test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".gz" and path.stem.endswith(".csv"):
        return pd.read_csv(path, compression="gzip")
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported data format: {path}")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model(run_dir: Path):
    import joblib

    model_path = run_dir / "artifacts" / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        return obj.get("pipeline"), obj.get("model"), obj.get("feature_cols", [])
    return None, obj, []


def _ensemble_members(ens_dir: Path):
    """Return [(run_dir, weight), ...] from ensemble_metadata.json (references/07_ensemble.md)."""
    meta = _read_json(ens_dir / "ensemble_metadata.json")
    run_ids = meta["run_ids"]
    weights = meta.get("weights") or [1.0 / len(run_ids)] * len(run_ids)
    return [(PROJECT_ROOT / "runs" / rid, float(w)) for rid, w in zip(run_ids, weights)]


def _predict(pipeline, model, X: pd.DataFrame, family: str) -> np.ndarray:
    if pipeline is not None:
        X = pipeline.transform(X)
    if hasattr(model, "predict_proba") and family.startswith("classification"):
        preds = model.predict_proba(X)
        if getattr(preds, "shape", (0, 0))[1] == 2:
            preds = preds[:, 1]
        return np.asarray(preds)
    return np.asarray(model.predict(X))


def _auto_threshold(run_dir: Path, target_col: str) -> tuple[float, str]:
    """Best-F1 threshold computed on OOF predictions (never the test set)."""
    for name in ("oof_predictions.parquet", "oof_predictions.csv"):
        p = run_dir / "artifacts" / name
        if not p.exists():
            continue
        oof = _read_table(p)
        if target_col not in oof.columns or "oof_pred" not in oof.columns:
            continue
        y = pd.to_numeric(oof[target_col], errors="coerce").to_numpy()
        s = pd.to_numeric(oof["oof_pred"], errors="coerce").to_numpy()
        mask = np.isfinite(y) & np.isfinite(s)
        y, s = y[mask], s[mask]
        if y.size == 0:
            continue
        grid = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (s >= t).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return round(best_t, 6), f"auto:OOF best-F1={best_f1:.4f}"
    return 0.5, "auto:no OOF found -> defaulted to 0.5"


def _project_mode() -> str:
    p = PROJECT_ROOT / "project.yaml"
    if p.exists():
        try:
            import yaml

            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            mode = (cfg.get("project") or {}).get("mode")
            if mode:
                return "submission_csv" if mode == "kaggle" else "batch_scoring"
        except Exception:
            pass
    return "batch_scoring"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentML delivery (references/08_delivery.md)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run_id", type=str, help="Single-model run id under runs/")
    g.add_argument("--ensemble_run_id", type=str, help="Ensemble run id under runs/")
    p.add_argument("--mode", choices=["submission_csv", "batch_scoring", "api_contract"], default=None)
    p.add_argument("--threshold", type=str, default="auto", help="float or 'auto' (OOF best-F1)")
    p.add_argument("--input", type=str, default=None, help="Override input table path")
    p.add_argument("--output", type=str, default=None, help="Override output file path")
    p.add_argument("--out-dir", type=str, default="deliverables")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    mode = args.mode or _project_mode()
    run_id = args.ensemble_run_id or args.run_id
    run_dir = PROJECT_ROOT / "runs" / run_id

    params_path = run_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"params.json not found for run '{run_id}'")
    config = _read_json(params_path)

    task_cfg = config.get("task", {})
    family = task_cfg.get("family", "classification_binary")
    target_col = task_cfg.get("target", "target")
    data_cfg = config.get("data", {})
    id_cols = list(data_cfg.get("id_cols") or [])

    if args.input:
        input_path = Path(args.input)
    elif data_cfg.get("test_path"):
        input_path = Path(data_cfg["test_path"])
    else:
        raise ValueError("No test_path in params.json and no --input given.")
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    df = _read_table(input_path)

    # --- score -----------------------------------------------------------------
    if args.ensemble_run_id:
        members = _ensemble_members(run_dir)
        scores, feat_cols = None, None
        for member_dir, weight in members:
            pipeline, model, fc = _load_model(member_dir)
            feat_cols = fc or feat_cols
            preds = _predict(pipeline, model, df[fc], family)
            scores = preds * weight if scores is None else scores + preds * weight
        print(f"[deliver] ensemble of {len(members)} members (weighted)")
    else:
        pipeline, model, feat_cols = _load_model(run_dir)
        scores = _predict(pipeline, model, df[feat_cols], family)

    missing = [c for c in (feat_cols or []) if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing feature columns: {missing}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    id_cols_present = [c for c in id_cols if c in df.columns]
    return _emit(mode, df, scores, feat_cols, family, target_col, id_cols_present,
                 run_dir, run_id, bool(args.ensemble_run_id), args, data_cfg, input_path)


def _emit(mode, df, scores, feat_cols, family, target_col, id_cols_present,
          run_dir, run_id, is_ensemble, args, data_cfg, input_path) -> int:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "api_contract":
        contract = {
            "task_family": family,
            "target": target_col,
            "run_id": run_id,
            "features": [
                {"name": c, "dtype": str(df[c].dtype), "nullable": bool(df[c].isna().any())}
                for c in (feat_cols or [])
            ],
            "missing_policy": "impute within pipeline (see references/05_features.md)",
        }
        out_path = Path(args.output) if args.output else (out_dir / "api_contract.json")
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[deliver] api_contract -> {out_path}")
        return 0

    if mode == "submission_csv":
        out_df = df[id_cols_present].copy() if id_cols_present else pd.DataFrame(index=df.index)
        col = target_col if target_col != "target" else "prediction"
        out_df[col] = scores
        name = "submission.csv"
    else:  # batch_scoring
        out_df = df[id_cols_present].copy() if id_cols_present else pd.DataFrame(index=df.index)
        out_df["score"] = scores
        if family.startswith("classification"):
            if str(args.threshold) == "auto":
                thr, src = _auto_threshold(run_dir, target_col)
            else:
                thr, src = float(args.threshold), f"fixed:{args.threshold}"
            out_df["predicted_label"] = (np.asarray(scores) >= thr).astype(int)
            print(f"[deliver] threshold={thr} ({src})")
        name = f"scored_{run_id}.csv"

    out_path = Path(args.output) if args.output else (out_dir / name)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        out_df.to_parquet(out_path, index=False)
    print(f"[deliver] mode={mode} -> {out_path}  rows={len(out_df)}")

    manifest = {
        "mode": mode,
        "run_id": run_id,
        "ensemble": is_ensemble,
        "task_family": family,
        "target": target_col,
        "n_rows": int(len(out_df)),
        "input": Path(input_path).as_posix(),
        "output": out_path.as_posix(),
        "data_version": data_cfg.get("data_version"),
    }
    mpath = out_dir / "scoring_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[deliver] manifest -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())