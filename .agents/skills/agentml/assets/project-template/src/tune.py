"""
src/tune.py — Optuna hyper-parameter search over the locked CV (HPO runtime).

Contract:
- Reads configs/search_space.yaml -> search.hyperparams.<Model>; baseline params are the start point.
- Objective = mean per-fold primary metric over the LOCKED folds from src/data.py (never re-splits).
- Guardrail G3: this file must never reference held-out evaluation data paths (tuning-path rule).
- Outputs: runs/<study_id>/{study.sqlite3, trials.csv, best_params.json} + ONE ledger record for
  the study; with --final-run (default) a full CV retrain of the winning params under
  runs/hpo_<model>_seed<seed>_cv<n> + a second ledger record (attribution: one param set per run).
- Keep/discard gate: compare the final-run OOF against the current best member before promoting;
  promote by editing configs/baseline.yaml::model.params (manual, auditable).
- Queue position: idea family #3 in references/10_iteration_loop.md §3.

Usage:
  python src/tune.py --config configs/baseline.yaml --model LightGBM --n-trials 60 --timeout-s 7200
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import data as data_mod
import evaluate as eval_mod
import features as feat_mod
import models as model_mod

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Observed fold std of a typical incumbent family (adjust per project in doc/06); a trial
# trailing the incumbent by more than PRUNE_SIGMA * FOLD_STD_REFERENCE after >= 3 folds is
# uninteresting -> prune.
FOLD_STD_REFERENCE = 0.001
PRUNE_SIGMA = 10.0


def _parse_args():
    p = argparse.ArgumentParser(description="AgentML HPO (Optuna) over the locked CV")
    p.add_argument("--config", type=str, default="configs/baseline.yaml")
    p.add_argument("--space", type=str, default="configs/search_space.yaml")
    p.add_argument("--model", type=str, default="LightGBM")
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--timeout-s", type=int, default=7200)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--study-id", type=str, default=None)
    p.add_argument(
        "--final-run",
        dest="final_run",
        action="store_true",
        help="retrain the winner and write standard run artifacts + ledger record",
    )
    p.add_argument("--no-final-run", dest="final_run", action="store_false")
    p.set_defaults(final_run=True)
    return p.parse_args()


def _append_to_results_json(results_path: Path, record: dict) -> None:
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        ledger = json.loads(content) if content else []
    else:
        ledger = []
    ledger.append(record)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)


def _load_space(space_path: str, model_name: str) -> dict:
    cfg = data_mod.load_config(space_path)
    spaces = (cfg.get("search") or {}).get("hyperparams") or {}
    return spaces.get(model_name) or {}


def _suggest_params(trial, space: dict, baseline: dict) -> dict:
    params = dict(baseline)
    for key, spec in space.items():
        s_type = (spec or {}).get("type")
        if s_type == "log_uniform":
            params[key] = trial.suggest_float(key, float(spec["min"]), float(spec["max"]), log=True)
        elif s_type == "uniform":
            params[key] = trial.suggest_float(key, float(spec["min"]), float(spec["max"]))
        elif s_type == "int":
            params[key] = trial.suggest_int(key, int(spec["min"]), int(spec["max"]))
        elif s_type == "choice":
            params[key] = trial.suggest_categorical(key, list(spec["values"]))
        else:
            raise ValueError(f"Unsupported search space type for '{key}': {s_type}")
    return params


def _base_record(run_id: str, config: dict, model_name: str, n_splits: int, seed: int) -> dict:
    task_cfg = config.get("task", {})
    data_cfg = config.get("data", {})
    return {
        "run_id": run_id,
        "datetime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "task": {
            "family": task_cfg.get("family", "classification_binary"),
            "target": task_cfg.get("target", "target"),
            "primary_metric": task_cfg.get("primary_metric", "AUC"),
            "secondary_metrics": task_cfg.get("secondary_metrics", []),
        },
        "data": {
            "data_version": data_cfg.get("data_version", ""),
            "row_count_train": None,
        },
        "cv": {
            "cv_type": config.get("cv", {}).get("cv_type", ""),
            "n_splits": n_splits,
            "random_state": int(config.get("cv", {}).get("random_state", seed)),
        },
        "features": {
            "feature_set_id": config.get("features", {}).get("feature_set_id", ""),
            "dropped_columns": data_cfg.get("drop_cols", []) or [],
        },
        "model": {"name": model_name, "objective": config.get("model", {}).get("objective", "")},
        "metrics": {"primary": {"name": task_cfg.get("primary_metric", "AUC"), "mean": 0.0, "std": 0.0}},
        "resources": {},
        "artifacts": {},
        "decision": {"status": "keep", "reason": "HPO study over the locked CV."},
        "notes_short": "",
    }


def main():
    args = _parse_args()
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("optuna is required for tuning: pip install optuna") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = data_mod.load_config(args.config)
    seed = int(config.get("project", {}).get("seed", 42))
    if args.seed is not None:
        seed = int(args.seed)
    np.random.seed(seed)

    task_cfg = config.get("task", {})
    task_family = task_cfg.get("family", "classification_binary")
    target_col = task_cfg.get("target", "target")
    primary_metric = task_cfg.get("primary_metric", "AUC")
    higher_is_better = bool(task_cfg.get("metric_higher_is_better", True))
    secondary_metrics = task_cfg.get("secondary_metrics", [])

    data_cfg = config.get("data", {})
    features_cfg = config.get("features", {})
    if bool((features_cfg.get("flags") or {}).get("use_derived_features", False)):
        raise SystemExit("use_derived_features must stay false: the materialised parquet already carries it")

    train_df, _ = data_mod.load_data(data_cfg.get("train_path"), None)
    id_cols = data_cfg.get("id_cols", []) or []
    drop_cols = data_cfg.get("drop_cols", []) or []
    feature_cols = feat_mod.get_feature_columns(train_df, target_col, id_cols=id_cols, drop_cols=drop_cols)
    folds = data_mod.get_cv_folds(config, train_df, target_col)
    n_splits = len(folds)

    model_name = args.model
    baseline_params = dict((config.get("model") or {}).get("params", {}))
    space = _load_space(args.space, model_name)
    if not space:
        raise SystemExit(f"No search space for model '{model_name}' in {args.space}")
    training_cfg = config.get("training", {})

    study_id = args.study_id or (
        "tune_" + model_name + "_" + str(args.n_trials) + "trials_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    )
    study_dir = PROJECT_ROOT / "runs" / study_id
    (study_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    storage = "sqlite:///" + str(study_dir / "study.sqlite3").replace("\\", "/")
    direction = "maximize" if higher_is_better else "minimize"

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest_params(trial, space, baseline_params)
        started = time.time()
        per_fold: list[float] = []
        for fold_idx, (train_idx, valid_idx) in enumerate(folds):
            pipeline, feature_names = feat_mod.build_feature_pipeline(
                config, feature_cols, train_df.iloc[train_idx]
            )
            X_tr, X_va, y_tr, y_va, _, _ = feat_mod.prepare_fold(
                config, train_idx=train_idx, valid_idx=valid_idx, train_df=train_df,
                target_col=target_col, feature_cols=feature_cols,
                pipeline_and_names=(pipeline, feature_names),
            )
            model = model_mod.create_model(model_name, params, seed, use_gpu=False, task_family=task_family)
            model_mod.fit_model(model, model_name, X_tr, y_tr, X_va, y_va, training_cfg)
            preds = model_mod.predict_scores(model, X_va)
            fold_metrics = eval_mod.compute_metrics(
                y_va, preds, task_family=task_family,
                primary_metric=primary_metric, secondary_metrics=secondary_metrics,
            )
            per_fold.append(float(fold_metrics["primary"]))

            running = float(np.mean(per_fold))
            trial.set_user_attr("per_fold", list(per_fold))
            trial.report(running, fold_idx)
            if fold_idx >= 2:
                try:
                    incumbent = trial.study.best_value
                except ValueError:
                    incumbent = None
                if incumbent is not None:
                    gap = (incumbent - running) if higher_is_better else (running - incumbent)
                    if gap > PRUNE_SIGMA * FOLD_STD_REFERENCE:
                        raise optuna.TrialPruned()
            if trial.should_prune():
                raise optuna.TrialPruned()
            budget = float((config.get("policy") or {}).get("resources_limits", {}).get("train_seconds_max", 1800))
            if time.time() - started > budget:
                raise optuna.TrialPruned()

        value = float(np.mean(per_fold))
        trial.set_user_attr("fold_mean", value)
        trial.set_user_attr("fold_std", float(np.std(per_fold)))
        trial.set_user_attr("wall_s", round(time.time() - started, 1))
        return value

    study = optuna.create_study(
        study_name=study_id, storage=storage, direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=float(args.timeout_s), gc_after_trial=True)

    best = study.best_trial
    best_params = dict(best.params)
    best_value = float(best.user_attrs.get("fold_mean", best.value))
    best_std = float(best.user_attrs.get("fold_std", 0.0))

    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
    with open(study_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "primary_metric": primary_metric,
                "fold_mean": best_value,
                "fold_std": best_std,
                "per_fold": best.user_attrs.get("per_fold", []),
                "n_trials": len(study.trials),
                "n_complete": len([t for t in study.trials if t.state.name == "COMPLETE"]),
                "n_pruned": len([t for t in study.trials if t.state.name == "PRUNED"]),
                "params": best_params,
            },
            f, indent=2, ensure_ascii=False,
        )

    record = _base_record(study_id, config, model_name, n_splits, seed)
    record["model"]["params"] = best_params
    record["metrics"]["primary"] = {"name": primary_metric, "mean": best_value, "std": best_std}
    record["data"]["row_count_train"] = int(len(train_df))
    record["artifacts"] = {
        "run_dir": str(study_dir).replace("\\", "/"),
        "best_params_path": str(study_dir / "best_params.json").replace("\\", "/"),
        "trials_path": str(study_dir / "trials.csv").replace("\\", "/"),
    }
    record["decision"] = {
        "status": "keep",
        "reason": f"Optuna study ({len(study.trials)} trials) over the locked CV.",
    }
    record["notes_short"] = f"HPO {model_name}: best fold-mean {primary_metric}={best_value:.6f}"
    _append_to_results_json(
        PROJECT_ROOT / config.get("output", {}).get("results_path", "results.json"), record
    )

    print(f"[tune] study={study_id} best fold-mean {primary_metric}={best_value:.6f} +/- {best_std:.6f}")
    print(f"[tune] best params: {best_params}")

    if not args.final_run:
        return

    # --- final retrain with the winning params (attribution: one param set per run) ---
    import joblib

    final_cfg = copy.deepcopy(config)
    final_cfg.setdefault("model", {})["params"] = {**baseline_params, **best_params}
    run_id = f"hpo_{model_name}_seed{seed}_cv{n_splits}"
    run_dir = PROJECT_ROOT / "runs" / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    oof_preds = np.full(len(train_df), np.nan)
    oof_fold_idx = np.full(len(train_df), -1, dtype=np.int32)
    per_fold_final: list[float] = []
    models = []
    feature_names: list[str] = []
    pipeline = None
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        pipeline, feature_names = feat_mod.build_feature_pipeline(
            config, feature_cols, train_df.iloc[train_idx]
        )
        X_tr, X_va, y_tr, y_va, _, _ = feat_mod.prepare_fold(
            config, train_idx=train_idx, valid_idx=valid_idx, train_df=train_df,
            target_col=target_col, feature_cols=feature_cols,
            pipeline_and_names=(pipeline, feature_names),
        )
        model = model_mod.create_model(
            model_name, {**baseline_params, **best_params}, seed,
            use_gpu=False, task_family=task_family,
        )
        model_mod.fit_model(model, model_name, X_tr, y_tr, X_va, y_va, training_cfg)
        preds = model_mod.predict_scores(model, X_va)
        oof_preds[valid_idx] = preds
        oof_fold_idx[valid_idx] = fold_idx
        models.append(model)
        fold_metrics = eval_mod.compute_metrics(
            y_va, preds, task_family=task_family,
            primary_metric=primary_metric, secondary_metrics=secondary_metrics,
        )
        per_fold_final.append(float(fold_metrics["primary"]))

    oof_metrics = eval_mod.compute_metrics(
        train_df[target_col].values, oof_preds,
        task_family=task_family, primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
    )
    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(final_cfg, f, indent=2, ensure_ascii=False)
    metrics_payload = {
        "primary": {
            "name": primary_metric,
            "mean": float(np.mean(per_fold_final)),
            "std": float(np.std(per_fold_final)),
            "per_fold": per_fold_final,
            "oof": float(oof_metrics["primary"]),
        },
        "secondary": oof_metrics.get("secondary", {}),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    pd.DataFrame(
        {"oof_pred": oof_preds, "fold": oof_fold_idx, target_col: train_df[target_col].values}
    ).to_parquet(run_dir / "artifacts" / "oof_predictions.parquet", index=False)
    if models:
        joblib.dump(
            {"pipeline": pipeline, "model": models[-1], "feature_cols": feature_cols,
             "feature_names": feature_names},
            run_dir / "artifacts" / "model.pkl",
        )
        with open(run_dir / "artifacts" / "feature_list.json", "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2, ensure_ascii=False)

    final_record = _base_record(run_id, final_cfg, model_name, n_splits, seed)
    final_record["model"]["params"] = {**baseline_params, **best_params}
    final_record["metrics"]["primary"] = {
        "name": primary_metric,
        "mean": float(np.mean(per_fold_final)),
        "std": float(np.std(per_fold_final)),
    }
    final_record["data"]["row_count_train"] = int(len(train_df))
    final_record["artifacts"] = {
        "run_dir": str(run_dir).replace("\\", "/"),
        "params_path": str(run_dir / "params.json").replace("\\", "/"),
        "metrics_path": str(run_dir / "metrics.json").replace("\\", "/"),
        "oof_path": str(run_dir / "artifacts" / "oof_predictions.parquet").replace("\\", "/"),
    }
    final_record["decision"] = {
        "status": "keep",
        "reason": (
            "HPO winner retrained; compare OOF "
            f"{float(oof_metrics['primary']):.6f} against the shipped member before promoting."
        ),
    }
    final_record["notes_short"] = f"HPO retrain {model_name} OOF={float(oof_metrics['primary']):.6f}"
    _append_to_results_json(
        PROJECT_ROOT / config.get("output", {}).get("results_path", "results.json"), final_record
    )
    print(f"[tune] final run: {run_id} OOF {primary_metric}={float(oof_metrics['primary']):.6f}")


if __name__ == "__main__":
    main()
