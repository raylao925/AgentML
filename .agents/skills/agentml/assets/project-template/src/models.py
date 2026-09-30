"""
src/models.py

Model factory shared by train.py and train_multi_model.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def get_available_models(task_family: str) -> list[str]:
    model_map = {
        "classification_binary": ["LightGBM", "XGBoost", "CatBoost", "LogisticRegression"],
        "classification_multiclass": ["LightGBM", "XGBoost", "CatBoost"],
        "regression": ["LightGBM", "XGBoost", "CatBoost", "ElasticNet"],
        "ranking": ["LGBMRanker", "XGBRanker"],
        "time_series": ["LightGBM", "XGBoost"],
    }
    return model_map.get(task_family, ["LightGBM"])


def create_model(
    model_name: str,
    model_params: dict[str, Any],
    seed: int = 42,
    use_gpu: bool = False,
    *,
    task_family: str = "classification_binary",
):
    device = "gpu" if use_gpu else "cpu"

    if model_name == "LightGBM":
        import lightgbm as lgb

        objective = "binary"
        if task_family == "regression":
            objective = "regression"
        elif task_family == "classification_multiclass":
            objective = "multiclass"
        lgb_params = {
            "objective": objective,
            "n_estimators": model_params.get("n_estimators", 2000),
            "learning_rate": model_params.get("learning_rate", 0.05),
            "max_depth": model_params.get("max_depth", -1),
            "num_leaves": model_params.get("num_leaves", 64),
            "min_data_in_leaf": model_params.get("min_data_in_leaf", 50),
            "subsample": model_params.get("subsample", 0.8),
            "colsample_bytree": model_params.get("colsample_bytree", 0.8),
            "reg_lambda": model_params.get("reg_lambda", 1.0),
            "random_state": seed,
            "verbose": -1,
        }
        if use_gpu:
            lgb_params["device"] = device
            lgb_params["gpu_platform_id"] = model_params.get("gpu_platform_id", 0)
            lgb_params["gpu_device_id"] = model_params.get("gpu_device_id", 0)
        if task_family == "regression":
            return lgb.LGBMRegressor(**lgb_params)
        return lgb.LGBMClassifier(**lgb_params)

    if model_name == "XGBoost":
        import xgboost as xgb

        if task_family == "regression":
            objective = "regression:squarederror"
            cls = xgb.XGBRegressor
        else:
            objective = "binary:logistic"
            cls = xgb.XGBClassifier
        xgb_params = {
            "objective": objective,
            "n_estimators": model_params.get("n_estimators", 2000),
            "learning_rate": model_params.get("learning_rate", 0.05),
            "max_depth": model_params.get("max_depth", 6),
            "min_child_weight": model_params.get("min_child_weight", 1.0),
            "subsample": model_params.get("subsample", 0.8),
            "colsample_bytree": model_params.get("colsample_bytree", 0.8),
            "reg_lambda": model_params.get("reg_lambda", 1.0),
            "random_state": seed,
            "verbosity": 0,
        }
        if use_gpu:
            xgb_params["tree_method"] = "gpu_hist"
            xgb_params["gpu_id"] = model_params.get("gpu_id", 0)
        return cls(**xgb_params)

    if model_name == "CatBoost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cb_params = {
            "iterations": model_params.get("n_estimators", 2000),
            "learning_rate": model_params.get("learning_rate", 0.05),
            "depth": model_params.get("max_depth", 6),
            "l2_leaf_reg": model_params.get("reg_lambda", 1.0),
            "random_seed": seed,
            "verbose": 0,
        }
        if use_gpu:
            cb_params["task_type"] = "GPU"
            cb_params["devices"] = str(model_params.get("gpu_id", 0))
        if task_family == "regression":
            return CatBoostRegressor(**cb_params)
        return CatBoostClassifier(**cb_params)

    if model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=model_params.get("C", 1.0),
            max_iter=model_params.get("max_iter", 1000),
            random_state=seed,
        )

    if model_name == "ElasticNet":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(
            alpha=model_params.get("alpha", 1.0),
            l1_ratio=model_params.get("l1_ratio", 0.5),
            max_iter=model_params.get("max_iter", 1000),
            random_state=seed,
        )

    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=model_params.get("n_estimators", 200),
        learning_rate=model_params.get("learning_rate", 0.05),
        random_state=seed,
        verbose=-1,
    )


def coerce_binary_target(y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.dtype == object or y_arr.dtype.kind in ("U", "S"):
        return (y_arr == "Yes").astype(int)
    return y_arr.astype(int)


def predict_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(X)
        if preds.ndim == 2 and preds.shape[1] == 2:
            return preds[:, 1]
        return preds
    return model.predict(X)


def fit_model(
    model,
    model_name: str,
    X_tr,
    y_tr,
    X_va,
    y_va,
    training_cfg: dict[str, Any],
) -> None:
    y_tr_b = coerce_binary_target(y_tr)
    y_va_b = coerce_binary_target(y_va)
    early_stopping = training_cfg.get("early_stopping", True)
    early_stopping_rounds = training_cfg.get("early_stopping_rounds", 200)
    verbose = training_cfg.get("verbose", 50)

    if early_stopping and hasattr(model, "fit"):
        try:
            if model_name in ["LightGBM", "LGBMRanker"]:
                import lightgbm as lgb

                model.fit(
                    X_tr,
                    y_tr_b,
                    eval_set=[(X_va, y_va_b)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
                )
            elif model_name in ["XGBoost", "XGBRanker"]:
                model.fit(X_tr, y_tr_b, eval_set=[(X_va, y_va_b)], verbose=verbose)
            elif model_name == "CatBoost":
                model.fit(X_tr, y_tr_b, eval_set=(X_va, y_va_b), verbose=verbose)
            else:
                model.fit(X_tr, y_tr_b)
        except Exception:
            model.fit(X_tr, y_tr_b)
    else:
        model.fit(X_tr, y_tr_b)
