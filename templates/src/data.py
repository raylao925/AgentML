"""
templates/src/data.py

Data module template for AgentML projects.

Goal:
- Provide config-driven data loading and fold construction.
- Enforce CV authority from `docs/03_cv_strategy.md` and follow AGENT_RULES (no leakage).

Agent development steps (commentary):
1) Parse YAML config (`configs/baseline.yaml` and/or composed config from OpenClaw).
2) Load train/test from `data/raw/*` (dispatch by file extension).
3) Build folds according to `config.cv.cv_type` and keys:
   - `group_key` => GroupKFold (entity-level leakage protection)
   - `time_col` => TimeSeriesSplit (strict chronological ordering)
   - `stratify_col` => StratifiedKFold for classification
4) Return a list of (train_idx, valid_idx) indices.
5) Keep this module limited to split/schema logic only.
   - Feature fitting and target encoding belong in `src/features.py`.
   - Model training loop belongs in `src/train.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit


def _project_root() -> Path:
    # projects/<project_slug>/src/data.py => project root is parent of src/
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = _project_root() / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_table(path: Path) -> pd.DataFrame:
    """
    Dispatch reader by extension.
    This template supports common formats:
    - .csv, .csv.gz
    - .parquet
    - .feather
    - .xlsx/.xls
    """
    suf = path.suffix.lower()
    if path.suffix.lower() == ".gz" and path.stem.endswith(".csv"):
        return pd.read_csv(path, compression="gzip")
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf == ".feather":
        return pd.read_feather(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported data format: {path}")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterate through all columns of a dataframe and modify the data type
    to reduce memory usage.
    
    Args:
        df: Input DataFrame
        verbose: Whether to print memory usage statistics
    
    Returns:
        DataFrame with optimized memory usage
    
    Example:
        >>> df = pd.read_csv("large_file.csv")
        >>> df = reduce_mem_usage(df)
        Memory usage of dataframe is 125.43 MB
        Memory usage after optimization is: 45.21 MB
        Decreased by 63.9%
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def load_data(
    train_path: str,
    test_path: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load datasets specified by config.
    Expected paths are relative to project root unless absolute.
    """
    root = _project_root()
    tr = root / train_path
    if not tr.exists():
        raise FileNotFoundError(f"Train data not found: {tr}")
    train_df = _read_table(tr)

    test_df = None
    if test_path:
        te = root / test_path
        if te.exists():
            test_df = _read_table(te)
    return train_df, test_df


def get_cv_folds(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    target_col: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build folds based on config.

    IMPORTANT:
    - Do not change split logic here without updating `docs/03_cv_strategy.md`.
    - All fold-safe preprocessing should be fit only on the train indices.
    """
    cv_cfg = config.get("cv", {})
    data_cfg = config.get("data", {})

    cv_type = cv_cfg.get("cv_type", "StratifiedKFold")
    n_splits = int(cv_cfg.get("n_splits", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(cv_cfg.get("random_state", config.get("seed", 42)))

    group_key = cv_cfg.get("group_key") or data_cfg.get("group_key")
    time_col = cv_cfg.get("time_col") or data_cfg.get("time_col")
    stratify_col = cv_cfg.get("stratify_col")

    n = len(train_df)

    # Choose y / group arrays
    y = train_df[stratify_col].values if stratify_col else train_df[target_col].values
    group = train_df[group_key].values if group_key else None

    if cv_type == "TimeSeriesSplit":
        if time_col is None or time_col not in train_df.columns:
            raise ValueError("TimeSeriesSplit requires `time_col` in config and in data.")
        # Sort in-place so caller's train_df is chronologically ordered; indices then match.
        train_df.sort_values(time_col, inplace=True)
        train_df.reset_index(drop=True, inplace=True)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(np.arange(n)))

    if cv_type == "GroupKFold":
        if group is None:
            raise ValueError("GroupKFold requires group_key in config and data.")
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(np.arange(n), groups=group))

    if cv_type == "StratifiedKFold":
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(skf.split(np.arange(n), y))

    # Default: plain KFold
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(kf.split(np.arange(n)))

