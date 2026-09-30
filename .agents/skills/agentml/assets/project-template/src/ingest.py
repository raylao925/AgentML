"""
src/ingest.py — data-source adapter (raw -> data/processed + reports/data_manifest.json).

Contract: references/04_ingestion.md. Declare sources in `data_sources/<name>.yaml`,
then run (from the project root):

    python src/ingest.py --source data_sources/<name>.yaml

Supported source_type:
  local_files         CSV / Parquet / XLSX / JSON already under data/raw/   (implemented)
  kaggle_competition  files fetched via the Kaggle CLI into data/raw/       (implemented)
  database            SQL via a connection string + query (needs sqlalchemy)
  object_store / api  stubs -> extend this file (clear NotImplementedError)

Always produces:
  data/processed/train.parquet  (+ test.parquet when a test split is declared)
  reports/data_manifest.json    (data_version, rows, columns+dtypes, sha256)
"""

from __future__ import annotations

import argparse
import glob as _glob
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------- helpers

def load_source(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "source_type" not in cfg:
        raise ValueError(f"{path}: missing required key 'source_type'")
    return cfg


def _resolve(patterns, base: Path) -> list[Path]:
    if patterns is None:
        return []
    if isinstance(patterns, (str, Path)):
        patterns = [patterns]
    found: list[Path] = []
    for pat in patterns:
        matches = sorted(_glob.glob(str((base / pat).as_posix())))
        found.extend(Path(m) for m in matches)
    return sorted(set(found))


def _read_table(path: Path, csv_opts: dict) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **csv_opts)
    if suffix == ".json":
        return pd.read_json(path)
    opts = dict(csv_opts or {})
    return pd.read_csv(path, **opts)


def _read_concat(files: list[Path], csv_opts: dict) -> pd.DataFrame:
    if not files:
        raise FileNotFoundError("no input files matched the configured globs")
    frames = [_read_table(f, csv_opts) for f in files]
    df = frames[0] if len(frames) == 1 else pd.concat(frames, axis=0, ignore_index=True)
    return df.reset_index(drop=True)


def _apply_pii(df: pd.DataFrame, privacy: dict) -> pd.DataFrame:
    cols = list((privacy or {}).get("pii_columns") or [])
    mode = (privacy or {}).get("hash", "sha256")
    live = [c for c in cols if c in df.columns]
    if not live:
        return df
    if mode == "none":
        print(f"  WARNING: PII columns {live} kept unhashed (privacy.hash: none)")
        return df
    salt = os.environ.get("AGENTML_PII_SALT", "agentml").encode("utf-8")
    df = df.copy()
    for col in live:
        df[col] = (
            df[col].astype("string")
            .fillna("")
            .map(lambda v: hashlib.sha256(salt + str(v).encode("utf-8")).hexdigest())
        )
    print(f"  hashed PII columns: {live}")
    return df


def _binarize_target(df: pd.DataFrame, target: dict) -> pd.DataFrame:
    col = (target or {}).get("column")
    pos = (target or {}).get("positive_label")
    if not col or pos is None or col not in df.columns:
        return df
    df = df.copy()
    df[col] = (df[col].astype("string").str.strip() == str(pos).strip()).astype(int)
    print(f"  binarized target '{col}': positive_label={pos!r} -> 1/0")
    return df


def _split_by_column(df: pd.DataFrame, split_cfg: dict):
    col = split_cfg.get("column")
    if not col or col not in df.columns:
        return df, None
    tr_val = split_cfg.get("train_value", "train")
    te_val = split_cfg.get("test_value", "test")
    train = df[df[col] == tr_val].reset_index(drop=True)
    test = df[df[col] == te_val].reset_index(drop=True)
    return train, (test if len(test) else None)


def _write(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


# ---------------------------------------------------------------- adapters

def ingest_local_files(cfg: dict) -> dict[str, pd.DataFrame]:
    paths = cfg.get("paths", {})
    raw = Path(paths.get("raw", "data/raw"))
    csv_opts = cfg.get("csv_opts") or {}
    files = cfg.get("files", {}) or {}

    train_files = _resolve(files.get("train"), raw)
    test_files = _resolve(files.get("test"), raw)

    if train_files:
        train = _read_concat(train_files, csv_opts)
        test = _read_concat(test_files, csv_opts) if test_files else None
    else:
        # single-source mode: read all raw files, optionally split by a column
        all_files = _resolve(files.get("all") or ["*"], raw)
        all_files = [f for f in all_files if f.suffix.lower() in {".csv", ".parquet", ".xlsx", ".json"}]
        combined = _read_concat(all_files, csv_opts)
        train, test = _split_by_column(combined, cfg.get("split") or {})

    out = {"train": train}
    if test is not None and len(test):
        out["test"] = test
    return out


def ingest_kaggle_competition(cfg: dict) -> dict[str, pd.DataFrame]:
    """Kaggle files are expected under data/raw/ (fetch via the Kaggle CLI first)."""
    competition = cfg.get("competition") or cfg.get("name")
    raw = Path(cfg.get("paths", {}).get("raw", "data/raw"))
    any_raw = _resolve(["*"], raw) if raw.exists() else []
    if not any_raw:
        raise FileNotFoundError(
            f"No raw files in {raw}. Download first, e.g.:\n"
            f"  python -m kaggle competitions download -c {competition} -p {raw}"
        )
    return ingest_local_files(cfg)


def ingest_database(cfg: dict) -> dict[str, pd.DataFrame]:
    try:
        from sqlalchemy import create_engine  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise NotImplementedError(
            "source_type 'database' needs sqlalchemy. Install it and a driver, "
            "e.g. `pip install sqlalchemy psycopg2-binary`."
        ) from exc
    db = cfg.get("database", {})
    engine = create_engine(db["connection_string"])
    train = pd.read_sql(db["train_query"], engine)
    out = {"train": train}
    if db.get("test_query"):
        out["test"] = pd.read_sql(db["test_query"], engine)
    return out


def ingest_stub(cfg: dict) -> dict[str, pd.DataFrame]:
    raise NotImplementedError(
        f"source_type '{cfg.get('source_type')}' is not implemented in the template. "
        f"Extend src/ingest.py and populate the relevant config keys "
        f"(e.g. credentials/endpoint/bucket)."
    )


ADAPTERS = {
    "local_files": ingest_local_files,
    "kaggle_competition": ingest_kaggle_competition,
    "database": ingest_database,
    "object_store": ingest_stub,
    "api": ingest_stub,
}


# ---------------------------------------------------------------- main

def run(cfg: dict) -> int:
    st = cfg["source_type"]
    if st not in ADAPTERS:
        raise ValueError(f"unknown source_type '{st}'; expected one of {sorted(ADAPTERS)}")

    print(f"[ingest] source_type={st} name={cfg.get('name')}")
    frames = ADAPTERS[st](cfg)

    target = cfg.get("target") or {}
    privacy = cfg.get("privacy") or {}
    processed_dir = Path(cfg.get("paths", {}).get("processed", "data/processed"))

    manifest = {
        "data_version": cfg.get("name") or "raw_snapshot",
        "source_type": st,
        "splits": {},
    }
    for split, df in frames.items():
        df = _apply_pii(df, privacy)
        df = _binarize_target(df, target)
        out_path = processed_dir / f"{split}.parquet"
        digest = _write(df, out_path)
        manifest["splits"][split] = {**_profile(df), "path": out_path.as_posix(), "sha256": digest}
        print(f"  wrote {out_path}  rows={len(df)}  sha256={digest[:12]}...")

    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    manifest_path = reports / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ingest] manifest -> {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ingest a data source per references/04_ingestion.md")
    ap.add_argument("--source", required=True, help="Path to data_sources/<name>.yaml")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.source)
    if not src.exists():
        print(f"Error: source file not found: {src}", file=sys.stderr)
        return 1
    return run(load_source(src))


if __name__ == "__main__":
    raise SystemExit(main())