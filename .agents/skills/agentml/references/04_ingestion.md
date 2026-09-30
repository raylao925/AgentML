# 04 — Ingestion (data source adapters)

> Owns **how** raw data becomes `data/processed/{train,test}.parquet`. The concrete source
> is declared per project in `data_sources/<name>.yaml`; the code is `src/ingest.py`.

## Adapter contract

`src/ingest.py` reads a `data_sources/<name>.yaml`, dispatches on `source_type`, and always
produces:
1. `data/processed/train.parquet` (+ `test.parquet` when a test split is declared)
2. `reports/data_manifest.json` — row counts, columns+dtypes, sha256 of outputs, and the
   `data_version` string. Every run's `params.json` references this manifest.

Supported `source_type` values:

| source_type | meaning | notes |
|---|---|---|
| `local_files` | CSV/Parquet/XLSX already under `data/raw/` | default; fully offline |
| `kaggle_competition` | files fetched via the Kaggle CLI | see `sources/kaggle.md` |
| `database` | SQL via a connection string + query | requires `sqlalchemy` + driver |
| `object_store` | S3/GCS/Azure path listing | requires credentials env vars |
| `api` | REST endpoint with pagination | requires credentials env vars |

`local_files` and `kaggle_competition` are implemented in the template. `database` is
implemented when `sqlalchemy` is importable. `object_store` / `api` raise a clear
`NotImplementedError` with the fields they would need — extend `src/ingest.py` for those.

## `data_sources/<name>.yaml` schema

```yaml
source_type: local_files
name: churn_v1
files:                       # glob(s), relative to paths.raw
  train: ["train.csv"]
  test:  ["test.csv"]        # optional
paths:
  raw: "data/raw"
  processed: "data/processed"
csv_opts: {sep: ",", encoding: "utf-8"}
target: {column: churn_flag, positive_label: "Yes", available_after: "T+30d"}
entity: {primary_key: customer_id, time_col: snapshot_date}   # split hints
privacy:
  pii_columns: [full_name, email]
  hash: sha256               # sha256 | none  (one-way hash before writing processed)
```

## Privacy / PII
- Columns in `privacy.pii_columns` are **one-way hashed** before they touch `data/processed`,
  unless `hash: none` is set (then a warning is printed).
- Raw files under `data/raw/` are gitignored and must never be committed.

## Rules
- Ingest is idempotent: same source ⇒ byte-identical `data/processed` (sorted columns,
  stable dtypes).
- Do not apply target-dependent transforms here (that belongs to `05_features`, per-fold).
- After ingest, `doc/01_data_card.md` must reflect the produced schema.

## Output
- `data/processed/{train,test}.parquet`
- `reports/data_manifest.json`