# data_sources/ — ingestion adapters

Declare **one YAML per data source** here, then run:

```bash
python ../../.agents/skills/agentml/scripts/agentml.py ingest --source data_sources/<name>.yaml
# or, inside the project:
python src/ingest.py --source data_sources/<name>.yaml
```

`src/ingest.py` reads the YAML, dispatches on `source_type`, and always writes:
- `data/processed/train.parquet` (+ `test.parquet` if a test split is declared)
- `reports/data_manifest.json` (rows, columns+dtypes, sha256, `data_version`)

See `references/04_ingestion.md` for the full contract. `example_customer.yaml` is a
copy-me template; delete it once your own source files exist.

Supported `source_type`: `local_files`, `kaggle_competition`, `database`
(needs `sqlalchemy`), `object_store`, `api` (last two are stubs — extend `src/ingest.py`).

## Pipeline position

```
data/raw/*  --[ingest]-->  data/processed/{train,test}.parquet + reports/data_manifest.json
```
Never commit raw customer data (`data/raw/` is gitignored).