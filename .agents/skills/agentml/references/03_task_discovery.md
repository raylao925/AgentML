# 03 — Task Discovery (customer mode)

> **When**: `project.yaml` `project.mode: customer`, or the user drops a dataset without a
> stated target/metric (Kaggle-style clues absent). For `mode: kaggle` use `sources/kaggle.md`.
>
> **Why**: Kaggle gives you the metric and target for free. A customer dataset does not —
> guessing them silently is the single largest source of wasted modeling effort.

## Protocol

### Step 1 — Schema scan (no external calls)
Run a read-only scan of `data/raw/*`:
- dtype, cardinality, null rate per column
- datetime candidates (name + parseability), primary-key candidates (unique / near-unique)
- target candidates (low-cardinality, name hints, class balance)

Write the result into `doc/01_data_card.md`.

### Step 2 — Infer hypotheses (with confidence)
Propose, each with a confidence and the evidence:
- task family (binary / multiclass / regression / ranking / time-series)
- target column candidate(s)
- primary metric candidate(s) + whether higher-is-better
- CV candidate (group key / time col / stratify) — feeds Step 4

### Step 3 — Must-ask interview
Ask the user (do not assume):
1. **What** to predict and **at what decision point** (T-0)?
2. How is the label created, and how long until it is **available** (`T+?`)?
3. Which error is more expensive — a false positive or a false negative? (→ metric / threshold)
4. Is inference **batch** or **real-time**? Any latency budget?
5. Any **PII / compliance** limits? May we use external/web search on this data?
6. Which entity/time defines the split (customer, store, case, snapshot date)?

### Step 4 — Confirm and write back
Write the confirmed answers into `doc/00_problem_statement.md` and set
`confirmed_by_user: true`. Only then may modeling proceed (Lifecycle Gate A).

### Step 5 — External search is opt-in
If `project.yaml` `policy.allow_external_search: true` (or the user explicitly allows it),
the agent may run web/domain search to suggest features. Otherwise it uses **schema metadata
only** — never send raw customer rows anywhere.

## Output
- Updated `doc/00_problem_statement.md` (`confirmed_by_user: true`)
- Updated `doc/01_data_card.md` (schema, PII, leakage list, target availability)
- A candidate metric + CV proposal handed to `04_cv_strategy.md`

## Do not
- Do not invent a target when the label creation process is unclear — ask.
- Do not run external search on PII-bearing data unless explicitly allowed.
- Do not treat the metric as fixed until Step 4 is confirmed.