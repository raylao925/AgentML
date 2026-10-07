# 10 — Iteration Loop (multi-round creative improvement)

> Code: queue template in payload `doc/06_experiment_log.md` §G · lifecycle steps in
> `01_lifecycle.md` · HPO runtime `src/tune.py` · stacking `src/ensemble.py --method stacking`.
> Purpose: turn "one pipeline pass" into a **driven loop** that keeps inventing and testing
> higher-CV hypotheses until a stopping criterion fires.

## 1) Where the loop sits

```
01_lifecycle steps A–D   Plan → Execute → Log → Keep/Discard
        │
        ▼
E  Queue refresh   ── re-rank memory/NEXT.md (live, authoritative) + mirror into doc/06 §G (always, every round)
        │
        ▼
F  Divergence      ── ONLY when the queue is empty: invent candidates from §3 idea bank
        │
        ▼
G  Stop check      ── plateau / budget / deadline / exhaustion (§5)
        │
   not stopped → back to A with the top queue item
   stopped     → retrospective (§5) and hand over
```

The loop is **agent-driven**: finishing a run is not finishing the work. The exit is a documented
stopping criterion, not "the pipeline ran once".

## 2) Round queue protocol

- **Live location: `memory/NEXT.md`** — the authoritative queue, rewritten every round
  (`01_lifecycle.md` Step H). **Archive: `doc/06_experiment_log.md` §G** — one `G.x` block per round,
  appended for the record. On disagreement `memory/NEXT.md` wins and §G is behind.
  A project without a §G block after its first run is out of contract (`check_ssot` R8),
  and a round without a `memory/` update fails `guardrails` G10.
- Read the queue (then `WINS.md` + `FAILURES.md`) at session start before planning anything
  (`01_lifecycle.md` Step 0.5).
- Every candidate carries five fields:
  `hypothesis · expected Δ (magnitude + reasoning) · cost (wall-clock) · change family · evidence link`.
- Rank by **expected value per unit cost**, not by excitement.
- Depth cap: **5 live candidates**. More stay in the idea bank (§3) until promoted.
- Refresh rule: re-rank after every run — a DISCARD often demotes its whole family (the evidence
  says that lever is weak here); a KEEP often promotes siblings of the winning family. Record the
  DISCARD in `memory/FAILURES.md` and the KEEP in `memory/WINS.md` in the same Step H pass.
- The queue answers "what next" without re-deriving context; it is the agent's working memory
  across turns (write it down — do not expect the reader to remember it).

## 3) Idea bank (creative method families, in typical EV order)

"EV" = expected Δ per unit cost on a fresh tabular problem; always check the cheap evidence first
(probe OOF correlations, feature importance, per-fold spread) before spending a round.

| # | Family | Methods | Pays when | Evidence to check first |
|---|---|---|---|---|
| 1 | Features | interactions (gate×continuous), fixed-width bins, fold-safe target/frequency encoding, drop noise cols, external/original data (kaggle mode, if rules allow) | signal is concentrated in a few columns or in interactions trees learn slowly | importance mass, partial dependence, cell-rate probe vs model score |
| 2 | Variance reduction | multi-seed bagging (`seed_list` × `train_multi_model`), repeated CV | member spread ≈ or > ensemble gain | spread of per-seed OOF AUC |
| 3 | HPO | `src/tune.py` (Optuna over `search_space.yaml`, locked folds) | baseline params look un-tuned vs the space | distance of current params from space midpoints; fold std |
| 4 | Diversity | extra families (ExtraTrees/RandomForest/MLP), `ensemble.py --method stacking` | members are highly correlated (Spearman ≈ 0.99+) | pairwise OOF rank correlation matrix |
| 5 | Post-processing | rank average, calibration, clipping | metric is rank-based AND members disagree on ordering; or probability metric with miscalibration | prob-avg vs rank-avg AUC; reliability curve |
| 6 | CV/diagnostics | error slicing (which rows/pairs are mis-ranked), fold-level forensics | you cannot name WHERE the remaining error lives | slice-level AUC, fold spread > improvement threshold |
| 7 | Data | dedup audit, original-dataset merge (Playground Series), resampling for imbalance | duplicates or synthetic-generation artifacts suspected; rare positive class hurts | duplicate scan, train/test marginal drift |

Rules for adding to the bank: an idea enters a queue only with a **falsifiable** hypothesis and a
number ("+0.001–0.003 because …"), never as "try X, might help".

## 4) Creative divergence pass (Step F)

Run when the queue is empty and §5 says stop has NOT fired. Timebox: one pass, not open-ended.

1. **Collect**: walk the §3 table and list every family not yet exhausted on THIS dataset
   (a family is exhausted when its cheapest test was DISCARD with a stated reason).
2. **Generate**: for each viable family, write 1–3 concrete candidates with hypothesis,
   expected Δ range, cost, and the first command/file to touch.
3. **Kill**: drop any candidate that (a) needs a frozen-zone change (CV, metric, test usage —
   `02_policy.md`), (b) has no measurable prediction, or (c) costs more than the remaining budget.
4. **Rank** survivors by EV/cost, promote the top ≤5 into `memory/NEXT.md` (and mirror the round
   into `doc/06` §G), execute the #1 next round.

Divergence is bounded creativity: the output is ranked queue entries, not a brainstorm document.

## 5) Stopping criteria (Step G) — all four are explicit

| Criterion | Trigger | Action |
|---|---|---|
| **Plateau** | K = **3** consecutive rounds with Δ < `improve_threshold` (any family) | stop; the queue's expected values have collapsed |
| **Budget** | wall-clock rounds/hours declared in `doc/00_problem_statement.md` (or user budget) exhausted | stop |
| **Deadline** | competition close / user deadline / user says stop | stop immediately |
| **Exhaustion** | no survivable candidate in §3 (all families tested or ruled out) AND queue empty | stop |

On stop, append the final `G.x` **retrospective**: final best run + metric, every KEEP that
contributed (in order), what was DISCARDed and why, LB result if any (`doc/06` §H), and
**reusable lessons** (which families paid here — feed the next project's §3 EV priors).

Never stop because "the pipeline works" — that is the Step-0 condition, not the finish line.

## 6) Anti-patterns

1. **OOF overfitting**: unbounded weight/stacking search on the same OOF inflates CV — keep
   member/weight searches inside `search_space.yaml:policy` and prefer new evidence
   (new seeds, new families) over re-fitting weights.
2. **LB probing**: tuning to public-LB moves past the submission budget in `doc/00` =
   test-set usage (guardrail violation family `use_test_for_tuning`).
3. **Queue without evidence**: candidates with no hypothesis/number are noise; §2 rejects them.
4. **CV churn mid-loop**: changing `doc/04` invalidates every prior Δ — the loop compares runs
   only within one locked CV (`02_policy.md` frozen zone).
5. **Silent stopping**: ending without a `G.x` retrospective makes the next session re-derive
   everything; the queue and retrospective ARE the handover.
