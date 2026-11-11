# Bayu's FutureEval bot — design & operations

A dependency-free (Python stdlib) forecasting bot for the Metaculus
**Fall 2026 FutureEval Bot Tournament** ($50k) and the bi-weekly **$1k
MiniBench**, built alongside the official template
(`../metac-bot-template/`, cloned from
github.com/Metaculus/metac-bot-template).

## Pipeline (`bot.py`)

1. **Fetch** — `GET /api/posts/?tournaments=<id>&statuses=open` for the Fall
   tournament (33121 / `fall-futureeval-2026`) and `minibench`. As of Fall
   2026 the API requires `Authorization: Token <METACULUS_TOKEN>` even for
   reads; without a token it returns a permission error.
2. **Reason** — per question type (binary / multiple choice / numeric /
   discrete):
   - *LLM path* (OpenRouter): the prompt forces base-rate-first reasoning,
     strongest evidence for and against, explicit weighing of the community
     prediction, and a machine-parseable final line (`Probability: ZZ%`,
     `Option_X: NN%`, or percentile lines). Any LLM/parse failure drops to
     the fallback — the run never dies on one question.
   - *Fallback path* (no key needed): base-rate prior blended with the
     community prediction (binary: 0.7·CP + 0.3·prior), uniform base rates
     for multiple choice, wide midpoint-centred percentiles for numeric.
     The reasoning text says plainly that it is the fallback.
3. **Submit** (only with `--submit` + token) — `POST /api/questions/forecast/`
   with the same payload shape as the official template, then
   `POST /api/comments/create/` with the reasoning as a **private** comment
   (`included_forecast: true`). Comments are *required* for prize
   eligibility, so this step is not optional in live runs.
4. **Dedupe** — submitted question IDs are stored in `forecasted.json` and
   skipped on later runs (FutureEval tournaments don't require updates).

## Running

```bash
./run.sh --questions-file questions_sample.json --out SAMPLE_FORECASTS.md  # offline dry-run (what we tested)
./run.sh                                                                    # live dry-run (needs token for reads)
./run.sh --submit                                                           # live forecasts
python3 test_bot.py                                                         # offline unit tests
```

`run.sh` loads `.env` if present and otherwise reads the OpenRouter key from
`~/.openrouter/key` into the process environment only.

## Operational facts that shape the design

- **Short windows:** MiniBench questions are open for only ~3 hours, mostly
  in the first days of each 2-week round (rounds start every other Monday);
  Fall questions also open for limited periods. Poll often — the official
  template's GitHub Actions cadence is every 20 minutes. Recommended: a
  cron/systemd loop every 15–20 min once live.
- **Free model today:** `nvidia/nemotron-3-ultra-550b-a55b:free` on
  OpenRouter (verified 2026-11-10). Free slugs rotate and rate-limit; the
  fallback keeps the bot forecasting through outages. Once Metaculus
  sponsored credits arrive (see `../STATUS.md`), point `OPENROUTER_MODEL`
  at a stronger model.
- **Question types in FutureEval:** Binary, Numeric, Discrete, Multiple
  Choice only. Numeric/discrete forecasts are sent as a 201-point (or
  outcome-count+1) CDF built from percentiles with Metaculus
  standardization constraints (monotonic, min step, ≤0.2 mass per step) —
  unit-tested in `test_bot.py`.
- **Testing area:** tournament `bot-testing-area` has one question of each
  type — the right first live target once the token exists.

## Files

- `bot.py` — the pipeline (single file, stdlib only)
- `run.sh`, `.env.template` — secret loading
- `test_bot.py` — offline tests (all passing)
- `questions_sample.json` — 4 real questions (public sources) for dry runs
- `SAMPLE_FORECASTS.md` — output of the verified dry run
- `../metac-bot-template/` — the untouched official template clone
