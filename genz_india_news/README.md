# genz_india_news

A standalone curation pipeline that fetches India-specific news, scores it
for **India relevance** and **GenZ/Alpha relevance** using purely
rule-based signals against the Google News candidate pool (no LLM calls,
no external engagement APIs), exports a survey-ready Excel workbook, and
later ingests real survey responses to retrain its own scoring weights.

## Pipeline

```
1. FETCH   fetch_and_score.py  -> pull India news candidates per sector from Google News RSS
2. SCORE   fetch_and_score.py  -> rank by (India-relevance + GenZ/Alpha-relevance)
3. EXPORT  fetch_and_score.py  -> top N per sector into a survey-ready Excel workbook
4. INGEST  ingest_responses.py -> read back survey responses tied to headlines
5. LEARN   retrain_weights.py  -> recompute scoring weights from real response data
```

Stages 1-3 run together in `fetch_and_score.py`. Stages 4-5 are separate
scripts you run later, once survey responses exist.

No LLM/Claude API calls, and no external engagement APIs (Reddit, YouTube,
etc.) either -- every reliable free option for a real-time behavioral
signal turned out to be blocked, quota-limited, or gated behind an approval
process that a personal research script doesn't qualify for. GenZ/Alpha
relevance is therefore purely rule-based for now, combining:
1. A youth-skewing publisher whitelist match
2. A GenZ/Alpha topic keyword whitelist match (gaming, K-pop, memes, campus
   culture, AI tools, mental health, etc.)

The real behavioral signal comes later, from `retrain_weights.py` fitting
actual survey responses against these rule-based scores -- so the weights
improve over time based on what real GenZ/Alpha respondents say they're
interested in, not a proxy engagement number from a third-party platform.

## Setup

```bash
pip install -r requirements.txt
```

No API keys, no `.env` file, no app registration -- everything runs
against Google News RSS and static config, so there's nothing to set up
beyond installing dependencies.

## Frontend

`app.py` is a one-page Streamlit UI: a "Run pipeline now" button that
triggers stages 1-3 for every sector, then download buttons for the two
Excel workbooks plus a per-sector preview table. Meant for a manual,
once-a-day click rather than a scheduler.

Run locally:

```bash
streamlit run app.py
```

Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud): push
this project to a GitHub repo, connect it on share.streamlit.io, and point
the deployment at `app.py`. No secrets/API keys to configure -- the only
dependency is `requirements.txt`. Note that Streamlit Cloud's filesystem is
ephemeral (files don't persist between app restarts), so download the
Excel files right after each run rather than expecting them to still be
there days later.

## Running each stage from the CLI

### Stage 1-3: fetch, score, export

```bash
python fetch_and_score.py --once   # run once and exit
python fetch_and_score.py          # run once immediately, then daily at
                                    # config/settings.json's schedule_time
```

Outputs land in `output/`:
- `scored_headlines.xlsx` -- one sheet per sector, all scoring columns, for
  your own review and as the basis for retraining
- `survey_clean.xlsx` -- respondent-facing export: `Headline`, `Sector`, a
  blank `Relevant` column with an in-cell Yes/No dropdown (Excel data
  validation), and a blank `Remark` column for free-text comments -- no
  HeadlineID and no scores visible. (`ingest_responses.py` recomputes the
  HeadlineID deterministically from the Headline text itself, so dropping
  it from the visible sheet loses nothing.)

A flat `data/latest_scored_signals.csv` is also written (overwritten every
run) with every granular sub-signal per headline -- this is what
`ingest_responses.py` joins survey responses against.

### Stage 4: ingest survey responses

Once respondents have filled in the `Relevant` dropdown (Yes/No, and
optionally a `Remark`) in `survey_clean.xlsx`, drop that file (or a
CSV/Excel with the same `Headline, Relevant` columns) into
`data/survey_responses/`, then run:

```bash
python ingest_responses.py
```

`Relevant` (Yes/No) is converted to a binary `Response` column (Yes=1,
No=0) along the way -- rows left blank or filled in with anything other
than Yes/No are dropped with a warning rather than silently miscounted.
`HeadlineID` is recomputed from the `Headline` text (same deterministic
hash used everywhere else), then joined against
`data/latest_scored_signals.csv` and appended (every scoring sub-signal +
the binary response + any `Remark` + a freshness feature) to
`data/training_data.csv`, which accumulates across survey cycles and is
never overwritten.

### Stage 5: retrain weights

```bash
python retrain_weights.py
```

Fits a regression (`LogisticRegression`, since `Response` is binary
Yes/No) from the individual signal scores onto the real survey `Response`,
prints feature importances (which signal actually predicts relevance),
backs up `config/scoring_weights.json` to a timestamped `.bak` file, and
writes new weights derived from the fit. Safe to re-run repeatedly as more
survey data accumulates -- this is the mechanism that lets the scoring get
smarter over time, in place of a live engagement API.

## Where a survey tool plugs in

The simplest path: hand out `output/survey_clean.xlsx` directly (e.g. via
email or a shared drive) and have respondents pick Yes/No from the
in-cell dropdown next to each headline, then drop the filled-in file back
into `data/survey_responses/` as-is.

If you'd rather use a dedicated survey tool (Google Forms, Typeform, a
WhatsApp form, etc.), `ingest_responses.py` only cares about ending up with
a CSV/Excel file with `Headline` and `Relevant` (Yes/No) columns (`Remark`
optional) in `data/survey_responses/` -- map that tool's response export
back to those columns before dropping it in.

## Config files

All tunable behavior lives in `config/`, nothing hardcoded in the scoring
logic:

| File | Contents |
|---|---|
| `sectors.json` | sector name -> list of Google News search keywords |
| `settings.json` | candidate pool size, output cap, locale, request retry/backoff, schedule time, log path |
| `scoring_weights.json` | `india_weight`, `genz_weight`, GenZ sub-signal weights (`w1_source`/`w2_topic_keyword`), India sub-signal weights, freshness half-life |
| `source_lists.json` | India-publisher and GenZ/Alpha-publisher weight maps |
| `genz_topic_keywords.json` | flat GenZ/Alpha topic keyword list, reused across all sectors |

## Project layout

```
genz_india_news/
├── config/                  # all tunable JSON config (see table above)
├── data/
│   ├── survey_responses/    # drop survey exports here for ingest_responses.py
│   ├── latest_scored_signals.csv   # overwritten every fetch_and_score.py run
│   └── training_data.csv           # accumulates across survey cycles
├── output/                  # scored_headlines.xlsx, survey_clean.xlsx
├── logs/                    # pipeline.log
├── config_utils.py          # JSON config loading + logging setup
├── text_utils.py            # normalization, hashing, source lookup
├── fetcher.py                # Stage 1: Google News RSS fetch + dedupe
├── india_scorer.py           # India-relevance sub-score
├── genz_scorer.py             # GenZ/Alpha source + topic keyword sub-signals
├── exporter.py                 # Stage 3: Excel + raw signal CSV export
├── fetch_and_score.py           # Stages 1-3 orchestrator + CLI
├── ingest_responses.py          # Stage 4
├── retrain_weights.py            # Stage 5
├── app.py                         # Streamlit frontend (run button + downloads)
├── requirements.txt
└── README.md
```

## Notes

- Google News redirect links are kept as-is (not resolved to final publisher
  URLs).
- India-entity detection is a simple keyword list (states, cities, major
  companies/institutions), not full NER -- adequate for v1.
- Everything is file-based (Excel/CSV/JSON) -- no database.
- No external engagement API (Reddit, YouTube, etc.) is used. Reddit's
  OAuth app approval is now gated behind a moderation use-case review, and
  its public JSON endpoints are blocked outright for non-browser requests
  (confirmed live -- 403 on `.json`/API paths, and its RSS feeds don't carry
  score/comment counts anyway). Rather than add a different third-party
  dependency, the plan is to let `retrain_weights.py` learn real interest
  signal directly from survey responses instead.
