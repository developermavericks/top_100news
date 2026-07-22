# genz_india_news

A standalone curation pipeline that fetches India-specific news, scores it
for **India relevance** and **GenZ/Alpha relevance**, exports a
survey-ready Excel workbook, and later ingests real survey responses to
retrain its own scoring weights. Scoring is rule-based by default (no
external engagement APIs); two **optional** LLM touchpoints via
[Groq](https://console.groq.com) can be layered on top -- see
[Optional: LLM scoring via Groq](#optional-llm-scoring-via-groq) below.

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

No external engagement APIs (Reddit, YouTube, etc.) are used -- every
reliable free option for a real-time behavioral signal turned out to be
blocked, quota-limited, or gated behind an approval process that a
personal research script doesn't qualify for. GenZ/Alpha relevance is
rule-based by default, combining:
1. A youth-skewing publisher whitelist match
2. A GenZ/Alpha topic keyword whitelist match (gaming, K-pop, memes, campus
   culture, AI tools, mental health, etc.)
3. *(optional, Groq)* an LLM's judgment of cultural currency and framing --
   see below

The real behavioral signal comes later, from `retrain_weights.py` fitting
actual survey responses against these rule-based scores -- so the weights
improve over time based on what real GenZ/Alpha respondents say they're
interested in, not a proxy engagement number from a third-party platform.

## Setup

```bash
pip install -r requirements.txt
```

No API keys, no `.env` file, no app registration required -- the pipeline
runs against Google News RSS and static config out of the box. The two
LLM touchpoints are opt-in on top of that; see the next section if you
want them.

## Optional: LLM scoring via Groq

Two touchpoints layer an LLM on top of the existing rule-based scoring,
using [Groq](https://console.groq.com) (chosen because it's a plain
outbound HTTPS API call -- no local model server, which Streamlit
Community Cloud can't run anyway):

1. **Universal relevance filter** (`llm_relevance_filter.py`) -- asks the
   model whether each surviving headline is genuinely about its sector.
   The existing keyword-based gate only checks "general" (multi-topic)
   curated sources; this checks every candidate, including "dedicated"
   sources the keyword gate skips entirely.
2. **LLM GenZ-relevance signal** (`genz_scorer.get_llm_genz_relevance_score`)
   -- rates 0-100 how likely a headline is to interest Indian Gen Z/Alpha
   readers, folded into `genz_alpha_score` as a third signal alongside the
   existing source-whitelist and topic-keyword matches. Only runs on the
   top `llm_genz_scoring_top_n` headlines per sector (by the existing
   rule-based ranking), not the full candidate pool, to control cost.

Both are **off by default** and fail open: if disabled, or if
`GROQ_API_KEY` isn't set, or if a Groq call fails after retries, the
pipeline runs exactly as it did before this feature existed -- rule-based
scoring only, nothing crashes, nothing silently empties a sector.

### Getting a key

Free at [console.groq.com/keys](https://console.groq.com/keys).

### Local development

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
# then edit .env: GROQ_API_KEY=gsk_...
```

`.env` is gitignored -- never commit your real key.

### Streamlit Community Cloud deployment

`.env` files aren't used on Community Cloud. Instead, open your app's
**Settings → Secrets** in the Streamlit Cloud dashboard and add:

```toml
GROQ_API_KEY = "gsk_..."
```

The same code path (`llm_client.get_groq_client`) checks `st.secrets`
first and falls back to the environment, so no code changes are needed
between local and deployed.

### Turning the features on

Both touchpoints need a real API key *and* their own flag in
`config/settings.json` -- having a key alone doesn't turn anything on:

```json
"use_llm_relevance_filter": false,
"use_llm_genz_scoring": false,
```

Flip either to `true` to enable it. Other tunables in the same file:
`groq_model` (default `llama-3.1-8b-instant`), `groq_request_delay_seconds`
(delay between Groq calls, respects rate limits), `groq_max_retries`, and
`llm_genz_scoring_top_n` (how many top-ranked headlines per sector get the
GenZ-scoring call).

`app.py`'s UI shows a one-line status ("LLM scoring: ON/OFF") reflecting
whether a usable key + the relevant flag are both present for the current
run -- there's no separate on/off control in the UI itself, only in
settings.json.

### New Excel/CSV columns

When either touchpoint runs, `scored_headlines.xlsx` and
`data/latest_scored_signals.csv` gain two columns:
- **LLMGenZScore** -- the raw 0-100 LLM cultural-relevance rating (blank
  for any headline the GenZ-scoring touchpoint didn't reach, e.g. outside
  the top-N shortlist, or if the feature's off).
- **LLMRelevancePassed** -- `True` for any headline the relevance filter
  actually processed and kept (blank if the filter never ran on that
  sector/row).

### Embeddings + the diagnostic retraining fit

Independent of the two Groq touchpoints (this part is entirely local, no
API cost): every export run also computes a
[sentence-transformers](https://www.sbert.net) embedding
(`all-MiniLM-L6-v2`) for each headline and appends it to
`data/embeddings_cache.jsonl`, keyed by `HeadlineID`. `ingest_responses.py`
joins these into `training_data.csv` (as `embed_0`..`embed_(dim-1)`
columns) alongside `llm_genz_score`, and `retrain_weights.py` runs a
**separate, diagnostic-only** model that also sees these features and
prints its feature importances -- this diagnostic fit is *not* written
back to `scoring_weights.json` (there's no clean way to fold a ~384-dim
embedding into the existing weighted-sum scoring formula, and at realistic
survey volumes a model with hundreds of features risks serious overfitting
anyway). It exists purely to show whether embeddings or the LLM signal
actually predict real survey responses before investing further in either.

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
| `sectors.json` | sector name -> list of Google News search keywords (`_`-prefixed keys are disabled) |
| `publications.json` | curated publisher whitelist per sector (domains, direct RSS, weights, `topic_scope`) |
| `settings.json` | candidate pool size, output cap, locale, request retry/backoff, news lookback window, thin-pool fallback, Groq model/toggles/retry settings, schedule time, log path |
| `scoring_weights.json` | `india_weight`, `genz_weight`, GenZ sub-signal weights (`w1_source`/`w2_topic_keyword`/`w3_llm_genz`), India sub-signal weights, freshness half-life |
| `source_lists.json` | India-publisher and GenZ/Alpha-publisher weight maps (fuzzy-matched; only used by non-curated sectors) |
| `genz_topic_keywords.json` | flat GenZ/Alpha topic keyword list, reused across all sectors |

## Project layout

```
genz_india_news/
├── config/                  # all tunable JSON config (see table above)
├── data/
│   ├── survey_responses/    # drop survey exports here for ingest_responses.py
│   ├── latest_scored_signals.csv   # overwritten every fetch_and_score.py run
│   ├── embeddings_cache.jsonl       # local sentence-embedding cache, keyed by HeadlineID
│   └── training_data.csv           # accumulates across survey cycles
├── output/                  # scored_headlines.xlsx, survey_clean.xlsx
├── logs/                    # pipeline.log
├── config_utils.py          # JSON config loading + logging setup
├── text_utils.py            # normalization, hashing, source lookup
├── fetcher.py                # Stage 1: open Google News RSS fetch + dedupe (non-curated sectors)
├── publication_fetcher.py     # Stage 1b: curated-publisher fetch (direct RSS + site:-restricted)
├── india_scorer.py           # India-relevance sub-score
├── genz_scorer.py             # GenZ/Alpha source + topic keyword + optional LLM sub-signals
├── llm_client.py               # shared Groq client resolution (st.secrets / env) -- optional
├── llm_relevance_filter.py      # optional universal LLM relevance gate (Groq)
├── exporter.py                 # Stage 3: Excel + raw signal CSV + embeddings cache export
├── fetch_and_score.py           # Stages 1-3 orchestrator + CLI
├── ingest_responses.py          # Stage 4
├── retrain_weights.py            # Stage 5
├── app.py                         # Streamlit frontend (run button + downloads)
├── .env.example                    # GROQ_API_KEY placeholder for local dev
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
