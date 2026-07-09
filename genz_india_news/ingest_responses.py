"""Stage 4: ingest survey responses and join them back against the scoring
signals from the most recent fetch_and_score.py run, accumulating into
data/training_data.csv for later retraining.

Expects response files in data/survey_responses/ (CSV or Excel) with columns:
    Headline, Relevant, Remark      (Relevant: "Yes"/"No", filled in via the
                                      in-cell dropdown in output/survey_clean.xlsx;
                                      Remark: optional free-text comment)

There's no HeadlineID column in the respondent-facing sheet -- it's
recomputed deterministically from the Headline text itself (the same hash
fetch_and_score.py used to generate it originally), which is what this
joins against data/latest_scored_signals.csv by.

Relevant is converted to a binary Response column (Yes=1, No=0) before
being appended to training_data.csv, which is what retrain_weights.py
expects as its target column.

Run manually once survey data exists:
    python ingest_responses.py
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config_utils import load_scoring_weights, load_settings, setup_logging
from text_utils import headline_id

# pandas 2.2.x's "ChainedAssignmentError" FutureWarning fires even on freshly
# .copy()'d DataFrames doing plain, correct column assignment (verified: the
# same warning fires on a bare in-memory DataFrame with no file I/O involved
# at all) -- a known false positive in that warning's heuristic, not a real
# risk in this code. Silenced narrowly by message rather than broadly by
# category so any other FutureWarning still surfaces normally.
warnings.filterwarnings("ignore", message=".*ChainedAssignmentError.*", category=FutureWarning)

logger = logging.getLogger("genz_india_news.ingest_responses")

RESPONSES_DIR = Path("data/survey_responses")
RAW_SIGNALS_PATH = Path("data/latest_scored_signals.csv")
TRAINING_DATA_PATH = Path("data/training_data.csv")

REQUIRED_RESPONSE_COLUMNS = {"Headline", "Relevant"}

_YES_NO_TO_BINARY = {"yes": 1, "no": 0}


def _relevant_to_response(value: Any) -> float:
    """Map a Yes/No dropdown value to 1/0. Blank/unanswered or unrecognized
    values map to NaN so they can be dropped rather than silently counted."""
    text = str(value).strip().lower()
    return _YES_NO_TO_BINARY.get(text, float("nan"))


def load_response_files(responses_dir: Path) -> pd.DataFrame:
    """Load and concatenate every CSV/Excel file in responses_dir that has
    the required Headline/Relevant columns, recomputing HeadlineID from the
    Headline text and converting Relevant (Yes/No) into a binary Response
    column. Skips and warns on bad files, and drops rows the respondent left
    blank or filled in with something other than Yes/No."""
    frames = []
    files = sorted(responses_dir.glob("*.csv")) + sorted(responses_dir.glob("*.xlsx"))

    if not files:
        logger.warning("No response files found in %s", responses_dir)
        return pd.DataFrame(columns=["HeadlineID", "Response", "Remark"])

    for path in files:
        try:
            df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
        except Exception as exc:
            logger.warning("Could not read response file %s: %s", path, exc)
            continue

        if not REQUIRED_RESPONSE_COLUMNS.issubset(df.columns):
            logger.warning(
                "Skipping %s: missing required columns %s (has %s)",
                path, REQUIRED_RESPONSE_COLUMNS, list(df.columns),
            )
            continue

        remark_column = df["Remark"].fillna("").astype(str) if "Remark" in df.columns else ""

        df = df[["Headline", "Relevant"]].copy()
        df["HeadlineID"] = df["Headline"].astype(str).apply(headline_id)
        df["Response"] = df["Relevant"].apply(_relevant_to_response)
        df["Remark"] = remark_column

        unanswered = df["Response"].isna().sum()
        if unanswered > 0:
            logger.warning(
                "%d of %d rows in %s were blank/unrecognized (not Yes/No); dropping them.",
                unanswered, len(df), path.name,
            )
        df = df.dropna(subset=["Response"]).drop(columns=["Relevant", "Headline"])
        df["Response"] = df["Response"].astype(int)
        df["SourceFile"] = path.name
        frames.append(df)
        logger.info("Loaded %d valid responses from %s", len(df), path.name)

    if not frames:
        return pd.DataFrame(columns=["HeadlineID", "Response", "Remark"])

    return pd.concat(frames, ignore_index=True)


def compute_freshness_score(published: str, half_life_hours: float) -> float:
    """Exponential decay freshness score (0-100) from an RSS 'Published' date
    string. Returns 0.0 if the date can't be parsed (e.g. missing/malformed)."""
    if not published:
        return 0.0
    try:
        published_dt = parsedate_to_datetime(published)
        if published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return 0.0

    age_hours = (datetime.now(timezone.utc) - published_dt).total_seconds() / 3600
    age_hours = max(age_hours, 0.0)
    return 100 * (0.5 ** (age_hours / half_life_hours))


def ingest_responses() -> None:
    settings = load_settings()
    setup_logging(settings.get("log_file"))

    if not RAW_SIGNALS_PATH.exists():
        logger.error(
            "No raw signals cache found at %s. Run fetch_and_score.py first "
            "so there's a scored run to join responses against.",
            RAW_SIGNALS_PATH,
        )
        return

    responses_df = load_response_files(RESPONSES_DIR)
    if responses_df.empty:
        logger.warning("No valid survey responses loaded; nothing to ingest.")
        return

    raw_signals_df = pd.read_csv(RAW_SIGNALS_PATH, dtype={"HeadlineID": str})

    merged = responses_df.merge(raw_signals_df, on="HeadlineID", how="inner")
    unmatched = len(responses_df) - len(merged)
    if unmatched > 0:
        logger.warning(
            "%d of %d responses had no matching HeadlineID in %s (stale run or typo).",
            unmatched, len(responses_df), RAW_SIGNALS_PATH,
        )

    if merged.empty:
        logger.warning("No responses matched the current scored run; nothing to append.")
        return

    weights = load_scoring_weights()
    half_life_hours = weights.get("freshness_half_life_hours", 48)
    merged["freshness_score"] = merged["Published"].apply(
        lambda p: compute_freshness_score(p, half_life_hours)
    )
    merged["IngestedAt"] = datetime.now(timezone.utc).isoformat()

    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not TRAINING_DATA_PATH.exists()
    merged.to_csv(TRAINING_DATA_PATH, mode="a", header=write_header, index=False)

    logger.info(
        "Appended %d rows to %s (write_header=%s)",
        len(merged), TRAINING_DATA_PATH, write_header,
    )


def main() -> None:
    ingest_responses()


if __name__ == "__main__":
    main()
