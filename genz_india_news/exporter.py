"""Stage 3: export scored candidates to survey-ready Excel workbooks."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

logger = logging.getLogger("genz_india_news.exporter")

# Excel sheet names can't contain \ / ? * [ ] : and are capped at 31 chars.
# Sector names like "Entertainment (Film/TV/Music)" need the slashes stripped
# before they're usable as a sheet name.
_INVALID_SHEET_CHARS = re.compile(r"[\\/?*\[\]:]")


def _safe_sheet_name(sector: str) -> str:
    return _INVALID_SHEET_CHARS.sub("-", sector)[:31]

FULL_COLUMNS = [
    "HeadlineID", "Rank", "Headline", "Source", "Published", "Link",
    "IndiaScore", "GenZAlphaScore", "FinalScore", "Sector",
]

SURVEY_CLEAN_COLUMNS = ["Headline", "Sector", "Relevant", "Remark"]
_RELEVANT_COLUMN_LETTER = get_column_letter(SURVEY_CLEAN_COLUMNS.index("Relevant") + 1)

# All granular sub-signals, not just the rolled-up scores shown in the
# survey workbook. ingest_responses.py needs these to build training_data.csv
# with per-signal features (source score, topic keyword score, etc.), so this
# cache is overwritten every run alongside the survey-facing exports.
RAW_SIGNAL_COLUMNS = [
    "HeadlineID", "Sector", "Headline", "Source", "Published", "Link",
    "india_source_score", "india_keyword_score", "india_entity_score", "india_score",
    "genz_source_score", "genz_topic_keyword_score",
    "genz_alpha_score", "final_score",
]


def _sector_to_dataframe(sector: str, ranked_articles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rank, article in enumerate(ranked_articles, start=1):
        rows.append(
            {
                "HeadlineID": article["headline_id"],
                "Rank": rank,
                "Headline": article["headline"],
                "Source": article.get("source", ""),
                "Published": article.get("published", ""),
                "Link": article.get("link", ""),
                "IndiaScore": round(article["india_score"], 2),
                "GenZAlphaScore": round(article["genz_alpha_score"], 2),
                "FinalScore": round(article["final_score"], 2),
                "Sector": sector,
            }
        )
    return pd.DataFrame(rows, columns=FULL_COLUMNS)


def export_full_workbook(
    sector_results: dict[str, list[dict[str, Any]]],
    output_dir: str,
    filename: str = "scored_headlines.xlsx",
) -> Path:
    """Write one sheet per sector with all scoring columns, for internal
    review and as the source data for later retraining."""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sector, ranked_articles in sector_results.items():
            df = _sector_to_dataframe(sector, ranked_articles)
            sheet_name = _safe_sheet_name(sector)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info("Wrote %d rows to sheet '%s' in %s", len(df), sheet_name, output_path.name)

    return output_path


def export_survey_clean(
    sector_results: dict[str, list[dict[str, Any]]],
    output_dir: str,
    filename: str = "survey_clean.xlsx",
) -> Path:
    """Write the respondent-facing export: Headline, Sector, a blank
    "Relevant" column with an in-cell Yes/No dropdown, and a blank "Remark"
    column for free-text comments. No HeadlineID and no scores visible --
    ingest_responses.py recomputes the deterministic HeadlineID from the
    Headline text itself, so dropping the column here loses nothing. One
    sheet per sector, matching the full workbook."""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sector, ranked_articles in sector_results.items():
            rows = [
                {
                    "Headline": article["headline"],
                    "Sector": sector,
                    "Relevant": "",
                    "Remark": "",
                }
                for article in ranked_articles
            ]
            df = pd.DataFrame(rows, columns=SURVEY_CLEAN_COLUMNS)
            sheet_name = _safe_sheet_name(sector)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            if len(df) > 0:
                worksheet = writer.sheets[sheet_name]
                dropdown = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
                dropdown.error = "Please select Yes or No from the dropdown."
                dropdown.prompt = "Is this headline relevant/interesting to you?"
                cell_range = f"{_RELEVANT_COLUMN_LETTER}2:{_RELEVANT_COLUMN_LETTER}{len(df) + 1}"
                dropdown.add(cell_range)
                worksheet.add_data_validation(dropdown)

    logger.info("Wrote survey-clean export to %s", output_path)
    return output_path


def export_raw_signals_cache(
    sector_results: dict[str, list[dict[str, Any]]],
    data_dir: str,
    filename: str = "latest_scored_signals.csv",
) -> Path:
    """Overwrite a flat CSV of every granular scoring signal for the most
    recent run, keyed by HeadlineID. Consumed by ingest_responses.py to build
    training_data.csv -- separate from the accumulating training data itself."""
    output_path = Path(data_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for sector, ranked_articles in sector_results.items():
        for article in ranked_articles:
            rows.append(
                {
                    "HeadlineID": article["headline_id"],
                    "Sector": sector,
                    "Headline": article["headline"],
                    "Source": article.get("source", ""),
                    "Published": article.get("published", ""),
                    "Link": article.get("link", ""),
                    "india_source_score": article["india_source_score"],
                    "india_keyword_score": article["india_keyword_score"],
                    "india_entity_score": article["india_entity_score"],
                    "india_score": article["india_score"],
                    "genz_source_score": article["genz_source_score"],
                    "genz_topic_keyword_score": article["genz_topic_keyword_score"],
                    "genz_alpha_score": article["genz_alpha_score"],
                    "final_score": article["final_score"],
                }
            )

    df = pd.DataFrame(rows, columns=RAW_SIGNAL_COLUMNS)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d rows of raw signal data to %s", len(df), output_path)
    return output_path
