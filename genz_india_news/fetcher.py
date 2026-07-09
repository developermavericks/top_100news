"""Stage 1: fetch India-specific news candidates per sector from Google News RSS."""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import quote_plus

import feedparser
import requests

from text_utils import normalize_headline

logger = logging.getLogger("genz_india_news.fetcher")

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl={hl}&gl={gl}&ceid={ceid}"


def build_rss_url(keyword: str, locale: dict) -> str:
    """Build a Google News RSS search URL for a keyword using locale settings."""
    query = quote_plus(keyword)
    return GOOGLE_NEWS_RSS_URL.format(
        query=query, hl=locale["hl"], gl=locale["gl"], ceid=locale["ceid"]
    )


def fetch_keyword_articles(
    keyword: str,
    locale: dict,
    request_settings: dict,
) -> list[dict[str, Any]]:
    """Fetch and parse Google News RSS results for a single keyword, with
    retry/backoff. Returns an empty list (never raises) on repeated failure
    so one bad keyword can't kill the whole run."""
    url = build_rss_url(keyword, locale)
    max_retries = request_settings.get("max_retries", 3)
    backoff_base = request_settings.get("backoff_base_seconds", 2)
    timeout = request_settings.get("timeout_seconds", 10)

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            articles = []
            for entry in feed.entries:
                articles.append(
                    {
                        "headline": entry.get("title", "").strip(),
                        "source": _extract_source(entry),
                        "published": entry.get("published", ""),
                        "link": entry.get("link", ""),
                        "keyword": keyword,
                    }
                )
            logger.info(
                "Fetched %d articles for keyword '%s' (attempt %d)",
                len(articles), keyword, attempt,
            )
            return articles

        except Exception as exc:
            logger.warning(
                "Attempt %d/%d failed for keyword '%s': %s",
                attempt, max_retries, keyword, exc,
            )
            if attempt < max_retries:
                time.sleep(backoff_base ** attempt)

    logger.error("All %d attempts failed for keyword '%s'; skipping.", max_retries, keyword)
    return []


def _extract_source(entry: Any) -> str:
    """Google News RSS entries carry the source in entry.source.title when present,
    otherwise fall back to parsing it out of the title's ' - Source' suffix."""
    source_field = entry.get("source")
    if isinstance(source_field, dict) and source_field.get("title"):
        return source_field["title"].strip()

    title = entry.get("title", "")
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    return "Unknown"


def fetch_sector_candidates(
    sector: str,
    keywords: list[str],
    locale: dict,
    request_settings: dict,
    candidate_pool_size: int,
) -> list[dict[str, Any]]:
    """Query all keywords for a sector, merge results, dedupe by normalized
    headline, and cap the pool at candidate_pool_size."""
    all_articles: list[dict[str, Any]] = []
    delay = request_settings.get("delay_between_requests_seconds", 1)

    for i, keyword in enumerate(keywords):
        articles = fetch_keyword_articles(keyword, locale, request_settings)
        all_articles.extend(articles)
        if i < len(keywords) - 1:
            time.sleep(delay)

    deduped: dict[str, dict[str, Any]] = {}
    for article in all_articles:
        key = normalize_headline(article["headline"])
        if not key:
            continue
        if key in deduped:
            existing_keywords = deduped[key].setdefault("keywords", [deduped[key]["keyword"]])
            if article["keyword"] not in existing_keywords:
                existing_keywords.append(article["keyword"])
        else:
            article["keywords"] = [article["keyword"]]
            deduped[key] = article

    candidates = list(deduped.values())[:candidate_pool_size]
    logger.info(
        "Sector '%s': %d raw articles -> %d deduped -> %d after pool cap",
        sector, len(all_articles), len(deduped), len(candidates),
    )
    return candidates
