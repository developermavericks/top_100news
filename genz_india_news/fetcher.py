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

# requests' own default ("python-requests/x.x.x") is a well-known signature
# of an unmodified automated script that some publisher WAFs block on sight
# (confirmed: rollingstoneindia.com/feed/ returns 403 with the default UA
# and 200 with this one, same IP, same request otherwise). A normal browser
# UA is what every real RSS reader sends -- this doesn't bypass a paywall,
# login, or robots.txt disallow rule, it just stops announcing "unmodified
# script" for no reason.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def build_rss_url(keyword: str, locale: dict, lookback_hours: int | None = None) -> str:
    """Build a Google News RSS search URL for a keyword using locale settings.
    If lookback_hours is set, appends Google's `when:` search operator so
    results are restricted server-side to articles published within that
    window, rather than filtering client-side after the fact."""
    query_text = f"{keyword} when:{lookback_hours}h" if lookback_hours else keyword
    query = quote_plus(query_text)
    return GOOGLE_NEWS_RSS_URL.format(
        query=query, hl=locale["hl"], gl=locale["gl"], ceid=locale["ceid"]
    )


def fetch_keyword_articles(
    keyword: str,
    locale: dict,
    request_settings: dict,
    lookback_hours: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch and parse Google News RSS results for a single keyword, with
    retry/backoff. Returns an empty list (never raises) on repeated failure
    so one bad keyword can't kill the whole run."""
    url = build_rss_url(keyword, locale, lookback_hours)
    max_retries = request_settings.get("max_retries", 3)
    backoff_base = request_settings.get("backoff_base_seconds", 2)
    timeout = request_settings.get("timeout_seconds", 10)

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
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
    lookback_hours: int | None = None,
) -> list[dict[str, Any]]:
    """Query all keywords for a sector, merge results, dedupe by normalized
    headline, and cap the pool at candidate_pool_size."""
    all_articles: list[dict[str, Any]] = []
    delay = request_settings.get("delay_between_requests_seconds", 1)

    for i, keyword in enumerate(keywords):
        articles = fetch_keyword_articles(keyword, locale, request_settings, lookback_hours)
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
