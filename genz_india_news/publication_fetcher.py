"""Stage 1b: fetch India/GenZ news candidates from ONLY a curated publisher
whitelist (config/publications.json), rather than an open keyword search
across all of Google News. Each publication is reached either via its own
direct RSS feed (when one is confirmed live) or via a Google News query
restricted to its domain(s) with `site:` -- Google does the crawling in
that case, this pipeline never scrapes a publisher's site directly.

Attribution is by domain (which feed/query we deliberately targeted), not
by trusting Google's <source> text -- that string is frequently a mismatched
brand name or bare domain (e.g. "Rolling Stone India" reports as
"rollingstoneindia.com", "InStyle" as "instyle.com"), which silently breaks
name-based whitelist matching.
"""

from __future__ import annotations

import calendar
import logging
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus, urlparse

import feedparser
import requests

from fetcher import GOOGLE_NEWS_RSS_URL, REQUEST_HEADERS
from text_utils import normalize_headline

logger = logging.getLogger("genz_india_news.publication_fetcher")


def _domain_of(url: str) -> str:
    """Bare host for a link, e.g. 'https://www.vogue.in/x' -> 'vogue.in'."""
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _within_lookback(entry: Any, lookback_hours: int | None) -> bool:
    """True if `entry` was published within the last `lookback_hours`.
    A direct RSS feed (unlike a Google News `when:` query) has no built-in
    recency filter -- a publisher's feed can include days- or weeks-old
    items, so this is what actually enforces the window for that path.

    RSS 2.0 feeds carry the date as <pubDate>, which feedparser exposes as
    published_parsed -- but RDF/RSS 1.0 feeds (e.g. Nature's) carry it as
    Dublin Core <dc:date>, which feedparser maps to updated_parsed instead.
    Check both rather than assuming one format. Entries with neither are
    dropped, not kept -- a strict "only the last N hours" window can't
    make an exception for an item whose age is simply unknown (confirmed:
    every normal RSS 2.0/Atom feed this pipeline uses carries one of these
    two fields; the only observed exception is a journal alert feed with
    no per-item timestamp at all, which shouldn't pass a recency filter
    by default anyway)."""
    if not lookback_hours:
        return True
    date_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not date_parsed:
        return False
    published_ts = calendar.timegm(date_parsed)  # feedparser's struct_time is UTC
    age_hours = (datetime.now(timezone.utc).timestamp() - published_ts) / 3600
    return age_hours <= lookback_hours


def fetch_publication_rss(
    pub: dict[str, Any], request_settings: dict, lookback_hours: int | None = None
) -> list[dict[str, Any]]:
    """Fetch a publication's own RSS feed directly. Returns [] (never raises)
    if it has no configured feed or the fetch fails after retries -- callers
    should rely on the site-restricted Google News query in that case."""
    rss_url = pub.get("rss")
    if not rss_url:
        return []

    max_retries = request_settings.get("max_retries", 3)
    backoff_base = request_settings.get("backoff_base_seconds", 2)
    timeout = request_settings.get("timeout_seconds", 10)

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(rss_url, headers=REQUEST_HEADERS, timeout=timeout)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            in_window = [e for e in feed.entries if _within_lookback(e, lookback_hours)]
            dropped = len(feed.entries) - len(in_window)

            articles = [
                {
                    "headline": entry.get("title", "").strip(),
                    "source": pub["name"],
                    # RDF/RSS 1.0 feeds (e.g. Nature's) carry the date as
                    # dc:date, which feedparser maps to 'updated' rather
                    # than 'published' -- fall back so this isn't blank.
                    "published": entry.get("published") or entry.get("updated", ""),
                    "link": entry.get("link", ""),
                    "fetch_method": "direct_rss",
                    "_pub_india_weight": pub.get("india_weight", 0.0),
                    "_pub_genz_weight": pub.get("genz_weight", 0.0),
                    "_pub_topic_scope": pub.get("topic_scope", "dedicated"),
                }
                for entry in in_window
            ]
            logger.info(
                "Direct RSS '%s': fetched %d articles within the %sh window (%d dropped as stale, attempt %d)",
                pub["name"], len(articles), lookback_hours, dropped, attempt,
            )
            return articles

        except Exception as exc:
            logger.warning("Direct RSS '%s' attempt %d/%d failed: %s", pub["name"], attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(backoff_base ** attempt)

    logger.warning("Direct RSS '%s' failed after %d attempts; relying on Google News fallback instead.", pub["name"], max_retries)
    return []


def _build_site_restricted_url(domains: list[str], keyword: str, locale: dict, lookback_hours: int | None) -> str:
    site_clause = " OR ".join(f"site:{d}" for d in domains)
    query_text = f"({site_clause}) {keyword}"
    if lookback_hours:
        query_text += f" when:{lookback_hours}h"
    query = quote_plus(query_text)
    return GOOGLE_NEWS_RSS_URL.format(query=query, hl=locale["hl"], gl=locale["gl"], ceid=locale["ceid"])


def fetch_site_restricted(
    domain_to_pub: dict[str, dict[str, Any]],
    keyword: str,
    locale: dict,
    request_settings: dict,
    lookback_hours: int | None,
) -> list[dict[str, Any]]:
    """Query Google News for `keyword` restricted to the given domains via
    `site:`, attributing each result back to its publication by matching the
    article link's domain rather than Google's <source> text."""
    domains = list(domain_to_pub.keys())
    if not domains:
        return []

    url = _build_site_restricted_url(domains, keyword, locale, lookback_hours)
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
                # entry.link is always a news.google.com redirect wrapper,
                # never the publisher's real URL -- the actual outbound
                # domain lives in entry.source.href instead.
                source_field = entry.get("source")
                source_href = source_field.get("href", "") if isinstance(source_field, dict) else ""
                source_domain = _domain_of(source_href)
                pub = next((p for d, p in domain_to_pub.items() if d in source_domain or source_domain in d), None)
                if pub is None:
                    continue  # off-target result Google slipped in; discard rather than mis-attribute

                if not _within_lookback(entry, lookback_hours):
                    continue  # defensive: the when: query operator usually enforces this, but not guaranteed

                articles.append(
                    {
                        "headline": entry.get("title", "").strip(),
                        "source": pub["name"],
                        "published": entry.get("published", ""),
                        "link": entry.get("link", ""),
                        "keyword": keyword,
                        "fetch_method": "google_news_site_restricted",
                        "_pub_india_weight": pub.get("india_weight", 0.0),
                        "_pub_genz_weight": pub.get("genz_weight", 0.0),
                        "_pub_topic_scope": pub.get("topic_scope", "dedicated"),
                    }
                )
            logger.info(
                "Site-restricted query (%d domains, keyword '%s'): %d attributable articles (attempt %d)",
                len(domains), keyword, len(articles), attempt,
            )
            return articles

        except Exception as exc:
            logger.warning(
                "Site-restricted query (%d domains, keyword '%s') attempt %d/%d failed: %s",
                len(domains), keyword, attempt, max_retries, exc,
            )
            if attempt < max_retries:
                time.sleep(backoff_base ** attempt)

    logger.error(
        "Site-restricted query (%d domains, keyword '%s') failed after %d attempts; skipping.",
        len(domains), keyword, max_retries,
    )
    return []


DEFAULT_MAX_CANDIDATES_PER_PUBLICATION = 40


def fetch_sector_publications(
    sector: str,
    publications: list[dict[str, Any]],
    keywords: list[str],
    locale: dict,
    request_settings: dict,
    candidate_pool_size: int,
    lookback_hours: int | None = None,
    max_candidates_per_publication: int = DEFAULT_MAX_CANDIDATES_PER_PUBLICATION,
) -> list[dict[str, Any]]:
    """Fetch every candidate for a sector from ONLY its curated publisher
    list: direct RSS where a publication has a confirmed feed, a domain-
    restricted Google News query (batched across all no-feed publications,
    one query per keyword) otherwise. Merges, dedupes by normalized
    headline, caps each publication's contribution at
    max_candidates_per_publication, then caps the total at
    candidate_pool_size.

    The per-publication cap matters: publications are fetched in list order
    and a single high-volume source (e.g. a general newspaper's full daily
    feed, easily 200 items) would otherwise fill the entire candidate_pool_size
    budget by itself before any other curated publication gets a chance to
    contribute -- confirmed happening to Indian Express in the Entertainment
    sector, silently crowding out all 12 other curated sources every run."""
    if not publications:
        logger.warning("Sector '%s' has no curated publications configured; returning no candidates.", sector)
        return []

    delay = request_settings.get("delay_between_requests_seconds", 1)
    all_articles: list[dict[str, Any]] = []

    with_rss = [p for p in publications if p.get("rss")]
    without_rss = [p for p in publications if not p.get("rss")]

    for pub in with_rss:
        all_articles.extend(fetch_publication_rss(pub, request_settings, lookback_hours))
        time.sleep(delay)

    if without_rss:
        domain_to_pub = {d: p for p in without_rss for d in p["domains"]}
        for i, keyword in enumerate(keywords):
            all_articles.extend(fetch_site_restricted(domain_to_pub, keyword, locale, request_settings, lookback_hours))
            if i < len(keywords) - 1:
                time.sleep(delay)

    deduped: dict[str, dict[str, Any]] = {}
    for article in all_articles:
        key = normalize_headline(article["headline"])
        if not key or key in deduped:
            continue
        deduped[key] = article

    per_pub_counts: dict[str, int] = {}
    balanced: list[dict[str, Any]] = []
    capped_out = 0
    for article in deduped.values():
        name = article.get("source", "")
        per_pub_counts[name] = per_pub_counts.get(name, 0) + 1
        if per_pub_counts[name] <= max_candidates_per_publication:
            balanced.append(article)
        else:
            capped_out += 1

    candidates = balanced[:candidate_pool_size]
    logger.info(
        "Sector '%s': %d raw articles -> %d deduped -> %d after per-publication cap "
        "(%d dropped, max %d/publication) -> %d after pool cap "
        "(%d publications: %d direct RSS, %d site-restricted)",
        sector, len(all_articles), len(deduped), len(balanced), capped_out,
        max_candidates_per_publication, len(candidates),
        len(publications), len(with_rss), len(without_rss),
    )
    return candidates
