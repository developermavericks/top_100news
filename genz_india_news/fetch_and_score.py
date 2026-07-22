"""Stages 1-3 orchestrator: fetch India news candidates per sector, score
them for India + GenZ/Alpha relevance (no LLM calls, no external engagement
APIs -- purely rule-based against the Google News candidate pool), and
export a survey-ready Excel workbook. Real behavioral signal comes later,
from retrain_weights.py refitting these weights against actual survey
responses once they exist.

Run modes:
    python fetch_and_score.py --once   # run once and exit
    python fetch_and_score.py          # run once immediately, then daily on a schedule
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import schedule

from config_utils import (
    load_genz_topic_keywords,
    load_publications,
    load_scoring_weights,
    load_sectors,
    load_settings,
    load_source_lists,
    setup_logging,
)
from exporter import export_full_workbook, export_raw_signals_cache, export_survey_clean
from fetcher import fetch_sector_candidates
from genz_scorer import combine_genz_score, compute_genz_source_and_topic_scores
from india_scorer import compute_india_score
from publication_fetcher import fetch_sector_publications
from text_utils import count_keyword_matches, headline_id, normalize_headline

logger = logging.getLogger("genz_india_news.fetch_and_score")


def score_sector(
    sector: str,
    candidates: list[dict[str, Any]],
    sector_keywords: list[str],
    source_lists: dict[str, dict[str, float]],
    genz_topic_keywords: list[str],
    scoring_weights: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach India + GenZ/Alpha sub-scores and final_score to every candidate
    in a sector's pool."""
    india_sources = source_lists.get("india_sources", {})
    genz_alpha_sources = source_lists.get("genz_alpha_sources", {})
    india_subsignal_weights = scoring_weights.get("india_subsignal_weights", {})
    genz_subsignal_weights = scoring_weights.get("genz_subsignal_weights", {})
    india_weight = scoring_weights.get("india_weight", 0.4)
    genz_weight = scoring_weights.get("genz_weight", 0.6)

    for article in candidates:
        article["headline_id"] = headline_id(article["headline"])

        india_result = compute_india_score(
            article, sector_keywords, india_sources, india_subsignal_weights
        )
        genz_result = compute_genz_source_and_topic_scores(
            article, genz_alpha_sources, genz_topic_keywords
        )

        # Candidates fetched via the curated publisher whitelist
        # (publication_fetcher.py) already know exactly which publication
        # they came from -- by domain, not by fuzzy-matching Google's often
        # mismatched <source> text -- so their source sub-score is set
        # directly from config/publications.json rather than looked up.
        if "_pub_india_weight" in article:
            india_result["india_source_score"] = article["_pub_india_weight"] * 100
            india_result["india_score"] = max(0.0, min(100.0, (
                india_result["india_source_score"] * india_subsignal_weights.get("source_weight", 0.4)
                + india_result["india_keyword_score"] * india_subsignal_weights.get("keyword_match_weight", 0.35)
                + india_result["india_entity_score"] * india_subsignal_weights.get("entity_weight", 0.25)
            )))
        if "_pub_genz_weight" in article:
            genz_result["genz_source_score"] = article["_pub_genz_weight"] * 100

        article.update(india_result)
        article.update(genz_result)

        article["genz_alpha_score"] = combine_genz_score(
            article["genz_source_score"],
            article["genz_topic_keyword_score"],
            genz_subsignal_weights,
        )
        article["final_score"] = (
            article["india_score"] * india_weight
            + article["genz_alpha_score"] * genz_weight
        )

    candidates.sort(key=lambda a: a["final_score"], reverse=True)
    logger.info(
        "Sector '%s': scored %d candidates, top final_score=%.2f",
        sector, len(candidates), candidates[0]["final_score"] if candidates else 0.0,
    )
    return candidates


def _gate_general_sources(
    candidates: list[dict[str, Any]], sector_keywords: list[str]
) -> list[dict[str, Any]]:
    """Drop articles from a "general" (multi-topic) curated source -- a wire
    service or general newspaper included for its sector-relevant slice, not
    because everything it publishes belongs here -- unless the headline
    actually matches one of this sector's own keywords. Without this, a
    trusted general source's off-topic output (sports, unrelated politics,
    other wire stories) still scores well on source-trust alone and leaks
    into a sector it has nothing to do with (confirmed: Reuters' UEFA/
    Coca-Cola stories ranking near the top of the AI sector). Sector-
    dedicated sources (the default, unset) are exempt -- their whole output
    already belongs to the sector even when a given headline doesn't happen
    to contain one of the sector's narrow search keywords verbatim.

    Applied before the thin-pool-fallback check in run_pipeline, not inside
    score_sector, so that check sees the pool size the gate actually leaves
    behind rather than the pre-gate count."""
    kept = []
    dropped = 0
    for article in candidates:
        if article.get("_pub_topic_scope") == "general":
            if count_keyword_matches(article.get("headline", ""), sector_keywords) == 0:
                dropped += 1
                continue
        kept.append(article)
    if dropped:
        logger.info("Gated out %d off-topic article(s) from general/multi-topic sources.", dropped)
    return kept


def _merge_dedupe(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge `secondary` into `primary`, keeping primary's copy of any
    headline both share. Used to top up a thin curated-publisher pool with
    the broad Google News search without duplicating anything the curated
    fetch already found."""
    seen = {normalize_headline(a["headline"]) for a in primary if a.get("headline")}
    merged = list(primary)
    for article in secondary:
        key = normalize_headline(article.get("headline", ""))
        if key and key not in seen:
            seen.add(key)
            merged.append(article)
    return merged


def run_pipeline() -> dict[str, list[dict[str, Any]]]:
    """Run stages 1-3 for every configured sector and export the results.
    Returns the per-sector scored results (also useful for callers like a
    frontend that want counts/previews without re-reading the export files)."""
    settings = load_settings()
    sectors = load_sectors()
    scoring_weights = load_scoring_weights()
    source_lists = load_source_lists()
    genz_topic_keywords = load_genz_topic_keywords()
    publications = load_publications()

    candidate_pool_size = settings.get("candidate_pool_size", 200)
    max_candidates_per_publication = settings.get("max_candidates_per_publication", 40)
    max_headlines_per_sector = settings.get("max_headlines_per_sector", 100)
    locale = settings.get("locale", {"hl": "en", "gl": "IN", "ceid": "IN:en"})
    request_settings = settings.get("request", {})
    output_dir = settings.get("output_dir", "output")
    news_lookback_hours = settings.get("news_lookback_hours")
    thin_pool_cfg = settings.get("thin_pool_fallback", {})
    thin_pool_sectors = set(thin_pool_cfg.get("sectors", []))
    thin_pool_min_candidates = thin_pool_cfg.get("min_candidates", 30)

    sector_results: dict[str, list[dict[str, Any]]] = {}

    for sector, keywords in sectors.items():
        logger.info("=== Fetching sector '%s' ===", sector)
        sector_publications = publications.get(sector, [])
        if sector_publications:
            # Curated-publisher sourcing: ONLY these domains, direct RSS
            # first, a site:-restricted Google News query as fallback.
            candidates = fetch_sector_publications(
                sector, sector_publications, keywords, locale, request_settings,
                candidate_pool_size, news_lookback_hours, max_candidates_per_publication,
            )
            candidates = _gate_general_sources(candidates, keywords)
            # A handful of sectors are dominated by sources that rarely
            # produce indexed news (academic journals, literature indexes)
            # and can come up thin some days -- for those specifically,
            # top up with the broad, unrestricted Google News search rather
            # than exporting a near-empty sector. Sectors not listed here
            # stay strictly limited to their curated websites, thin or not.
            if sector in thin_pool_sectors and len(candidates) < thin_pool_min_candidates:
                logger.info(
                    "Sector '%s': curated pool is thin (%d < %d) -- topping up with the broad Google News search.",
                    sector, len(candidates), thin_pool_min_candidates,
                )
                broad_candidates = fetch_sector_candidates(
                    sector, keywords, locale, request_settings, candidate_pool_size, news_lookback_hours
                )
                candidates = _merge_dedupe(candidates, broad_candidates)
        else:
            # No curated publisher list configured for this sector yet --
            # fall back to the open keyword search across all of Google News.
            candidates = fetch_sector_candidates(
                sector, keywords, locale, request_settings, candidate_pool_size, news_lookback_hours
            )

        if not candidates:
            logger.warning("Sector '%s' returned zero candidates; skipping.", sector)
            sector_results[sector] = []
            continue

        scored = score_sector(
            sector, candidates, keywords, source_lists, genz_topic_keywords, scoring_weights,
        )
        sector_results[sector] = scored[:max_headlines_per_sector]

    total_exported = sum(len(v) for v in sector_results.values())
    logger.info(
        "Pipeline run complete: %d sectors, %d total headlines exported",
        len(sector_results), total_exported,
    )

    if total_exported == 0:
        logger.warning("No headlines were scored across any sector; skipping export.")
        return sector_results

    export_full_workbook(sector_results, output_dir)
    export_survey_clean(sector_results, output_dir)
    export_raw_signals_cache(sector_results, "data")

    return sector_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and score India GenZ/Alpha news candidates.")
    parser.add_argument("--once", action="store_true", help="Run once and exit (no daily schedule).")
    args = parser.parse_args()

    settings = load_settings()
    setup_logging(settings.get("log_file"))

    run_pipeline()

    if args.once:
        return

    schedule_time = settings.get("schedule_time", "07:00")
    schedule.every().day.at(schedule_time).do(run_pipeline)
    logger.info("Scheduled daily run at %s. Press Ctrl+C to stop.", schedule_time)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
