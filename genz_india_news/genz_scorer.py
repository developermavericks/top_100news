"""GenZ/Alpha-relevance sub-score: source whitelist + topic keyword whitelist
signals. No external engagement API -- purely rule-based against the
Google News candidate pool, refined later by retrain_weights.py once real
survey responses come in."""

from __future__ import annotations

from typing import Any

from text_utils import count_keyword_matches, lookup_source_weight

# Cap on topic keyword matches before the signal saturates at 100 -- a
# headline hitting 3+ GenZ topic keywords is unambiguously youth-relevant.
TOPIC_MATCH_SATURATION = 3


def compute_genz_source_and_topic_scores(
    article: dict[str, Any],
    genz_alpha_sources: dict[str, float],
    genz_topic_keywords: list[str],
) -> dict[str, float]:
    """Compute the source-whitelist and topic-keyword-whitelist signals (0-100 each)."""
    source_score = lookup_source_weight(article.get("source", ""), genz_alpha_sources) * 100

    headline = article.get("headline", "")
    matched_topics = count_keyword_matches(headline, genz_topic_keywords)
    topic_keyword_score = min(matched_topics / TOPIC_MATCH_SATURATION, 1.0) * 100

    return {
        "genz_source_score": source_score,
        "genz_topic_keyword_score": topic_keyword_score,
    }


def combine_genz_score(
    genz_source_score: float,
    genz_topic_keyword_score: float,
    subsignal_weights: dict[str, float],
) -> float:
    """Combine the GenZ/Alpha sub-signals into genz_alpha_score (0-100)."""
    score = (
        genz_source_score * subsignal_weights.get("w1_source", 0.5)
        + genz_topic_keyword_score * subsignal_weights.get("w2_topic_keyword", 0.5)
    )
    return max(0.0, min(100.0, score))
