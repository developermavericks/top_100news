"""GenZ/Alpha-relevance sub-score: source whitelist + topic keyword whitelist
signals, plus an optional third signal -- an LLM's judgment of cultural
currency and framing, via Groq (see llm_client.py) -- that the first two,
purely rule-based signals can't capture. The LLM signal is opt-in
(config/settings.json's use_llm_genz_scoring) and fails open: unavailable
or disabled, callers fall back to the original two-signal blend."""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_client import get_groq_client, get_groq_model
from text_utils import count_keyword_matches, lookup_source_weight

logger = logging.getLogger("genz_india_news.genz_scorer")

# Cap on topic keyword matches before the signal saturates at 100 -- a
# headline hitting 3+ GenZ topic keywords is unambiguously youth-relevant.
TOPIC_MATCH_SATURATION = 3

_LLM_GENZ_PROMPT_TEMPLATE = (
    "Headline: {headline}\n"
    "Rate 0-100 how likely this headline is to interest Indian Gen Z "
    "(born ~1997-2012) and Gen Alpha (born ~2010+) readers, considering "
    "topical relevance, cultural currency, and framing. "
    "Answer with only a number from 0 to 100."
)


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


def get_llm_genz_relevance_score(headline: str, settings: dict[str, Any]) -> float | None:
    """Ask Groq to rate 0-100 how likely `headline` is to interest Indian
    Gen Z / Gen Alpha readers -- topical relevance, cultural currency, and
    framing, not just keyword presence. Returns None (never 0) on any API
    failure or if the feature is disabled/unconfigured -- 0 would wrongly
    assert "not relevant" rather than "couldn't ask"; callers must fall
    back to the existing rule-based signals when this is None (see
    combine_genz_score)."""
    client = get_groq_client(settings, "use_llm_genz_scoring")
    if client is None:
        return None

    model = get_groq_model(settings)
    max_retries = settings.get("groq_max_retries", 3)
    prompt = _LLM_GENZ_PROMPT_TEMPLATE.format(headline=headline)
    backoff = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            text = (response.choices[0].message.content or "").strip()
            digits = "".join(c for c in text if c.isdigit() or c == ".")
            if not digits:
                raise ValueError(f"no numeric score in response: {text!r}")
            return max(0.0, min(100.0, float(digits)))
        except Exception as exc:
            logger.warning(
                "LLM GenZ score attempt %d/%d failed for %r: %s",
                attempt, max_retries, headline[:60], exc,
            )
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2

    return None


def combine_genz_score(
    genz_source_score: float,
    genz_topic_keyword_score: float,
    subsignal_weights: dict[str, float],
    llm_genz_score: float | None = None,
) -> float:
    """Combine the GenZ/Alpha sub-signals into genz_alpha_score (0-100).
    llm_genz_score is the optional third, LLM-judged signal (see
    get_llm_genz_relevance_score above) -- when it's unavailable (feature
    disabled, or every API attempt failed), its weighted share falls back
    to genz_source_score rather than being dropped, so the formula doesn't
    silently collapse to a different weight split for headlines that never
    got an LLM call."""
    w1 = subsignal_weights.get("w1_source", 0.3)
    w2 = subsignal_weights.get("w2_topic_keyword", 0.3)
    w3 = subsignal_weights.get("w3_llm_genz", 0.4)

    llm_component = llm_genz_score if llm_genz_score is not None else genz_source_score
    score = (
        genz_source_score * w1
        + genz_topic_keyword_score * w2
        + llm_component * w3
    )
    return max(0.0, min(100.0, score))
