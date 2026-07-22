"""Touchpoint 1: a universal LLM relevance check applied to EVERY surviving
candidate for a sector, regardless of whether its publication is marked
"dedicated" or "general" in config/publications.json. The existing
keyword-based gate (fetch_and_score._gate_general_sources) only checks
"general" sources -- a "dedicated" publication's off-topic headline
currently sails through untouched. This is additive, not a replacement:
it runs after that free/instant gate (and after the thin-pool fallback
merge, so topped-up candidates get checked too) so Groq calls are only
spent on whatever survives everything cheaper first.

Fails open: if Groq is unreachable, rate-limited, or misconfigured, this
must never shrink a sector's pool to zero or halt the pipeline -- a
candidate whose classification fails after retries is KEPT, not dropped,
and counted separately in the summary log so API health stays visible.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_client import get_groq_client, get_groq_model

logger = logging.getLogger("genz_india_news.llm_relevance_filter")

_PROMPT_TEMPLATE = (
    "Headline: {headline}\n"
    "Sector: {sector}\n"
    "Is this headline genuinely about {sector}, not just tangentially "
    "mentioning it? Answer with only YES or NO."
)


def _classify_one(client: Any, model: str, headline: str, sector: str, max_retries: int) -> bool | None:
    """Ask Groq whether `headline` is genuinely about `sector`. Returns
    True/False, or None if every retry failed -- the caller decides the
    fail-open default in that case, not this function."""
    prompt = _PROMPT_TEMPLATE.format(headline=headline, sector=sector)
    backoff = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3,
                temperature=0,
            )
            text = (response.choices[0].message.content or "").strip().upper()
            return text.startswith("Y")
        except Exception as exc:
            logger.warning(
                "LLM relevance check attempt %d/%d failed for %r: %s",
                attempt, max_retries, headline[:60], exc,
            )
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2

    return None


def filter_by_llm_relevance(
    candidates: list[dict[str, Any]],
    sector: str,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """For each candidate, ask Groq whether the headline is genuinely about
    `sector`. Returns only candidates classified as relevant. Never crashes
    or empties the run: candidates that fail classification after retries
    are kept (fail open) and counted separately from genuinely off-topic
    drops, so a Groq outage never silently shrinks a sector's pool. Returns
    `candidates` unchanged if the feature is disabled or no API key is
    configured (see llm_client.get_groq_client)."""
    client = get_groq_client(settings, "use_llm_relevance_filter")
    if client is None or not candidates:
        return candidates

    model = get_groq_model(settings)
    max_retries = settings.get("groq_max_retries", 3)
    delay = settings.get("groq_request_delay_seconds", 0.3)

    kept: list[dict[str, Any]] = []
    dropped = 0
    failed_through = 0

    for i, article in enumerate(candidates):
        headline = article.get("headline", "")
        verdict = _classify_one(client, model, headline, sector, max_retries)

        if verdict is None:
            failed_through += 1
            article["_llm_relevance_passed"] = True  # fail-open: kept, but not a real classification
            kept.append(article)
        elif verdict:
            article["_llm_relevance_passed"] = True
            kept.append(article)
        else:
            dropped += 1

        if i < len(candidates) - 1:
            time.sleep(delay)

    logger.info(
        "Sector '%s' LLM relevance filter: %d in -> %d passed, %d dropped as off-topic, "
        "%d kept via fail-open after API errors.",
        sector, len(candidates), len(kept), dropped, failed_through,
    )
    return kept
