"""Text normalization, hashing, and keyword-extraction helpers shared across
the fetch, scoring, and ingestion stages."""

from __future__ import annotations

import hashlib
import re


def normalize_headline(headline: str) -> str:
    """Lowercase, strip punctuation/extra whitespace for stable dedupe/hash keys."""
    text = headline.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def headline_id(headline: str) -> str:
    """Deterministic short ID: first 8 chars of SHA256 of the normalized headline."""
    normalized = normalize_headline(headline)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:8]


def lookup_source_weight(source: str, source_weights: dict[str, float]) -> float:
    """Look up a publisher's weight (0-1) from a name->weight mapping.
    Tries an exact case-insensitive match first, then falls back to a
    substring match in either direction (RSS source names vary slightly,
    e.g. 'Times of India' vs 'The Times of India')."""
    if not source:
        return 0.0

    source_lower = source.lower().strip()
    for name, weight in source_weights.items():
        if name.lower().strip() == source_lower:
            return weight

    for name, weight in source_weights.items():
        name_lower = name.lower().strip()
        if name_lower in source_lower or source_lower in name_lower:
            return weight

    return 0.0


def count_keyword_matches(text: str, keywords: list[str]) -> int:
    """Count how many of `keywords` appear as substrings in normalized `text`."""
    normalized = normalize_headline(text)
    return sum(1 for kw in keywords if kw.lower() in normalized)
