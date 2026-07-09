"""India-relevance sub-score: source whitelist match + sector keyword match
strength + presence of India-specific entity keywords. No LLM, no full NER --
a simple keyword list is enough for v1."""

from __future__ import annotations

from typing import Any

from text_utils import count_keyword_matches, lookup_source_weight

# Simple, non-exhaustive list of India-specific entities: states, major
# cities, and well-known Indian companies/institutions. Good enough to catch
# "this headline is clearly India-grounded" without full NER.
INDIA_ENTITY_KEYWORDS = [
    # States / union territories
    "maharashtra", "karnataka", "tamil nadu", "uttar pradesh", "gujarat",
    "rajasthan", "kerala", "punjab", "bihar", "west bengal", "telangana",
    "andhra pradesh", "madhya pradesh", "odisha", "assam", "haryana",
    "jharkhand", "chhattisgarh", "goa", "delhi",
    # Cities
    "mumbai", "bengaluru", "bangalore", "chennai", "kolkata", "hyderabad",
    "pune", "ahmedabad", "jaipur", "lucknow", "chandigarh", "kochi",
    "surat", "nagpur", "indore", "gurugram", "gurgaon", "noida",
    # Institutions / companies / bodies
    "isro", "iit", "iim", "aiims", "rbi", "sebi", "niti aayog",
    "reliance", "tata", "infosys", "wipro", "tcs", "adani", "hcl",
    "flipkart", "paytm", "zomato", "swiggy", "ola", "byju", "bjp",
    "congress party", "lok sabha", "rajya sabha", "supreme court of india",
]


def compute_india_score(
    article: dict[str, Any],
    sector_keywords: list[str],
    india_sources: dict[str, float],
    subsignal_weights: dict[str, float],
) -> dict[str, float]:
    """Compute the India-relevance sub-score (0-100) and its component signals."""
    source_score = lookup_source_weight(article.get("source", ""), india_sources) * 100

    headline = article.get("headline", "")
    matched_keywords = count_keyword_matches(headline, sector_keywords)
    keyword_score = min(matched_keywords / max(len(sector_keywords), 1), 1.0) * 100

    matched_entities = count_keyword_matches(headline, INDIA_ENTITY_KEYWORDS)
    entity_score = min(matched_entities / 2, 1.0) * 100

    india_score = (
        source_score * subsignal_weights.get("source_weight", 0.4)
        + keyword_score * subsignal_weights.get("keyword_match_weight", 0.35)
        + entity_score * subsignal_weights.get("entity_weight", 0.25)
    )
    india_score = max(0.0, min(100.0, india_score))

    return {
        "india_source_score": source_score,
        "india_keyword_score": keyword_score,
        "india_entity_score": entity_score,
        "india_score": india_score,
    }
