"""Shared Groq client resolution for the two optional LLM touchpoints
(llm_relevance_filter.py, and genz_scorer.py's LLM GenZ-relevance signal).
Centralized here so both agree on where the API key comes from and how
each feature's on/off flag is read, rather than duplicating that logic.

Groq only -- this app is deployed on Streamlit Community Cloud, which
cannot run a local model server, so every LLM call here is an outbound
HTTPS call to Groq's API. Nothing in this module changes behavior for
someone who never sets GROQ_API_KEY: get_groq_client() returns None, and
every caller in this codebase treats None as "skip this feature, fall
back to the existing rule-based scoring."
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()  # local development: pull GROQ_API_KEY from a .env file, if present
except ImportError:
    pass  # python-dotenv not installed -- fine on Streamlit Community Cloud, which uses st.secrets instead

logger = logging.getLogger("genz_india_news.llm_client")

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

_warned_missing_key = False


def _resolve_api_key() -> str | None:
    """Streamlit's st.secrets first (how the key is supplied on Community
    Cloud), falling back to the environment (local .env via python-dotenv,
    or a real env var). Never raises -- a missing secrets.toml or a script
    running outside a Streamlit context both just fall through to the
    environment lookup."""
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def get_groq_client(settings: dict[str, Any], flag_key: str):
    """Return a configured Groq client only if BOTH `flag_key` is enabled in
    config/settings.json AND a usable API key is available; otherwise None.
    Logs a single warning the first time a key is missing (not once per
    call) so a missing key doesn't spam the log once per headline."""
    global _warned_missing_key

    if not settings.get(flag_key, False):
        return None

    api_key = _resolve_api_key()
    if not api_key:
        if not _warned_missing_key:
            logger.warning(
                "GROQ_API_KEY not found (checked st.secrets and the environment) -- "
                "LLM relevance filtering and GenZ scoring are disabled for this run; "
                "falling back to rule-based scoring only. See .env.example / README.md."
            )
            _warned_missing_key = True
        return None

    from groq import Groq
    return Groq(api_key=api_key)


def get_groq_model(settings: dict[str, Any]) -> str:
    """The configured Groq model name, e.g. 'llama-3.1-8b-instant'."""
    return settings.get("groq_model", DEFAULT_GROQ_MODEL)
