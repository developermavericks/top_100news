"""Shared helpers for loading JSON config files and setting up logging."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"


def load_json_config(filename: str) -> Any:
    """Load and parse a JSON config file from the config/ directory."""
    path = CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_settings() -> dict:
    """Load config/settings.json."""
    return load_json_config("settings.json")


def load_sectors() -> dict:
    """Load config/sectors.json (sector name -> list of search keywords)."""
    return load_json_config("sectors.json")


def load_scoring_weights() -> dict:
    """Load config/scoring_weights.json."""
    return load_json_config("scoring_weights.json")


def load_source_lists() -> dict:
    """Load config/source_lists.json (india_sources + genz_alpha_sources)."""
    return load_json_config("source_lists.json")


def load_genz_topic_keywords() -> list[str]:
    """Load config/genz_topic_keywords.json as a flat lowercase keyword list."""
    keywords = load_json_config("genz_topic_keywords.json")
    return [k.lower() for k in keywords]


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Configure root logging to stream + file handlers. Safe to call multiple times."""
    logger = logging.getLogger("genz_india_news")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_path = BASE_DIR / log_file
        os.makedirs(log_path.parent, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
