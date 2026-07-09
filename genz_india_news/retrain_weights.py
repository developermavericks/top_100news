"""Stage 5: refit scoring weights from accumulated survey response data.

Reads data/training_data.csv (built by ingest_responses.py), fits a
regression from individual signal scores onto the real survey Response,
prints feature importances, then backs up and rewrites
config/scoring_weights.json with weights derived from the fit.

Safe to re-run repeatedly as more survey data accumulates -- always refits
on the full accumulated dataset.

Run manually after ingesting responses:
    python retrain_weights.py
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from config_utils import CONFIG_DIR, load_settings, setup_logging

logger = logging.getLogger("genz_india_news.retrain_weights")

TRAINING_DATA_PATH = Path("data/training_data.csv")
WEIGHTS_PATH = CONFIG_DIR / "scoring_weights.json"

MIN_ROWS_FOR_RETRAIN = 10

FULL_DIAGNOSTIC_FEATURES = [
    "india_score", "genz_source_score", "genz_topic_keyword_score", "freshness_score",
]
TOP_LEVEL_FEATURES = ["india_score", "genz_alpha_score"]
GENZ_SUBSIGNAL_FEATURES = ["genz_source_score", "genz_topic_keyword_score"]


def _is_binary(series: pd.Series) -> bool:
    return set(series.dropna().unique()).issubset({0, 1})


def _fit(df: pd.DataFrame, features: list[str], target_col: str):
    """Fit LinearRegression, or LogisticRegression if the target is binarized."""
    X = df[features].fillna(0.0)
    y = df[target_col]

    if _is_binary(y):
        model = LogisticRegression()
        model.fit(X, y)
        coefs = model.coef_[0]
    else:
        model = LinearRegression()
        model.fit(X, y)
        coefs = model.coef_

    return dict(zip(features, coefs))


def _normalize_abs_weights(coefs: dict[str, float], keys_in_order: list[str]) -> dict[str, float]:
    """Take absolute values of coefficients and normalize to sum to 1.
    Falls back to equal weighting if all coefficients are ~zero."""
    abs_values = [abs(coefs.get(k, 0.0)) for k in keys_in_order]
    total = sum(abs_values)
    if total < 1e-9:
        logger.warning("All coefficients near zero for %s; falling back to equal weights.", keys_in_order)
        equal = 1.0 / len(keys_in_order)
        return {k: equal for k in keys_in_order}
    return {k: v / total for k, v in zip(keys_in_order, abs_values)}


def retrain_weights() -> None:
    settings = load_settings()
    setup_logging(settings.get("log_file"))

    if not TRAINING_DATA_PATH.exists():
        logger.error(
            "No training data found at %s. Run ingest_responses.py first.", TRAINING_DATA_PATH
        )
        return

    df = pd.read_csv(TRAINING_DATA_PATH)
    if len(df) < MIN_ROWS_FOR_RETRAIN:
        logger.warning(
            "Only %d training rows available (recommended minimum: %d). "
            "Proceeding, but weights may be unstable with this little data.",
            len(df), MIN_ROWS_FOR_RETRAIN,
        )

    missing = [c for c in FULL_DIAGNOSTIC_FEATURES + ["Response"] if c not in df.columns]
    if missing:
        logger.error("training_data.csv is missing required columns: %s", missing)
        return

    full_coefs = _fit(df, FULL_DIAGNOSTIC_FEATURES, "Response")
    logger.info("=== Feature importances (all signals, diagnostic) ===")
    for feature, coef in sorted(full_coefs.items(), key=lambda kv: abs(kv[1]), reverse=True):
        logger.info("  %-28s %+.4f", feature, coef)

    top_level_coefs = _fit(df, TOP_LEVEL_FEATURES, "Response")
    top_level_weights = _normalize_abs_weights(top_level_coefs, TOP_LEVEL_FEATURES)

    genz_sub_coefs = _fit(df, GENZ_SUBSIGNAL_FEATURES, "Response")
    genz_sub_weights = _normalize_abs_weights(genz_sub_coefs, GENZ_SUBSIGNAL_FEATURES)

    with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
        old_weights = json.load(f)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = WEIGHTS_PATH.with_suffix(f".json.{timestamp}.bak")
    shutil.copy2(WEIGHTS_PATH, backup_path)
    logger.info("Backed up old weights to %s", backup_path)

    new_weights = dict(old_weights)
    new_weights["india_weight"] = round(top_level_weights["india_score"], 4)
    new_weights["genz_weight"] = round(top_level_weights["genz_alpha_score"], 4)
    new_weights["genz_subsignal_weights"] = {
        "w1_source": round(genz_sub_weights["genz_source_score"], 4),
        "w2_topic_keyword": round(genz_sub_weights["genz_topic_keyword_score"], 4),
    }

    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(new_weights, f, indent=2)

    logger.info("=== New weights written to %s ===", WEIGHTS_PATH)
    logger.info(
        "india_weight=%.4f genz_weight=%.4f genz_subsignal_weights=%s",
        new_weights["india_weight"], new_weights["genz_weight"], new_weights["genz_subsignal_weights"],
    )


def main() -> None:
    retrain_weights()


if __name__ == "__main__":
    main()
