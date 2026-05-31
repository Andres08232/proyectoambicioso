"""Damp model probabilities toward market-implied prices."""

from __future__ import annotations

import pandas as pd

from app.ml.value_bets import implied_probability

DEFAULT_SMOOTHING_ALPHA = 0.7


def smooth_probability(
    model_prob: float,
    implied_prob: float,
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> float:
    """
    Compress model probability toward the market.

    new_prob = alpha * model_prob + (1 - alpha) * implied_market_prob
    """
    return (alpha * model_prob) + ((1.0 - alpha) * implied_prob)


def apply_probability_smoothing(
    df: pd.DataFrame,
    *,
    odds_column: str = "B365H",
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
    model_column: str = "model_prob",
    raw_column: str = "model_prob_raw",
    implied_column: str = "implied_prob",
) -> pd.DataFrame:
    """Add implied prob and replace model_column with smoothed values."""
    out = df.copy()
    odds = pd.to_numeric(out[odds_column], errors="coerce")
    valid = odds > 0

    raw = pd.to_numeric(out[model_column], errors="coerce").astype("float64")
    implied = pd.Series(index=out.index, dtype="float64")
    implied.loc[valid] = odds.loc[valid].map(implied_probability)

    smoothed = (alpha * raw) + ((1.0 - alpha) * implied)
    out[raw_column] = raw
    out[implied_column] = implied
    out[model_column] = smoothed
    return out
