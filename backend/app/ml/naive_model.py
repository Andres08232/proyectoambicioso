"""Legacy win-rate model (superseded by prediction_engine.PredictionEngine)."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

DEFAULT_PRIOR_PROB = 0.5
FORM_WINDOW = 5
ALL_TIME_WEIGHT = 0.5
FORM_WEIGHT = 0.5
PROB_FLOOR = 0.05
PROB_CEILING = 0.95


def _all_time_win_rate(home_results: list[int], default_prob: float) -> float:
    if not home_results:
        return default_prob
    return sum(home_results) / len(home_results)


def _form_win_rate(home_results: list[int], window: int, default_prob: float) -> float:
    if not home_results:
        return default_prob
    recent = home_results[-window:]
    return sum(recent) / len(recent)


def _hybrid_win_rate(
    home_results: list[int],
    form_window: int,
    default_prob: float,
    all_time_weight: float = ALL_TIME_WEIGHT,
    form_weight: float = FORM_WEIGHT,
) -> float:
    all_time = _all_time_win_rate(home_results, default_prob)
    form = _form_win_rate(home_results, form_window, default_prob)
    return (all_time_weight * all_time) + (form_weight * form)


def _league_hfa(home_results: list[int], default_prob: float = DEFAULT_PRIOR_PROB) -> float:
    """Average home-win rate for a league (walk-forward history)."""
    if not home_results:
        return default_prob
    return sum(home_results) / len(home_results)


def _league_adjusted_probability(
    team_rate: float,
    league_hfa: float,
    *,
    neutral_prob: float = DEFAULT_PRIOR_PROB,
) -> float:
    """
    Normalize team home strength relative to league HFA.

    model_prob = neutral + (team_rate - league_hfa)

    Centers predictions on a neutral baseline so high-HFA leagues (e.g. EPL)
    do not inflate probabilities versus lower-HFA leagues (e.g. Serie A).
    """
    adjusted = neutral_prob + (team_rate - league_hfa)
    return max(PROB_FLOOR, min(PROB_CEILING, adjusted))


def attach_walk_forward_probability(
    df: pd.DataFrame,
    default_prob: float = DEFAULT_PRIOR_PROB,
    form_window: int = FORM_WINDOW,
    all_time_weight: float = ALL_TIME_WEIGHT,
    form_weight: float = FORM_WEIGHT,
    league_column: str = "Div",
    use_hfa_adjustment: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward home-win probabilities with optional league HFA adjustment.

    Uses only matches strictly before each row's date. Same-day fixtures share
    the same priors. Teams with no home history use the league HFA as team_rate.
    Leagues with no history use default_prob.
    """
    if league_column not in df.columns:
        raise ValueError(f"Missing league column: {league_column}")

    sort_cols = ["Date", "Time"] if "Time" in df.columns else ["Date"]
    out = df.sort_values(sort_cols).reset_index(drop=True)

    team_history: dict[str, list[int]] = defaultdict(list)
    league_history: dict[str, list[int]] = defaultdict(list)
    probs: list[float] = []
    league_hfa_values: list[float] = []
    raw_team_rates: list[float] = []

    use_hybrid = form_weight > 0 and all_time_weight > 0

    for _match_date, day_rows in out.groupby("Date", sort=True):
        for _, row in day_rows.iterrows():
            team = row["HomeTeam"]
            league = row[league_column]
            league_hfa = _league_hfa(league_history[league], default_prob)

            if team_history[team]:
                raw_rate = (
                    _hybrid_win_rate(
                        team_history[team],
                        form_window,
                        league_hfa,
                        all_time_weight,
                        form_weight,
                    )
                    if use_hybrid
                    else _all_time_win_rate(team_history[team], league_hfa)
                )
            else:
                raw_rate = league_hfa

            if use_hfa_adjustment:
                prob = _league_adjusted_probability(
                    raw_rate, league_hfa, neutral_prob=default_prob
                )
            else:
                prob = raw_rate

            probs.append(prob)
            league_hfa_values.append(league_hfa)
            raw_team_rates.append(raw_rate)

        for _, row in day_rows.iterrows():
            ftr = row["FTR"]
            if pd.isna(ftr):
                continue
            outcome = 1 if ftr == "H" else 0
            team_history[row["HomeTeam"]].append(outcome)
            league_history[row[league_column]].append(outcome)

    out["model_prob"] = probs
    out["league_hfa"] = league_hfa_values
    out["raw_team_rate"] = raw_team_rates
    return out
