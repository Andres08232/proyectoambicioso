#!/usr/bin/env python3
"""
Diagnose extreme-edge matches: Elo vs xG vs market.

Usage:
    python scripts/diagnose_top_edges.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.ml.config import EngineConfig, LeagueConfig, default_engine_config  # noqa: E402
from app.ml.prediction_engine import PredictionEngine  # noqa: E402
from app.ml.probability_smoothing import apply_probability_smoothing  # noqa: E402
from app.ml.value_bets import implied_probability  # noqa: E402
from detect_value_bets import load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "VisionGoat_Master_WithOdds.csv"
ODDS_COLUMN = "B365H"

# Highest edges from prior analysis
TARGET_MATCHES: list[tuple[str, str, str]] = [
    ("2020-09-12", "Fulham", "Arsenal"),
    ("2020-10-04", "Aston Villa", "Liverpool"),
    ("2021-02-03", "Burnley", "Manchester City"),
]


@dataclass(frozen=True)
class MatchKey:
    date: str
    home: str
    away: str


def _config_with_elo_alpha(elo_alpha: float) -> EngineConfig:
    template = default_engine_config()
    base = template.for_league("E0")
    return EngineConfig(
        default=template.default,
        leagues={
            "E0": LeagueConfig(
                k_factor=base.k_factor,
                home_advantage_elo=base.home_advantage_elo,
                initial_rating=base.initial_rating,
                league_hfa=base.league_hfa,
                neutral_prob=base.neutral_prob,
                use_hfa_normalization=base.use_hfa_normalization,
                prob_floor=base.prob_floor,
                prob_ceiling=base.prob_ceiling,
            )
        },
        league_column=template.league_column,
        home_team_column=template.home_team_column,
        away_team_column=template.away_team_column,
        result_column=template.result_column,
        date_column=template.date_column,
        time_column=template.time_column,
        home_goals_column=template.home_goals_column,
        away_goals_column=template.away_goals_column,
        home_xg_column=template.home_xg_column,
        away_xg_column=template.away_xg_column,
        goal_adjusted_elo=template.goal_adjusted_elo,
        use_form_modifier=template.use_form_modifier,
        form_window=template.form_window,
        form_shift_per_point=template.form_shift_per_point,
        form_neutral_ppg=template.form_neutral_ppg,
        alpha=elo_alpha,
        use_probability_smoothing=False,
    )


def _run_predictions(df: pd.DataFrame, elo_alpha: float) -> pd.DataFrame:
    return PredictionEngine(_config_with_elo_alpha(elo_alpha)).attach_predictions(df)


def _match_mask(df: pd.DataFrame, key: MatchKey) -> pd.Series:
    dates = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return (
        (dates == key.date)
        & (df["HomeTeam"] == key.home)
        & (df["AwayTeam"] == key.away)
    )


def _prob_from_traditional_row(row: pd.Series, home_adv: float) -> float:
    return PredictionEngine.expected_home_win_prob(
        float(row["home_elo_traditional"]),
        float(row["away_elo_traditional"]),
        home_adv,
    )


def _prob_from_xg_row(row: pd.Series, home_adv: float) -> float:
    return PredictionEngine.expected_home_win_prob(
        float(row["home_elo_xg"]),
        float(row["away_elo_xg"]),
        home_adv,
    )


def diagnose_match(
    blended_row: pd.Series,
    trad_row: pd.Series,
    xg_row: pd.Series,
    *,
    home_adv: float,
    smoothing_alpha: float,
) -> None:
    odds = float(blended_row[ODDS_COLUMN])
    market_prob = implied_probability(odds)
    raw_model = float(blended_row["model_prob"])
    smoothed = (
        smoothing_alpha * raw_model + (1.0 - smoothing_alpha) * market_prob
    )

    trad_elo_prob = _prob_from_traditional_row(trad_row, home_adv)
    xg_elo_prob = _prob_from_xg_row(xg_row, home_adv)

    print(f"\n{'=' * 72}")
    print(
        f"{blended_row['Date']} | {blended_row['HomeTeam']} vs {blended_row['AwayTeam']} "
        f"| FTR={blended_row.get('FTR', 'N/A')}"
    )
    print(f"{'=' * 72}")

    print("\n--- Raw match data ---")
    print(f"  FTHG / FTAG:     {blended_row.get('FTHG')} / {blended_row.get('FTAG')}")
    print(f"  Home_xG / Away_xG: {blended_row.get('Home_xG')} / {blended_row.get('Away_xG')}")
    print(f"  B365H (odds):    {odds:.2f}")
    print(f"  Market_Prob:     {market_prob:.3f}  (1 / odds)")

    print("\n--- Elo ratings (pre-match, walk-forward) ---")
    print(
        f"  Traditional — Home: {trad_row['home_elo_traditional']:.1f}  "
        f"Away: {trad_row['away_elo_traditional']:.1f}"
    )
    print(
        f"  xG track    — Home: {xg_row['home_elo_xg']:.1f}  "
        f"Away: {xg_row['away_elo_xg']:.1f}"
    )
    print(
        f"  Blended     — Home: {blended_row['home_elo']:.1f}  "
        f"Away: {blended_row['away_elo']:.1f}  (elo_blend_alpha={default_engine_config().alpha})"
    )
    print(f"  league_hfa (walk-forward): {float(blended_row.get('league_hfa', 0)):.3f}")
    print(f"  form_shift:              {float(blended_row.get('form_shift', 0)):+.3f}")

    print("\n--- Home-win probabilities (before smoothing) ---")
    print(f"  From traditional Elo only: {trad_elo_prob:.3f}")
    print(f"  From xG Elo only:          {xg_elo_prob:.3f}")
    print(f"  elo_prob (blended+form+HFA): {blended_row['elo_prob']:.3f}")
    print(f"  model_prob (final):        {raw_model:.3f}")
    print(f"  Edge (raw):                {raw_model * odds:.4f}")

    print("\n--- After smoothing (alpha=0.7 toward market) ---")
    print(f"  Smoothed model_prob:       {smoothed:.3f}")
    print(f"  Edge (smoothed):           {smoothed * odds:.4f}")

    print("\n--- Diagnosis ---")
    gap_trad = trad_elo_prob - market_prob
    gap_xg = xg_elo_prob - market_prob
    gap_model = raw_model - market_prob
    print(f"  vs market — traditional: {gap_trad:+.3f}  xG: {gap_xg:+.3f}  final model: {gap_model:+.3f}")

    hfa_gap = raw_model - float(blended_row.get("prob_after_form", blended_row["elo_prob"]))
    if abs(gap_trad) >= abs(gap_xg) and abs(gap_trad) > 0.1:
        driver = "traditional Elo (results-based)"
    elif abs(gap_xg) > abs(gap_trad) and abs(gap_xg) > 0.1:
        driver = "xG Elo track"
    elif abs(hfa_gap) > 0.15:
        driver = "HFA normalization (league_hfa + elo offset)"
    elif abs(float(blended_row.get("form_shift", 0))) > 0.08:
        driver = "Form modifier"
    else:
        driver = "combined small effects"
    print(f"  Primary overconfidence driver: {driver}")

    hxg = blended_row.get("Home_xG")
    axg = blended_row.get("Away_xG")
    if pd.notna(hxg) and pd.notna(axg) and float(hxg) < float(axg) and blended_row.get("FTR") == "H":
        print(
            "  Note: Home won despite lower Home_xG — xG track may disagree with result path."
        )
    if raw_model > 0.7 and market_prob < 0.25:
        print(
            "  Note: Large home underdog (+EV on home) — check HFA normalization and form shift."
        )
    if float(blended_row.get("form_shift", 0)) > 0.05:
        print(f"  Note: Form shift adds {float(blended_row['form_shift']):+.3f} to elo_prob.")


def main() -> None:
    if not DEFAULT_CSV.exists():
        raise SystemExit(f"CSV not found: {DEFAULT_CSV}")

    df = load_matches(DEFAULT_CSV)
    cfg = default_engine_config()
    home_adv = cfg.for_league("E0").home_advantage_elo
    smoothing_alpha = cfg.probability_smoothing_alpha

    print("Running full-history predictions (blended Elo)...")
    blended = _run_predictions(df, elo_alpha=cfg.alpha)
    print("Running traditional-only Elo (alpha=1.0)...")
    trad_only = _run_predictions(df, elo_alpha=1.0)
    print("Running xG-only Elo (alpha=0.0)...")
    xg_only = _run_predictions(df, elo_alpha=0.0)

    for date_s, home, away in TARGET_MATCHES:
        key = MatchKey(date=date_s, home=home, away=away)
        mask = _match_mask(blended, key)
        if not mask.any():
            print(f"\nMatch not found: {date_s} {home} vs {away}")
            continue
        diagnose_match(
            blended.loc[mask].iloc[0],
            trad_only.loc[mask].iloc[0],
            xg_only.loc[mask].iloc[0],
            home_adv=home_adv,
            smoothing_alpha=smoothing_alpha,
        )


if __name__ == "__main__":
    main()
