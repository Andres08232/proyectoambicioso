#!/usr/bin/env python3
"""
3D grid-search: k_factor, home_advantage_elo, form_shift_per_point (E0).

Usage:
    python scripts/optimize_elo.py
    python scripts/optimize_elo.py --csv data/raw/PremierLeague26England.csv --step 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ml.config import EngineConfig, LeagueConfig, default_engine_config  # noqa: E402
from app.ml.prediction_engine import PredictionEngine  # noqa: E402
from app.ml.value_bets import find_value_bets, summarize_backtest  # noqa: E402

# Reuse loading and backtest constants from detect_value_bets
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from detect_value_bets import EDGE_THRESHOLD, load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "PremierLeague26England.csv"
LEAGUE_CODE = "E0"

K_FACTOR_MIN = 10
K_FACTOR_MAX = 40
HOME_ADV_MIN = 50
HOME_ADV_MAX = 150
FORM_SHIFT_VALUES = [-0.05, -0.02, 0.0, 0.02, 0.05]
TOP_N = 5


def build_engine_config(
    k_factor: int,
    home_advantage_elo: int,
    form_shift_per_point: float,
) -> EngineConfig:
    """E0-only config for a single grid point; other settings stay at defaults."""
    template = default_engine_config()
    base = template.for_league(LEAGUE_CODE)
    e0_config = LeagueConfig(
        k_factor=float(k_factor),
        home_advantage_elo=float(home_advantage_elo),
        initial_rating=base.initial_rating,
        league_hfa=base.league_hfa,
        neutral_prob=base.neutral_prob,
        use_hfa_normalization=base.use_hfa_normalization,
        prob_floor=base.prob_floor,
        prob_ceiling=base.prob_ceiling,
    )
    return EngineConfig(
        default=template.default,
        leagues={LEAGUE_CODE: e0_config},
        league_column=template.league_column,
        home_team_column=template.home_team_column,
        away_team_column=template.away_team_column,
        result_column=template.result_column,
        date_column=template.date_column,
        time_column=template.time_column,
        home_goals_column=template.home_goals_column,
        away_goals_column=template.away_goals_column,
        goal_adjusted_elo=template.goal_adjusted_elo,
        use_form_modifier=True,
        form_window=template.form_window,
        form_shift_per_point=form_shift_per_point,
        form_neutral_ppg=template.form_neutral_ppg,
    )


def run_backtest(
    df: pd.DataFrame,
    k_factor: int,
    home_advantage_elo: int,
    form_shift_per_point: float,
    *,
    odds_column: str,
    edge_threshold: float,
    stake: float,
) -> dict[str, float | int]:
    """Same pipeline as detect_value_bets: predict -> value bets -> summarize."""
    config = build_engine_config(k_factor, home_advantage_elo, form_shift_per_point)
    engine = PredictionEngine(config)
    predicted = engine.attach_predictions(df)
    value_bets = find_value_bets(
        predicted,
        odds_column=odds_column,
        edge_threshold=edge_threshold,
    )
    return summarize_backtest(
        value_bets,
        odds_column=odds_column,
        stake=stake,
    )


def grid_search(
    df: pd.DataFrame,
    *,
    k_min: int,
    k_max: int,
    ha_min: int,
    ha_max: int,
    step: int,
    form_shift_values: list[float],
    odds_column: str,
    edge_threshold: float,
    stake: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    k_values = range(k_min, k_max + 1, step)
    ha_values = range(ha_min, ha_max + 1, step)
    total = len(k_values) * len(ha_values) * len(form_shift_values)

    print(
        f"Grid search: {len(k_values)} k_factor x {len(ha_values)} home_advantage "
        f"x {len(form_shift_values)} form_shift = {total} combinations"
    )

    done = 0
    for i, k_factor in enumerate(k_values, start=1):
        for home_advantage_elo in ha_values:
            for form_shift_per_point in form_shift_values:
                summary = run_backtest(
                    df,
                    k_factor,
                    home_advantage_elo,
                    form_shift_per_point,
                    odds_column=odds_column,
                    edge_threshold=edge_threshold,
                    stake=stake,
                )
                rows.append(
                    {
                        "k_factor": k_factor,
                        "home_advantage_elo": home_advantage_elo,
                        "form_shift_per_point": form_shift_per_point,
                        **summary,
                    }
                )
                done += 1
        if i % 5 == 0 or i == len(k_values):
            print(f"  Progress: {done}/{total} combinations done")

    return pd.DataFrame(rows)


def print_top_results(results: pd.DataFrame, top_n: int) -> None:
    ranked = results.sort_values(
        ["roi_pct", "total_pnl", "bets"],
        ascending=[False, False, False],
    ).head(top_n)

    print(
        f"\n--- Top {top_n} (k_factor, home_advantage_elo, form_shift_per_point) by ROI ---\n"
    )
    display = ranked[
        [
            "k_factor",
            "home_advantage_elo",
            "form_shift_per_point",
            "bets",
            "wins",
            "hit_rate_pct",
            "total_pnl",
            "roi_pct",
        ]
    ].reset_index(drop=True)
    display.index = display.index + 1
    display.index.name = "rank"

    print(
        display.to_string(
            formatters={
                "form_shift_per_point": "{:+.2f}".format,
                "hit_rate_pct": "{:.1f}%".format,
                "total_pnl": "{:+.2f}".format,
                "roi_pct": "{:+.2f}%".format,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D grid-search Elo + form parameters for Premier League (E0)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"E0 CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--odds-column",
        default="B365H",
        help="Decimal odds column (default: B365H)",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=EDGE_THRESHOLD,
        help=f"Min edge to bet (default: {EDGE_THRESHOLD})",
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=1.0,
        help="Flat stake per bet (default: 1.0)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Grid step size for k_factor and home_advantage_elo (default: 1)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=TOP_N,
        help=f"Number of top results to show (default: {TOP_N})",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = load_matches(args.csv)
    df = df[df["Div"] == LEAGUE_CODE].copy()
    if df.empty:
        raise SystemExit(
            f"No rows with Div == '{LEAGUE_CODE}' in {args.csv}. "
            "Use a Premier League Football-Data file."
        )

    print(f"Loaded {len(df)} E0 matches from {args.csv.name}")
    print(
        f"Backtest: edge > {args.edge_threshold:.2%}, "
        f"stake={args.stake}, odds={args.odds_column}"
    )
    print(f"form_shift_per_point values: {FORM_SHIFT_VALUES}")

    results = grid_search(
        df,
        k_min=K_FACTOR_MIN,
        k_max=K_FACTOR_MAX,
        ha_min=HOME_ADV_MIN,
        ha_max=HOME_ADV_MAX,
        step=max(1, args.step),
        form_shift_values=FORM_SHIFT_VALUES,
        odds_column=args.odds_column,
        edge_threshold=args.edge_threshold,
        stake=args.stake,
    )

    print_top_results(results, args.top)


if __name__ == "__main__":
    main()
