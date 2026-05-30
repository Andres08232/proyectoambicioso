#!/usr/bin/env python3
"""
Detect home-win value bets using the universal Elo PredictionEngine.

Usage:
    python scripts/detect_value_bets.py --csv-dir data/raw
    python scripts/detect_value_bets.py --csv data/raw/E0.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ml.config import EngineConfig, default_engine_config  # noqa: E402
from app.ml.prediction_engine import PredictionEngine  # noqa: E402
from app.ml.value_bets import (  # noqa: E402
    find_optimal_edge_threshold,
    find_value_bets,
    optimal_edge_thresholds_by_league,
    summarize_backtest,
)

DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "VisionGoat_Matches_xG.csv"
DEFAULT_CSV_DIR = REPO_ROOT / "data" / "raw"

EDGE_THRESHOLD = 0.02
CORE_COLUMNS = [
    "Div",
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "Home_xG",
    "Away_xG",
    "FTR",
    "B365H",
    "B365D",
    "B365A",
]

LEAGUE_LABELS = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "SP1": "La Liga",
    "F1": "Ligue 1",
}


def _derive_ftr(df: pd.DataFrame) -> None:
    """Set FTR from FTHG/FTAG when result column is missing."""
    if "FTHG" not in df.columns or "FTAG" not in df.columns:
        return
    goals_ok = df["FTHG"].notna() & df["FTAG"].notna()
    if "FTR" not in df.columns:
        df["FTR"] = pd.Series([None] * len(df), dtype=object)
    else:
        df["FTR"] = df["FTR"].astype(object)
    needs_ftr = df["FTR"].isna() & goals_ok
    if not needs_ftr.any():
        return
    df.loc[needs_ftr & (df["FTHG"] > df["FTAG"]), "FTR"] = "H"
    df.loc[needs_ftr & (df["FTHG"] < df["FTAG"]), "FTR"] = "A"
    df.loc[needs_ftr & (df["FTHG"] == df["FTAG"]), "FTR"] = "D"


def load_matches(csv_path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    missing = [column for column in CORE_COLUMNS if column not in df_raw.columns]
    if missing:
        print(
            f"Warning [{csv_path.name}]: missing expected columns: "
            + ", ".join(missing)
            + ". Continuing with available columns."
        )

    df = df_raw.reindex(columns=CORE_COLUMNS)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["source_file"] = csv_path.name

    _derive_ftr(df)

    if df["Div"].isna().all():
        league_code = "E0"
        print(
            f"Warning [{csv_path.name}]: 'Div' missing; "
            f"assigning league code '{league_code}'."
        )
        df["Div"] = league_code

    return df


def load_all_matches(csv_dir: Path) -> pd.DataFrame:
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in: {csv_dir}")

    frames = [load_matches(path) for path in csv_files]
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(csv_files)} files, {len(combined)} matches total.")
    return combined


def print_league_hfa_snapshot(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["Div", "FTR"])
    if valid.empty:
        return

    print("\n--- League Home Field Advantage (full sample reference) ---")
    hfa = valid.groupby("Div")["FTR"].apply(lambda s: (s == "H").mean())
    for league, rate in hfa.sort_index().items():
        label = LEAGUE_LABELS.get(str(league), str(league))
        print(f"  {league} ({label}): {rate:.1%} home win rate")


def print_optimal_edge_thresholds(table: pd.DataFrame) -> None:
    print("\n--- Optimal edge threshold by league (max P/L) ---")
    if table.empty:
        print("No league data available.")
        return

    display = table.copy()
    display["league_name"] = display["league"].map(
        lambda code: LEAGUE_LABELS.get(str(code), str(code))
    )
    display = display[
        [
            "league",
            "league_name",
            "optimal_edge",
            "bets",
            "hit_rate_pct",
            "total_pnl",
            "roi_pct",
        ]
    ].sort_values("league")

    print(
        display.to_string(
            index=False,
            formatters={
                "optimal_edge": "{:.2%}".format,
                "hit_rate_pct": "{:.1f}%".format,
                "total_pnl": "{:+.2f}".format,
                "roi_pct": "{:+.2f}%".format,
            },
        )
    )


def print_backtest_summary(
    summary: dict[str, float | int],
    stake: float,
    edge_threshold: float,
    title: str = "Combined backtest",
) -> None:
    print(f"\n--- {title} (Elo engine, walk-forward) ---")
    print(f"Edge threshold:    {edge_threshold:.2%} (model_prob - implied_prob)")
    print(f"Stake per bet:     {stake:.2f} units")
    print(f"Value bets placed: {summary['bets']}")
    print(f"Wins:              {summary['wins']}")
    print(f"Hit rate:          {summary['hit_rate_pct']:.1f}%")
    print(f"Total staked:      {summary['total_staked']:.2f} units")
    print(f"Total P/L:         {summary['total_pnl']:+.2f} units")
    print(f"ROI:               {summary['roi_pct']:+.2f}%")


def print_backtest_by_league(
    df: pd.DataFrame,
    odds_column: str,
    edge_threshold: float,
    stake: float,
) -> None:
    print(f"\n--- Per-league backtest @ {edge_threshold:.2%} edge ---")
    rows = []
    for league, league_df in df.groupby("Div", sort=True):
        vb = find_value_bets(
            league_df, odds_column=odds_column, edge_threshold=edge_threshold
        )
        summary = summarize_backtest(vb, odds_column=odds_column, stake=stake)
        rows.append(
            {
                "league": league,
                "name": LEAGUE_LABELS.get(str(league), str(league)),
                **summary,
            }
        )

    out = pd.DataFrame(rows)
    print(
        out[
            ["league", "name", "bets", "hit_rate_pct", "total_pnl", "roi_pct"]
        ].to_string(
            index=False,
            formatters={
                "hit_rate_pct": "{:.1f}%".format,
                "total_pnl": "{:+.2f}".format,
                "roi_pct": "{:+.2f}%".format,
            },
        )
    )


def run_predictions(
    df: pd.DataFrame, engine_config: EngineConfig | None = None
) -> pd.DataFrame:
    config = engine_config or default_engine_config()
    engine = PredictionEngine(config)
    return engine.attach_predictions(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect home-win value bets (Elo PredictionEngine)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Match CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help=f"Directory of league CSV files (overrides --csv if set)",
    )
    parser.add_argument(
        "--odds-column",
        default="B365H",
        help="Decimal odds column for home win (default: B365H)",
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=1.0,
        help="Flat stake per value bet in units (default: 1.0)",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=EDGE_THRESHOLD,
        help=f"Min edge to bet (default: {EDGE_THRESHOLD})",
    )
    parser.add_argument(
        "--skip-optimal-edge",
        action="store_true",
        help="Skip per-league optimal edge threshold scan",
    )
    args = parser.parse_args()

    if args.csv_dir is not None:
        if not args.csv_dir.exists():
            raise SystemExit(f"CSV directory not found: {args.csv_dir}")
        df = load_all_matches(args.csv_dir)
    else:
        if not args.csv.exists():
            raise SystemExit(f"CSV not found: {args.csv}")
        df = load_matches(args.csv)

    if df["Div"].isna().all():
        raise SystemExit(
            "League column 'Div' is missing. Add 'Div' to CSV or use filenames "
            "that allow inference."
        )

    print_league_hfa_snapshot(df)
    df = run_predictions(df)

    if not args.skip_optimal_edge:
        optimal_table = optimal_edge_thresholds_by_league(
            df,
            league_column="Div",
            odds_column=args.odds_column,
            stake=args.stake,
        )
        print_optimal_edge_thresholds(optimal_table)

    value_bets = find_value_bets(
        df, odds_column=args.odds_column, edge_threshold=args.edge_threshold
    )
    summary = summarize_backtest(
        value_bets, odds_column=args.odds_column, stake=args.stake
    )

    print_backtest_by_league(
        df,
        odds_column=args.odds_column,
        edge_threshold=args.edge_threshold,
        stake=args.stake,
    )
    print_backtest_summary(
        summary,
        stake=args.stake,
        edge_threshold=args.edge_threshold,
        title="Combined backtest",
    )

    combined_optimal, combined_opt_summary = find_optimal_edge_threshold(
        df, odds_column=args.odds_column, stake=args.stake
    )
    print(
        f"\nCombined optimal edge (all leagues): {combined_optimal:.2%} "
        f"-> P/L {combined_opt_summary['total_pnl']:+.2f}, "
        f"ROI {combined_opt_summary['roi_pct']:+.2f}% "
        f"({combined_opt_summary['bets']} bets)"
    )


if __name__ == "__main__":
    main()
