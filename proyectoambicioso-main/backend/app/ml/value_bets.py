"""Implied probability, value-bet detection, and flat-stake backtest."""

from __future__ import annotations

import pandas as pd

VALUE_THRESHOLD = 1.0
DEFAULT_EDGE_GRID = [i / 100 for i in range(0, 11)]  # 0% .. 10%


def implied_probability(decimal_odds: float) -> float:
    """Bookmaker implied probability: 1 / decimal_odds."""
    return 1.0 / decimal_odds


def expected_value_multiplier(model_prob: float, decimal_odds: float) -> float:
    """EV check used for value: model_prob * decimal_odds."""
    return model_prob * decimal_odds


def is_value_bet(model_prob: float, decimal_odds: float) -> bool:
    return expected_value_multiplier(model_prob, decimal_odds) > VALUE_THRESHOLD


def find_value_bets(
    df: pd.DataFrame,
    odds_column: str = "B365H",
    edge_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Return rows where (model_prob * decimal_odds) > 1.0 and edge > edge_threshold.

    Adds implied_prob, ev_multiplier, and edge (model_prob - implied_prob).
    """
    work = df.copy()
    work[odds_column] = pd.to_numeric(work[odds_column], errors="coerce")
    work = work.dropna(subset=["model_prob", odds_column])
    work = work[work[odds_column] > 0]

    work["implied_prob"] = work[odds_column].apply(implied_probability)
    work["ev_multiplier"] = work["model_prob"] * work[odds_column]
    work["edge"] = work["model_prob"] - work["implied_prob"]

    value_mask = (work["ev_multiplier"] > VALUE_THRESHOLD) & (
        work["edge"] > edge_threshold
    )
    return work[value_mask].copy()


def attach_bet_pnl(
    value_bets: pd.DataFrame,
    odds_column: str = "B365H",
    stake: float = 1.0,
) -> pd.DataFrame:
    """Flat-stake home-win bets: win -> stake * (odds - 1), loss -> -stake."""
    out = value_bets.copy()
    out["won"] = out["FTR"] == "H"
    out["pnl"] = out.apply(
        lambda r: stake * (r[odds_column] - 1) if r["won"] else -stake,
        axis=1,
    )
    return out


def summarize_backtest(
    value_bets: pd.DataFrame,
    odds_column: str = "B365H",
    stake: float = 1.0,
) -> dict[str, float | int]:
    """Aggregate P/L and ROI for a flat-stake home-win backtest."""
    if value_bets.empty:
        return {
            "bets": 0,
            "wins": 0,
            "total_staked": 0.0,
            "total_pnl": 0.0,
            "roi_pct": 0.0,
            "hit_rate_pct": 0.0,
        }

    bets = attach_bet_pnl(value_bets, odds_column=odds_column, stake=stake)
    n = len(bets)
    wins = int(bets["won"].sum())
    total_staked = n * stake
    total_pnl = float(bets["pnl"].sum())

    return {
        "bets": n,
        "wins": wins,
        "total_staked": total_staked,
        "total_pnl": total_pnl,
        "roi_pct": (total_pnl / total_staked * 100) if total_staked else 0.0,
        "hit_rate_pct": (wins / n * 100) if n else 0.0,
    }


def find_optimal_edge_threshold(
    df: pd.DataFrame,
    odds_column: str = "B365H",
    stake: float = 1.0,
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, float | int]]:
    """Return the edge threshold that maximizes flat-stake P/L on df."""
    grid = thresholds if thresholds is not None else DEFAULT_EDGE_GRID
    best_threshold = grid[0]
    best_summary = summarize_backtest(
        find_value_bets(df, odds_column=odds_column, edge_threshold=best_threshold),
        odds_column=odds_column,
        stake=stake,
    )

    for threshold in grid[1:]:
        summary = summarize_backtest(
            find_value_bets(df, odds_column=odds_column, edge_threshold=threshold),
            odds_column=odds_column,
            stake=stake,
        )
        if summary["total_pnl"] > best_summary["total_pnl"]:
            best_threshold = threshold
            best_summary = summary

    return best_threshold, best_summary


def optimal_edge_thresholds_by_league(
    df: pd.DataFrame,
    league_column: str = "Div",
    odds_column: str = "B365H",
    stake: float = 1.0,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Grid-search edge thresholds per league; pick max-profit threshold."""
    rows: list[dict[str, float | int | str]] = []

    for league, league_df in df.groupby(league_column, sort=True):
        threshold, summary = find_optimal_edge_threshold(
            league_df,
            odds_column=odds_column,
            stake=stake,
            thresholds=thresholds,
        )
        rows.append(
            {
                "league": str(league),
                "optimal_edge": threshold,
                "bets": summary["bets"],
                "wins": summary["wins"],
                "hit_rate_pct": summary["hit_rate_pct"],
                "total_pnl": summary["total_pnl"],
                "roi_pct": summary["roi_pct"],
            }
        )

    return pd.DataFrame(rows)
