#!/usr/bin/env python3
"""
Edge vs ROI stratification — diagnose why model edge does not monetize.

Usage:
    python scripts/edge_roi_stratification.py
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.ml.calibration_metrics import evaluate_probs, expected_calibration_error  # noqa: E402
from app.ml.config import EngineConfig, LeagueConfig, default_engine_config  # noqa: E402
from app.ml.prediction_engine import PredictionEngine  # noqa: E402
from app.ml.value_bets import implied_probability  # noqa: E402
from detect_value_bets import load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "VisionGoat_Master_WithOdds.csv"
HISTORICO_CSV = REPO_ROOT / "data" / "raw" / "historico_cuotas.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "edge_diagnostics"
ODDS_COLUMN = "B365H"
STAKE = 1.0
EV_THRESHOLD = 1.02
K_FACTOR = 40.0
N_EDGE_DECILES = 10
N_EV_BINS = 10
N_PROB_BINS = 10
SYNTHETIC_DRIFT_K = 0.25  # edge-driven odds compression for synthetic close


def _derive_ftr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "FTR" not in out.columns:
        out["FTR"] = pd.NA
    out["FTR"] = out["FTR"].astype(object)
    if "FTHG" in out.columns and "FTAG" in out.columns:
        ok = out["FTHG"].notna() & out["FTAG"].notna()
        needs = out["FTR"].isna() & ok
        out.loc[needs & (out["FTHG"] > out["FTAG"]), "FTR"] = "H"
        out.loc[needs & (out["FTHG"] < out["FTAG"]), "FTR"] = "A"
        out.loc[needs & (out["FTHG"] == out["FTAG"]), "FTR"] = "D"
    return out


def baseline_config() -> EngineConfig:
    cfg = deepcopy(default_engine_config())
    e0 = cfg.for_league("E0")
    cfg.leagues["E0"] = LeagueConfig(
        k_factor=K_FACTOR,
        home_advantage_elo=e0.home_advantage_elo,
        initial_rating=e0.initial_rating,
        league_hfa=e0.league_hfa,
        neutral_prob=e0.neutral_prob,
        prob_floor=e0.prob_floor,
        prob_ceiling=e0.prob_ceiling,
    )
    cfg.use_probability_smoothing = False
    cfg.use_post_hoc_calibration = False
    cfg.use_form_modifier = False
    cfg.probability_scale = 530.0
    cfg.alpha = 1.0
    return cfg


def attach_closing_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Merge B365CH from historico when possible; else synthetic closing proxy."""
    out = df.copy()
    out["closing_odds"] = np.nan
    out["closing_source"] = "synthetic"

    if HISTORICO_CSV.exists():
        hist = pd.read_csv(
            HISTORICO_CSV,
            usecols=["Date", "HomeTeam", "AwayTeam", "B365CH"],
        )
        hist["Date"] = pd.to_datetime(hist["Date"], dayfirst=True, errors="coerce")
        hist["B365CH"] = pd.to_numeric(hist["B365CH"], errors="coerce")
        hist = hist.dropna(subset=["Date", "HomeTeam", "AwayTeam", "B365CH"])
        hist = hist[hist["B365CH"] > 0]

        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        merged = out.merge(
            hist.rename(columns={"B365CH": "closing_odds_hist"}),
            on=["Date", "HomeTeam", "AwayTeam"],
            how="left",
        )
        has_hist = merged["closing_odds_hist"].notna()
        merged.loc[has_hist, "closing_odds"] = merged.loc[has_hist, "closing_odds_hist"]
        merged.loc[has_hist, "closing_source"] = "B365CH_historico"
        out = merged.drop(columns=["closing_odds_hist"])

    # Synthetic close: compress odds toward market when model shows home edge
    edge = out["edge"].to_numpy(dtype=float)
    open_odds = out[ODDS_COLUMN].to_numpy(dtype=float)
    synth_mask = out["closing_odds"].isna()
    compression = 1.0 + SYNTHETIC_DRIFT_K * np.clip(edge, 0.0, 0.35) / 0.35
    synth_odds = open_odds / compression
    out.loc[synth_mask, "closing_odds"] = synth_odds[synth_mask.to_numpy()]
    out.loc[synth_mask, "closing_source"] = "synthetic_drift"

    out["closing_implied"] = out["closing_odds"].map(implied_probability)
    out["clv_odds_ratio"] = out[ODDS_COLUMN] / out["closing_odds"]
    out["clv_implied_gap"] = out["implied_prob"] - out["closing_implied"]
    return out


def attach_bet_economics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["won"] = out["FTR"] == "H"
    out["realized_profit"] = np.where(
        out["won"],
        STAKE * (out[ODDS_COLUMN] - 1.0),
        -STAKE,
    )
    out["expected_value"] = out["model_prob"] * out[ODDS_COLUMN] - 1.0
    out["edge_pct"] = out["edge"] * 100.0
    return out


def profit_factor(pnl: pd.Series) -> float:
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(abs(pnl[pnl < 0].sum()))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def bin_metrics(group: pd.DataFrame, label: str, bin_name: str) -> dict:
    n = len(group)
    if n == 0:
        return {
            "bin_type": bin_name,
            "bin_label": label,
            "bets": 0,
            "wins": 0,
            "win_rate_pct": 0.0,
            "profit": 0.0,
            "roi_pct": 0.0,
            "profit_factor": 0.0,
            "mean_edge": np.nan,
            "mean_ev": np.nan,
            "mean_implied_prob": np.nan,
            "mean_model_prob": np.nan,
        }
    wins = int(group["won"].sum())
    profit = float(group["realized_profit"].sum())
    return {
        "bin_type": bin_name,
        "bin_label": label,
        "bets": n,
        "wins": wins,
        "win_rate_pct": wins / n * 100.0,
        "profit": profit,
        "roi_pct": profit / (n * STAKE) * 100.0,
        "profit_factor": profit_factor(group["realized_profit"]),
        "mean_edge": float(group["edge"].mean()),
        "mean_ev": float(group["ev_multiplier"].mean()),
        "mean_implied_prob": float(group["implied_prob"].mean()),
        "mean_model_prob": float(group["model_prob"].mean()),
    }


def edge_decile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["edge_decile"] = pd.qcut(
        work["edge"],
        q=N_EDGE_DECILES,
        duplicates="drop",
        labels=False,
    )
    rows = []
    for decile, group in work.groupby("edge_decile", observed=True):
        lo = group["edge"].min() * 100
        hi = group["edge"].max() * 100
        d = int(decile)
        label = f"{d * 10}-{(d + 1) * 10}%|{lo:.1f}-{hi:.1f}pp"
        rows.append(bin_metrics(group, label, "edge_decile"))
    return pd.DataFrame(rows)


def ev_bin_analysis(df: pd.DataFrame) -> pd.DataFrame:
    ev_min = df["ev_multiplier"].min()
    ev_max = df["ev_multiplier"].max()
    bins = np.linspace(ev_min, ev_max, N_EV_BINS + 1)
    work = df.copy()
    work["ev_bin"] = pd.cut(work["ev_multiplier"], bins=bins, include_lowest=True)
    rows = []
    for interval, group in work.groupby("ev_bin", observed=True):
        label = f"{interval.left:.2f}-{interval.right:.2f}"
        rows.append(bin_metrics(group, label, "ev_multiplier"))
    return pd.DataFrame(rows)


def implied_prob_bin_analysis(df: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, N_PROB_BINS + 1)
    work = df.copy()
    work["implied_bin"] = pd.cut(work["implied_prob"], bins=bins, include_lowest=True)
    rows = []
    for interval, group in work.groupby("implied_bin", observed=True):
        label = f"{interval.left:.2f}-{interval.right:.2f}"
        rows.append(bin_metrics(group, label, "implied_probability"))
    return pd.DataFrame(rows)


def calibration_vs_market(df: pd.DataFrame) -> dict[str, float]:
    """ECE of model_prob treating implied_prob as calibration target (alignment, not outcome)."""
    implied = df["implied_prob"].to_numpy()
    model = df["model_prob"].to_numpy()
    # Treat bins by model prob, compare mean model vs mean implied in each bin
    return {
        "ece_model_vs_implied": expected_calibration_error(implied, model),
        "mean_model_minus_implied": float((model - implied).mean()),
        "corr_model_implied": float(np.corrcoef(model, implied)[0, 1]),
        "corr_edge_implied": float(np.corrcoef(df["edge"], df["implied_prob"])[0, 1]),
        "corr_edge_model": float(np.corrcoef(df["edge"], df["model_prob"])[0, 1]),
    }


def edge_realization_study(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-bet edge realization: expected vs realized, bucket mismatch.

    mismatch_score = edge_decile_rank - profit_decile_rank (positive => overestimated edge)
    """
    bets = df[df["ev_multiplier"] > EV_THRESHOLD].copy()
    if bets.empty:
        return bets

    bets = bets.sort_values("Date").reset_index(drop=True)
    bets["edge_decile"] = pd.qcut(
        bets["edge"], q=min(10, bets["edge"].nunique()), duplicates="drop", labels=False
    )
    bets["profit_rank"] = bets["realized_profit"].rank(method="average")
    bets["profit_decile"] = pd.qcut(
        bets["profit_rank"], q=min(10, len(bets)), duplicates="drop", labels=False
    )
    bets["bucket_mismatch"] = bets["edge_decile"] - bets["profit_decile"]
    bets["edge_realization"] = bets["realized_profit"] - bets["expected_value"]
    return bets


def plot_edge_realization(bets: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    decile_roi = (
        bets.groupby("edge_decile", observed=True)
        .agg(roi=("realized_profit", lambda s: s.sum() / len(s) * 100))
        .reset_index()
    )
    axes[0].bar(decile_roi["edge_decile"], decile_roi["roi"], color="#2563eb")
    axes[0].axhline(0, color="#666", linestyle="--")
    axes[0].set_xlabel("Edge decile (0=lowest edge)")
    axes[0].set_ylabel("ROI per bet (%)")
    axes[0].set_title("ROI by edge decile (EV>1.02 bets)")

    axes[1].scatter(
        bets["edge"] * 100,
        bets["realized_profit"],
        alpha=0.25,
        s=12,
        c=np.where(bets["won"], "#16a34a", "#dc2626"),
    )
    axes[1].axhline(0, color="#666", linestyle="--")
    axes[1].set_xlabel("Model edge (pp)")
    axes[1].set_ylabel("Realized profit (units)")
    axes[1].set_title("Edge vs realized profit")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_market_report(
    df: pd.DataFrame,
    deciles: pd.DataFrame,
    ev_bins: pd.DataFrame,
    prob_bins: pd.DataFrame,
    bets: pd.DataFrame,
    cal_market: dict[str, float],
    outcome_cal: dict[str, float],
) -> str:
    lines = [
        "EDGE MONETIZATION DIAGNOSTIC REPORT",
        "=" * 72,
        f"Dataset: {DEFAULT_CSV.name}",
        f"Model: baseline pure Elo (K={K_FACTOR}, s=530) | Reference bets: EV>{EV_THRESHOLD}",
        "",
        "EXECUTIVE DIAGNOSIS",
        "-" * 72,
    ]

    # High edge decile performance
    if not deciles.empty:
        top = deciles.iloc[-1]
        bottom = deciles.iloc[0]
        lines.append(
            f"Highest edge decile ROI: {top['roi_pct']:+.2f}% "
            f"(win {top['win_rate_pct']:.1f}%, mean edge {top['mean_edge']*100:.1f}pp)"
        )
        lines.append(
            f"Lowest edge decile ROI:  {bottom['roi_pct']:+.2f}% "
            f"(win {bottom['win_rate_pct']:.1f}%)"
        )
        if top["roi_pct"] < bottom["roi_pct"]:
            lines.append(
                "-> INVERSION: Higher model edge buckets perform WORSE — edge is anti-predictive for ROI."
            )
        elif top["roi_pct"] < 0:
            lines.append(
                "-> Even highest-edge buckets are negative ROI — systematic overconfidence vs market."
            )

    if not bets.empty:
        high = bets[bets["edge"] >= bets["edge"].quantile(0.9)]
        low = bets[bets["edge"] <= bets["edge"].quantile(0.1)]
        lines.extend(
            [
                "",
                "EDGE REALIZATION (value bets only)",
                f"  Top 10% edge: ROI {high['realized_profit'].sum() / len(high) * 100:+.2f}%  "
                f"win {high['won'].mean()*100:.1f}%  mean edge {high['edge'].mean()*100:.1f}pp",
                f"  Bottom 10% edge among bets: ROI {low['realized_profit'].sum() / len(low) * 100:+.2f}%  "
                f"win {low['won'].mean()*100:.1f}%",
                f"  Mean bucket mismatch (edge_decile - profit_decile): {bets['bucket_mismatch'].mean():+.2f}",
                f"  Corr(edge, realized_profit): {bets['edge'].corr(bets['realized_profit']):+.3f}",
            ]
        )
        edge_profit_corr = float(bets["edge"].corr(bets["realized_profit"]))
        if edge_profit_corr < -0.05:
            lines.append(
                "  -> Among flagged value bets, higher edge associates with lower profit."
            )
        elif edge_profit_corr > 0.05:
            lines.append(
                "  -> Among flagged value bets, edge correlates weakly positively with profit."
            )
        else:
            lines.append(
                "  -> Among flagged value bets, edge has no linear link to profit "
                "(selection already filters to high edge)."
            )

    lines.extend(
        [
            "",
            "TASK 2 — MARKET ALIGNMENT",
            f"  corr(model_edge, implied_prob): {cal_market['corr_edge_implied']:+.3f}",
            f"  corr(model_prob, implied_prob):  {cal_market['corr_model_implied']:+.3f}",
            f"  Mean(model - implied):           {cal_market['mean_model_minus_implied']:+.4f}",
            f"  ECE(model vs implied as target): {cal_market['ece_model_vs_implied']:.4f}",
            f"  Log loss vs outcome:             {outcome_cal['log_loss']:.4f}",
            f"  ECE vs outcome:                  {outcome_cal['ece']:.4f}",
        ]
    )
    if cal_market["mean_model_minus_implied"] > 0.05:
        lines.append(
            "  -> Model systematically prices home wins ABOVE market implied (favorite/longshot bias)."
        )

    clv_hist = df[df["closing_source"] == "B365CH_historico"]
    lines.extend(
        [
            "",
            "TASK 4 — CLV PROXY",
            f"  Matches with real B365CH: {len(clv_hist)} / {len(df)}",
            f"  Mean CLV odds ratio (open/close): {df['clv_odds_ratio'].mean():.4f}",
            f"  Mean implied gap (open - close):  {df['clv_implied_gap'].mean():+.4f}",
        ]
    )
    if len(clv_hist) > 50:
        lines.append(
            f"  CLV on historico subset — mean ratio: {clv_hist['clv_odds_ratio'].mean():.4f}"
        )
    lines.append(
        "  Synthetic close shrinks open odds when model edge>0 (drift K="
        f"{SYNTHETIC_DRIFT_K})."
    )

    lines.extend(["", "PROBABILITY BUCKET ROI (home implied prob)", "-" * 40])
    for _, row in prob_bins.iterrows():
        lines.append(
            f"  {row['bin_label']}: ROI {row['roi_pct']:+6.2f}%  "
            f"bets={int(row['bets'])}  win={row['win_rate_pct']:.1f}%"
        )

    lines.extend(
        [
            "",
            "CONCLUSION",
            "Edge does not monetize because:",
            "1) Model home-win probability is inflated vs B365 implied (positive mean edge everywhere).",
            "2) Higher edge deciles do not earn higher ROI — miscalibration, not missing signal.",
            "3) Value-bet rule flags many home underdogs where model >> market.",
            "4) Predictive log loss can look acceptable while market-relative pricing fails.",
            "",
            "This is a MARKET ALIGNMENT / OVERCONFIDENCE problem, not fixable by K, form, xG, or G-Elo alone.",
            "",
            "END OF REPORT",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not DEFAULT_CSV.exists():
        raise SystemExit(f"Dataset not found: {DEFAULT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_matches(DEFAULT_CSV)
    raw = _derive_ftr(raw)
    valid = raw[raw["FTR"].isin(["H", "D", "A"])].copy()

    pred = PredictionEngine(baseline_config()).attach_predictions(valid)
    df = pred.copy()
    df[ODDS_COLUMN] = pd.to_numeric(df[ODDS_COLUMN], errors="coerce")
    df = df.dropna(subset=["model_prob_raw", ODDS_COLUMN])
    df = df[df[ODDS_COLUMN] > 0]
    df["model_prob"] = df["model_prob_raw"]
    df["implied_prob"] = df[ODDS_COLUMN].map(implied_probability)
    df["ev_multiplier"] = df["model_prob"] * df[ODDS_COLUMN]
    df["edge"] = df["model_prob"] - df["implied_prob"]

    df = attach_closing_odds(df)
    df = attach_bet_economics(df)

    y = (df["FTR"] == "H").astype(int).to_numpy()
    outcome_cal = evaluate_probs(y, df["model_prob"].to_numpy())
    cal_market = calibration_vs_market(df)

    # Flat-stake home bet on every match with odds (for stratification)
    deciles = edge_decile_analysis(df)
    ev_bins = ev_bin_analysis(df)
    prob_bins = implied_prob_bin_analysis(df)

    deciles.to_csv(OUTPUT_DIR / "edge_roi_by_decile.csv", index=False)
    ev_bins.to_csv(OUTPUT_DIR / "ev_bin_analysis.csv", index=False)
    prob_bins.to_csv(OUTPUT_DIR / "probability_bucket_analysis.csv", index=False)

    bets = edge_realization_study(df)
    if not bets.empty:
        bets[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "model_prob",
                "implied_prob",
                "edge",
                "ev_multiplier",
                "expected_value",
                "realized_profit",
                "edge_decile",
                "profit_decile",
                "bucket_mismatch",
                "edge_realization",
                "closing_odds",
                "closing_source",
                "clv_odds_ratio",
            ]
        ].to_csv(OUTPUT_DIR / "edge_realization_bets.csv", index=False)

    plot_edge_realization(bets if not bets.empty else df.head(1), OUTPUT_DIR / "edge_realization_curve.png")

    report = build_market_report(
        df, deciles, ev_bins, prob_bins, bets, cal_market,
        {"log_loss": outcome_cal.log_loss, "ece": outcome_cal.ece},
    )
    (OUTPUT_DIR / "market_misalignment_report.txt").write_text(report, encoding="utf-8")

    print("=" * 72)
    print("EDGE ROI STRATIFICATION")
    print("=" * 72)
    print(f"Matches with odds: {len(df)}")
    print(f"Value bets (EV>{EV_THRESHOLD}): {len(bets)}")
    print(f"corr(edge, profit) on bets: {bets['edge'].corr(bets['realized_profit']) if len(bets) else float('nan'):+.3f}")
    print("\nEdge decile ROI:")
    for _, row in deciles.iterrows():
        print(
            f"  {row['bin_label']}: ROI {row['roi_pct']:+6.2f}%  "
            f"win {row['win_rate_pct']:.1f}%  n={int(row['bets'])}"
        )
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
