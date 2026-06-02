#!/usr/bin/env python3
"""
Conditional alpha analysis — test H1 (exploitable subsets) vs H0 (no alpha).

Usage:
    python scripts/conditional_alpha_analysis.py
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.ml.config import EngineConfig, LeagueConfig, default_engine_config  # noqa: E402
from app.ml.prediction_engine import PredictionEngine  # noqa: E402
from app.ml.value_bets import implied_probability  # noqa: E402
from detect_value_bets import load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "VisionGoat_Master_WithOdds.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "conditional_alpha"
ODDS_H, ODDS_D, ODDS_A = "B365H", "B365D", "B365A"
STAKE = 1.0
K_FACTOR = 40.0
N_DECILES = 10
AGREEMENT_QUANTILE = 0.25  # bottom 25% distance = agreement zone
PROB_EPS = 1e-15


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


def vig_free_home_implied(row: pd.Series) -> float:
    """Shin-free normalization across 1X2 when all odds present."""
    odds = [row.get(ODDS_H), row.get(ODDS_D), row.get(ODDS_A)]
    if any(pd.isna(o) or o <= 0 for o in odds):
        return implied_probability(float(row[ODDS_H]))
    inv = [1.0 / float(o) for o in odds]
    total = sum(inv)
    return inv[0] / total if total > 0 else implied_probability(float(row[ODDS_H]))


def segment_log_loss(group: pd.DataFrame) -> float:
    if len(group) < 2:
        return float("nan")
    y = (group["outcome"] == 1).astype(int).to_numpy()
    p = np.clip(group["model_prob"].to_numpy(dtype=float), PROB_EPS, 1 - PROB_EPS)
    return float(log_loss(y, p, labels=[0, 1]))


def segment_metrics(group: pd.DataFrame, segment_type: str, segment_label: str) -> dict:
    n = len(group)
    if n == 0:
        return {
            "segment_type": segment_type,
            "segment_label": segment_label,
            "n_matches": 0,
            "roi_pct": np.nan,
            "log_loss": np.nan,
            "win_rate_pct": np.nan,
            "mean_edge_pp": np.nan,
            "mean_ev": np.nan,
            "corr_ev_profit": np.nan,
            "corr_edge_profit": np.nan,
            "profit": 0.0,
        }
    profit = float(group["profit"].sum())
    ev_corr = (
        float(group["ev_multiplier"].corr(group["profit"]))
        if n > 2 and group["ev_multiplier"].std() > 0
        else float("nan")
    )
    edge_corr = (
        float(group["edge"].corr(group["profit"]))
        if n > 2 and group["edge"].std() > 0
        else float("nan")
    )
    return {
        "segment_type": segment_type,
        "segment_label": segment_label,
        "n_matches": n,
        "roi_pct": profit / (n * STAKE) * 100.0,
        "log_loss": segment_log_loss(group),
        "win_rate_pct": float(group["outcome"].mean() * 100.0),
        "mean_edge_pp": float(group["edge"].mean() * 100.0),
        "mean_ev": float(group["ev_multiplier"].mean()),
        "corr_ev_profit": ev_corr,
        "corr_edge_profit": edge_corr,
        "profit": profit,
    }


def decile_segments(
    df: pd.DataFrame, column: str, segment_type: str
) -> list[dict]:
    work = df.copy()
    work["_bin"] = pd.qcut(work[column], q=N_DECILES, duplicates="drop", labels=False)
    rows = []
    for b, g in work.groupby("_bin", observed=True):
        lo = g[column].min()
        hi = g[column].max()
        if segment_type == "model_edge":
            label = f"D{int(b)}|{lo*100:.1f}-{hi*100:.1f}pp"
        elif segment_type == "market_distance":
            label = f"D{int(b)}|{lo*100:.1f}-{hi*100:.1f}pp"
        else:
            label = f"D{int(b)}|{lo:.2f}-{hi:.2f}"
        rows.append(segment_metrics(g, segment_type, label))
    return rows


def model_prob_bins(df: pd.DataFrame) -> list[dict]:
    bins = np.linspace(0.0, 1.0, N_DECILES + 1)
    work = df.copy()
    work["_bin"] = pd.cut(work["model_prob"], bins=bins, include_lowest=True)
    rows = []
    for interval, g in work.groupby("_bin", observed=True):
        label = f"{interval.left:.2f}-{interval.right:.2f}"
        rows.append(segment_metrics(g, "model_prob_bin", label))
    return rows


def classify_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ex-post dominance (uses outcome) — forensic only, NOT tradable ex-ante.
    Ex-ante tradable regime uses agreement zone (low |model - market|) only.
    """
    out = df.copy()
    out["market_error"] = (out["market_implied_prob"] - out["outcome"]).abs()
    out["model_error"] = (out["model_prob"] - out["outcome"]).abs()
    out["model_closer_expost"] = out["model_error"] < out["market_error"]
    out["market_closer_expost"] = out["market_error"] < out["model_error"]

    dist_threshold = out["abs_distance"].quantile(AGREEMENT_QUANTILE)
    out["agreement_zone"] = out["abs_distance"] <= dist_threshold

    def regime_expost(row: pd.Series) -> str:
        if row["agreement_zone"]:
            return "C_agreement_exante"
        if row["market_closer_expost"] and not row["model_closer_expost"]:
            return "A_market_dominant_EXPOST"
        if row["model_closer_expost"] and not row["market_closer_expost"]:
            return "B_model_dominant_EXPOST"
        return "C_agreement_exante"

    out["dominance_regime_expost"] = out.apply(regime_expost, axis=1)
    # Tradable pre-match filter
    out["tradable_regime"] = np.where(
        out["agreement_zone"],
        "agreement_low_distance",
        np.where(out["edge"] > 0, "disagreement_model_high", "disagreement_market_high"),
    )
    out["outcome_correct_model"] = out["model_error"] < 0.5
    out["outcome_correct_market"] = out["market_error"] < 0.5
    return out


def ev_decile_roi_within(group: pd.DataFrame) -> list[dict]:
    if len(group) < N_DECILES:
        return []
    work = group.copy()
    work["_ev_d"] = pd.qcut(
        work["ev_multiplier"], q=min(N_DECILES, work["ev_multiplier"].nunique()),
        duplicates="drop",
        labels=False,
    )
    rows = []
    for b, g in work.groupby("_ev_d", observed=True):
        n = len(g)
        rows.append(
            {
                "ev_decile": int(b),
                "n": n,
                "mean_ev": float(g["ev_multiplier"].mean()),
                "roi_pct": float(g["profit"].sum() / (n * STAKE) * 100.0),
            }
        )
    return rows


def build_match_frame(valid: pd.DataFrame) -> pd.DataFrame:
    pred = PredictionEngine(baseline_config()).attach_predictions(valid)
    df = pred.copy()
    for col in (ODDS_H, ODDS_D, ODDS_A):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["model_prob_raw", ODDS_H])
    df = df[df[ODDS_H] > 0].copy()

    df["model_prob"] = df["model_prob_raw"].astype(float)
    df["market_implied_prob"] = df.apply(vig_free_home_implied, axis=1)
    df["raw_implied_prob"] = df[ODDS_H].map(implied_probability)
    df["abs_distance"] = (df["model_prob"] - df["market_implied_prob"]).abs()
    df["edge"] = df["model_prob"] - df["market_implied_prob"]
    df["ev_multiplier"] = df["model_prob"] * df[ODDS_H]
    df["outcome"] = (df["FTR"] == "H").astype(float)
    df["won"] = df["outcome"] == 1.0
    df["profit"] = np.where(df["won"], STAKE * (df[ODDS_H] - 1.0), -STAKE)
    return df.sort_values("Date").reset_index(drop=True)


def plot_dominance_map(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    tradable_roi = (
        df.groupby("tradable_regime")["profit"].sum()
        / df.groupby("tradable_regime").size()
        * 100.0
    )
    axes[0].bar(
        tradable_roi.index.astype(str),
        tradable_roi.values,
        color=["#16a34a", "#dc2626", "#2563eb"],
    )
    axes[0].axhline(0, color="#666", ls="--")
    axes[0].set_ylabel("ROI (%)")
    axes[0].set_title("ROI by ex-ante tradable regime")
    axes[0].tick_params(axis="x", rotation=15)

    sc = axes[1].scatter(
        df["market_implied_prob"] * 100,
        df["model_prob"] * 100,
        c=df["profit"],
        cmap="RdYlGn",
        alpha=0.45,
        s=14,
        vmin=-1,
        vmax=2,
    )
    lims = [0, 100]
    axes[1].plot(lims, lims, "--", color="#666", lw=1)
    axes[1].set_xlabel("Market implied P(home) %")
    axes[1].set_ylabel("Model P(home) %")
    axes[1].set_title("Model vs market (color = profit)")
    plt.colorbar(sc, ax=axes[1], label="Profit (u)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_regime_summary(df: pd.DataFrame, column: str, segment_type: str) -> pd.DataFrame:
    rows = []
    for regime, g in df.groupby(column, observed=True):
        base = segment_metrics(g, segment_type, regime)
        base["market_mean_error"] = float(g["market_error"].mean())
        base["model_mean_error"] = float(g["model_error"].mean())
        base["pct_model_closer"] = float(g["model_closer_expost"].mean() * 100.0)
        ev_deciles = ev_decile_roi_within(g)
        base["best_ev_decile_roi"] = (
            max((d["roi_pct"] for d in ev_deciles), default=float("nan"))
        )
        base["worst_ev_decile_roi"] = (
            min((d["roi_pct"] for d in ev_deciles), default=float("nan"))
        )
        rows.append(base)
    return pd.DataFrame(rows)


def build_ev_quality_report(
    df: pd.DataFrame,
    segment_df: pd.DataFrame,
    regime_expost_df: pd.DataFrame,
    regime_tradable_df: pd.DataFrame,
) -> str:
    positive_roi = segment_df[segment_df["roi_pct"] > 0]
    best_seg = segment_df.loc[segment_df["roi_pct"].idxmax()]
    tradable_best = regime_tradable_df.loc[regime_tradable_df["roi_pct"].idxmax()]
    any_tradable_positive = (regime_tradable_df["roi_pct"] > 0).any()
    any_decile_positive = int((segment_df["roi_pct"] > 0).sum())
    expost_model = regime_expost_df[
        regime_expost_df["segment_label"].str.contains("model_dominant", na=False)
    ]

    lines = [
        "CONDITIONAL ALPHA HYPOTHESIS TEST",
        "=" * 72,
        f"N matches: {len(df)}",
        "Model: baseline Elo K=40 s=530 | Flat home stake | Vig-free 1X2 implied",
        "",
        "H1: Exploitable conditional alpha exists in some regimes.",
        "H0: All edge is noise / market already prices information.",
        "",
        "TASK 1 — SEGMENT SCAN",
        f"  Segments with positive ROI: {len(positive_roi)} / {len(segment_df)}",
        f"  Best segment: {best_seg['segment_type']} {best_seg['segment_label']} "
        f"ROI {best_seg['roi_pct']:+.2f}% (n={int(best_seg['n_matches'])})",
    ]
    for stype in ("model_edge", "market_distance", "model_prob_bin"):
        sub = segment_df[segment_df["segment_type"] == stype]
        if sub.empty:
            continue
        pos = sub[sub["roi_pct"] > 0]
        lines.append(
            f"  {stype}: {len(pos)}/{len(sub)} deciles positive | "
            f"ROI range [{sub['roi_pct'].min():+.1f}%, {sub['roi_pct'].max():+.1f}%]"
        )

    lines.extend(
        [
            "",
            "TASK 2a — EX-ANTE TRADABLE REGIMES (no outcome lookahead)",
        ]
    )
    for _, r in regime_tradable_df.iterrows():
        lines.append(
            f"  {r['segment_label']}: n={int(r['n_matches'])} "
            f"ROI {r['roi_pct']:+.2f}% LL {r['log_loss']:.4f} "
            f"mean_edge {r['mean_edge_pp']:+.1f}pp "
            f"corr(EV,profit)={r['corr_ev_profit']:+.3f}"
        )

    lines.extend(
        [
            "",
            "TASK 2b — EX-POST DOMINANCE (forensic only — uses outcome)",
            "  WARNING: B_model_dominant_EXPOST is circular (97% win rate = selected post-hoc).",
        ]
    )
    for _, r in regime_expost_df.iterrows():
        lines.append(
            f"  {r['segment_label']}: n={int(r['n_matches'])} "
            f"ROI {r['roi_pct']:+.2f}% win {r['win_rate_pct']:.1f}%"
        )

    lines.extend(["", "TASK 3 — EDGE QUALITY (ex-ante tradable)"])
    for _, r in regime_tradable_df.iterrows():
        lines.append(
            f"  {r['segment_label']}: corr(edge,profit)={r['corr_edge_profit']:+.3f} "
            f"EV decile ROI [{r['worst_ev_decile_roi']:+.1f}%, {r['best_ev_decile_roi']:+.1f}%]"
        )

    lines.extend(["", "DECISION (ex-ante tradable subsets only)"])
    if (
        any_tradable_positive
        and tradable_best["roi_pct"] > 2.0
        and tradable_best["n_matches"] >= 150
    ):
        lines.append(
            f"  WEAK H1: Best tradable slice '{tradable_best['segment_label']}' "
            f"ROI {tradable_best['roi_pct']:+.2f}% — needs walk-forward confirmation."
        )
    else:
        lines.append(
            "  FAIL TO REJECT H0 (tradable): No ex-ante regime with robust positive ROI."
        )

    if not expost_model.empty and expost_model.iloc[0]["roi_pct"] > 50:
        lines.append(
            "  Ex-post 'model dominant' ROI is NOT actionable — built with outcome knowledge."
        )

    lines.append(
        f"\n  Best ex-ante tradable regime: {tradable_best['segment_label']} "
        f"({tradable_best['roi_pct']:+.2f}%)."
    )
    lines.append(
        "\n  RECOMMENDATION: Market re-anchoring required. Do not deploy unconditional "
        "or raw EV>1.02 rules. Any strategy must filter on PRE-MATCH agreement/low distance only."
    )
    lines.extend(["", "END OF REPORT"])
    return "\n".join(lines)


def main() -> None:
    if not DEFAULT_CSV.exists():
        raise SystemExit(f"Dataset not found: {DEFAULT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_matches(DEFAULT_CSV)
    raw = _derive_ftr(raw)
    valid = raw[raw["FTR"].isin(["H", "D", "A"])].copy()
    df = classify_dominance(build_match_frame(valid))

    segment_rows: list[dict] = []
    segment_rows.extend(decile_segments(df, "edge", "model_edge"))
    segment_rows.extend(decile_segments(df, "abs_distance", "market_distance"))
    segment_rows.extend(model_prob_bins(df))

    segment_df = pd.DataFrame(segment_rows)
    regime_expost_df = build_regime_summary(df, "dominance_regime_expost", "expost_dominance")
    regime_tradable_df = build_regime_summary(df, "tradable_regime", "exante_tradable")
    regime_df = pd.concat([regime_tradable_df, regime_expost_df], ignore_index=True)

    segment_df.to_csv(OUTPUT_DIR / "roi_by_segment.csv", index=False)
    regime_df.to_csv(OUTPUT_DIR / "alpha_regime_summary.csv", index=False)

    plot_dominance_map(df, OUTPUT_DIR / "model_vs_market_dominance.png")

    report = build_ev_quality_report(
        df, segment_df, regime_expost_df, regime_tradable_df
    )
    (OUTPUT_DIR / "ev_quality_by_regime.txt").write_text(report, encoding="utf-8")

    print("=" * 72)
    print("CONDITIONAL ALPHA ANALYSIS")
    print("=" * 72)
    print(f"Matches: {len(df)}")
    print("\nEx-ante tradable regimes:")
    print(
        regime_tradable_df[
            ["segment_label", "n_matches", "roi_pct", "corr_ev_profit"]
        ].to_string(index=False)
    )
    pos = segment_df[segment_df["roi_pct"] > 0]
    print(f"\nPositive-ROI segments: {len(pos)} / {len(segment_df)}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
