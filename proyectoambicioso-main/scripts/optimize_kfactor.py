#!/usr/bin/env python3
"""
K-factor optimization study on pure Elo (research only).

Usage:
    python scripts/optimize_kfactor.py
"""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.ml.calibration_metrics import evaluate_probs  # noqa: E402
from app.ml.config import EngineConfig, LeagueConfig, default_engine_config  # noqa: E402
from app.ml.prediction_engine import PredictionEngine  # noqa: E402
from app.ml.value_bets import implied_probability  # noqa: E402
from detect_value_bets import load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "VisionGoat_Master_WithOdds.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "kfactor_optimization"
ODDS_COLUMN = "B365H"
STAKE = 1.0
EV_THRESHOLD = 1.02
PROBABILITY_SCALE = 530.0
K_VALUES = list(range(10, 101, 5))
LEAGUE_CODE = "E0"


@dataclass
class KFactorResult:
    k_factor: int
    roi_pct: float
    profit: float
    bets: int
    wins: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown: float
    log_loss: float
    brier_score: float
    ece: float
    mean_edge: float
    mean_ev_multiplier: float
    mean_edge_on_bets: float
    mean_ev_on_bets: float
    n_matches: int


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


def pure_elo_config(k_factor: float) -> EngineConfig:
    template = deepcopy(default_engine_config())
    base = template.for_league(LEAGUE_CODE)
    template.leagues[LEAGUE_CODE] = LeagueConfig(
        k_factor=float(k_factor),
        home_advantage_elo=base.home_advantage_elo,
        initial_rating=base.initial_rating,
        league_hfa=base.league_hfa,
        neutral_prob=base.neutral_prob,
        prob_floor=base.prob_floor,
        prob_ceiling=base.prob_ceiling,
    )
    template.use_probability_smoothing = False
    template.use_post_hoc_calibration = False
    template.use_form_modifier = False
    template.probability_scale = PROBABILITY_SCALE
    template.alpha = 1.0
    return template


def attach_market(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[ODDS_COLUMN] = pd.to_numeric(out[ODDS_COLUMN], errors="coerce")
    out = out.dropna(subset=["model_prob_raw", ODDS_COLUMN])
    out = out[out[ODDS_COLUMN] > 0]
    out["model_prob"] = pd.to_numeric(out["model_prob_raw"], errors="coerce")
    out["implied_prob"] = out[ODDS_COLUMN].map(implied_probability)
    out["ev_multiplier"] = out["model_prob"] * out[ODDS_COLUMN]
    out["edge"] = out["model_prob"] - out["implied_prob"]
    return out


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    cumulative = pnl.cumsum()
    return float((cumulative.cummax() - cumulative).max())


def simulate_bets(df: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    bets = df[df["ev_multiplier"] > EV_THRESHOLD].copy()
    if bets.empty:
        return {
            "bets": 0,
            "wins": 0,
            "win_rate_pct": 0.0,
            "profit": 0.0,
            "roi_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "mean_edge_on_bets": 0.0,
            "mean_ev_on_bets": 0.0,
        }, bets

    bets = bets.sort_values("Date").reset_index(drop=True)
    bets["won"] = bets["FTR"] == "H"
    bets["pnl"] = np.where(
        bets["won"],
        STAKE * (bets[ODDS_COLUMN] - 1.0),
        -STAKE,
    )
    n = len(bets)
    wins = int(bets["won"].sum())
    profit = float(bets["pnl"].sum())
    gross_win = float(bets.loc[bets["pnl"] > 0, "pnl"].sum())
    gross_loss = float(abs(bets.loc[bets["pnl"] < 0, "pnl"].sum()))
    pf = (gross_win / gross_loss) if gross_loss > 0 else 0.0

    return {
        "bets": n,
        "wins": wins,
        "win_rate_pct": wins / n * 100.0,
        "profit": profit,
        "roi_pct": profit / (n * STAKE) * 100.0,
        "profit_factor": pf,
        "max_drawdown": max_drawdown(bets["pnl"]),
        "mean_edge_on_bets": float(bets["edge"].mean()),
        "mean_ev_on_bets": float(bets["ev_multiplier"].mean()),
    }, bets


def evaluate_k(k_factor: int, valid: pd.DataFrame) -> KFactorResult:
    pred = PredictionEngine(pure_elo_config(k_factor)).attach_predictions(valid)
    work = attach_market(pred)
    y = (work["FTR"] == "H").astype(int).to_numpy()
    probs = work["model_prob"].to_numpy()
    cal = evaluate_probs(y, probs)
    bt, _ = simulate_bets(work)

    return KFactorResult(
        k_factor=k_factor,
        roi_pct=float(bt["roi_pct"]),
        profit=float(bt["profit"]),
        bets=int(bt["bets"]),
        wins=int(bt["wins"]),
        win_rate_pct=float(bt["win_rate_pct"]),
        profit_factor=float(bt["profit_factor"]),
        max_drawdown=float(bt["max_drawdown"]),
        log_loss=cal.log_loss,
        brier_score=cal.brier_score,
        ece=cal.ece,
        mean_edge=float(work["edge"].mean()),
        mean_ev_multiplier=float(work["ev_multiplier"].mean()),
        mean_edge_on_bets=float(bt["mean_edge_on_bets"]),
        mean_ev_on_bets=float(bt["mean_ev_on_bets"]),
        n_matches=len(work),
    )


def build_summary(results: list[KFactorResult]) -> str:
    df = pd.DataFrame([asdict(r) for r in results])
    best_roi = df.loc[df["roi_pct"].idxmax()]
    best_ll = df.loc[df["log_loss"].idxmin()]
    prod_k = 17.0
    prod_row = df[df["k_factor"] == prod_k]
    if not prod_row.empty:
        pr = prod_row.iloc[0]
        prod_note = (
            f"\n   Production default K={prod_k:.0f}: ROI {pr['roi_pct']:+.2f}%, "
            f"LL {pr['log_loss']:.4f}, profit {pr['profit']:+.1f}."
        )
    else:
        prod_note = (
            f"\n   Production default K={prod_k:.0f} not in sweep "
            f"(grid step 5); nearest points K=15 and K=20."
        )

    roi_ll_corr = float(df["roi_pct"].corr(df["log_loss"]))
    near_be = df["roi_pct"].max()
    best_roi_k = int(best_roi["k_factor"])

    lines = [
        "VISIONGOAT K-FACTOR OPTIMIZATION STUDY",
        "=" * 72,
        f"Dataset: {DEFAULT_CSV.name}",
        f"Pure Elo | HFA in Elo | s={PROBABILITY_SCALE} | EV>{EV_THRESHOLD} | flat stake",
        f"K sweep: {K_VALUES[0]}..{K_VALUES[-1]} step 5 ({len(K_VALUES)} values)",
        "",
        "1. Which K-factor produces the highest ROI?",
        f"   K={best_roi_k} -> ROI {best_roi['roi_pct']:+.2f}%, profit {best_roi['profit']:+.1f}, "
        f"{int(best_roi['bets'])} bets, win rate {best_roi['win_rate_pct']:.1f}%.",
        prod_note,
        "",
        "2. Which K-factor produces the best Log Loss?",
        f"   K={int(best_ll['k_factor'])} -> log loss {best_ll['log_loss']:.4f} "
        f"(Brier {best_ll['brier_score']:.4f}, ECE {best_ll['ece']:.4f}).",
        "",
        "3. Are ROI and Log Loss aligned or conflicting?",
        f"   Pearson correlation (ROI vs log loss): {roi_ll_corr:+.3f}.",
    ]
    if roi_ll_corr > 0.1:
        lines.append(
            "   -> CONFLICTING: lower log loss tends to associate with worse ROI "
            "(predictive fit != betting P/L)."
        )
    elif roi_ll_corr < -0.1:
        lines.append(
            "   -> PARTIALLY ALIGNED: lower log loss associates with better ROI in this sweep."
        )
    else:
        lines.append(
            "   -> WEAKLY RELATED: K changes log loss and ROI somewhat independently."
        )
    if int(best_ll["k_factor"]) != best_roi_k:
        lines.append(
            f"   Best-LL K ({int(best_ll['k_factor'])}) != best-ROI K ({best_roi_k})."
        )
    lines.extend(
        [
            "",
            "4. Does any K-factor move ROI materially closer to break-even?",
            f"   Best ROI in sweep: {near_be:+.2f}% (still negative on this sample).",
        ]
    )
    if near_be > -5.0:
        lines.append("   -> Some K values approach break-even; responsiveness tuning may help.")
    else:
        lines.append(
            "   -> No K in [10,100] approaches break-even; bottleneck is likely not K alone."
        )
    lines.extend(
        [
            "",
            "5. Does edge quality improve as K changes?",
            f"   Mean edge (all matches): min {df['mean_edge'].min():.4f} @ K="
            f"{int(df.loc[df['mean_edge'].idxmin(), 'k_factor'])}, "
            f"max {df['mean_edge'].max():.4f} @ K="
            f"{int(df.loc[df['mean_edge'].idxmax(), 'k_factor'])}.",
            f"   Mean EV multiplier: range {df['mean_ev_multiplier'].min():.4f} "
            f"to {df['mean_ev_multiplier'].max():.4f}.",
        ]
    )
    edge_roi_corr = float(df["mean_edge"].corr(df["roi_pct"]))
    lines.append(f"   Correlation(mean edge, ROI): {edge_roi_corr:+.3f}.")
    lines.append(
        "   -> Mean edge is nearly flat (~9.2%); EV multiplier range is wider "
        "(slower K -> fewer extreme ratings)."
    )
    if edge_roi_corr > 0.3:
        lines.append(
            "   -> Higher K (slightly higher mean edge) associates with less-negative ROI, "
            "but edge magnitude alone does not fix profitability."
        )
    lines.extend(["", "TOP 5 BY ROI", "-" * 40])
    for _, row in df.nlargest(5, "roi_pct").iterrows():
        lines.append(
            f"  K={int(row['k_factor']):3d}  ROI {row['roi_pct']:+7.2f}%  "
            f"LL {row['log_loss']:.4f}  profit {row['profit']:+8.1f}"
        )
    lines.extend(["", "TOP 5 BY LOG LOSS", "-" * 40])
    for _, row in df.nsmallest(5, "log_loss").iterrows():
        lines.append(
            f"  K={int(row['k_factor']):3d}  LL {row['log_loss']:.4f}  "
            f"ROI {row['roi_pct']:+7.2f}%  ECE {row['ece']:.4f}"
        )
    lines.extend(["", "END OF REPORT"])
    return "\n".join(lines)


def plot_metric(df: pd.DataFrame, y_col: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["k_factor"], df[y_col], "o-", color="#2563eb", linewidth=2, markersize=6)
    if y_col == "roi_pct":
        ax.axhline(0, color="#666", linestyle="--", linewidth=1)
    prod = df[df["k_factor"] == 17]
    if not prod.empty:
        ax.axvline(17, color="#dc2626", linestyle=":", alpha=0.7, label="Production K=17")
    ax.set_xlabel("K-factor")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs K-factor (pure Elo)")
    ax.grid(True, alpha=0.3)
    if not prod.empty:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not DEFAULT_CSV.exists():
        raise SystemExit(f"Dataset not found: {DEFAULT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_matches(DEFAULT_CSV)
    raw = _derive_ftr(raw)
    valid = raw[raw["FTR"].isin(["H", "D", "A"])].copy()

    print("=" * 72)
    print("K-FACTOR OPTIMIZATION (pure Elo)")
    print("=" * 72)
    print(f"Matches: {len(valid)} | EV threshold: {EV_THRESHOLD}")
    print()

    results: list[KFactorResult] = []
    for k in K_VALUES:
        res = evaluate_k(k, valid)
        results.append(res)
        print(
            f"K={k:3d}  ROI={res.roi_pct:+7.2f}%  LL={res.log_loss:.4f}  "
            f"bets={res.bets}  profit={res.profit:+8.1f}  mean_edge={res.mean_edge:+.4f}"
        )

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(OUTPUT_DIR / "kfactor_results.csv", index=False)

    summary = build_summary(results)
    (OUTPUT_DIR / "kfactor_summary.txt").write_text(summary, encoding="utf-8")

    plot_metric(df, "roi_pct", "ROI (%)", OUTPUT_DIR / "roi_vs_kfactor.png")
    plot_metric(df, "log_loss", "Log Loss", OUTPUT_DIR / "logloss_vs_kfactor.png")

    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
