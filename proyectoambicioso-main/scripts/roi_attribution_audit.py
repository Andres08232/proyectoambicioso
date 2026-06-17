#!/usr/bin/env python3
"""
ROI attribution audit: economic value of Elo vs Elo+xG (research only).

Usage:
    python scripts/roi_attribution_audit.py
"""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.ml.calibration_metrics import evaluate_probs  # noqa: E402
from app.ml.config import EngineConfig, default_engine_config  # noqa: E402
from app.ml.prediction_engine import PredictionEngine  # noqa: E402
from app.ml.value_bets import implied_probability  # noqa: E402
from detect_value_bets import load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "VisionGoat_Master_WithOdds.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "roi_attribution"
ODDS_COLUMN = "B365H"
STAKE = 1.0
HYBRID_ALPHA = 0.8
EV_THRESHOLDS = [1.00, 1.02, 1.05, 1.10]
EXTREME_EV = 1.20
EXTREME_EDGE = 0.15


@dataclass
class BacktestResult:
    model: str
    ev_threshold: float
    bets: int
    wins: int
    win_rate_pct: float
    profit: float
    yield_per_bet: float
    roi_pct: float
    profit_factor: float
    avg_edge: float
    avg_ev_multiplier: float
    max_drawdown: float


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


def audit_config(*, alpha: float) -> EngineConfig:
    cfg = deepcopy(default_engine_config())
    cfg.use_probability_smoothing = False
    cfg.use_post_hoc_calibration = False
    cfg.use_form_modifier = False
    cfg.probability_scale = 530.0
    cfg.alpha = alpha
    return cfg


def attach_market_fields(df: pd.DataFrame, prob_col: str = "model_prob") -> pd.DataFrame:
    out = df.copy()
    out[ODDS_COLUMN] = pd.to_numeric(out[ODDS_COLUMN], errors="coerce")
    out = out.dropna(subset=[prob_col, ODDS_COLUMN])
    out = out[out[ODDS_COLUMN] > 0]
    out["implied_prob"] = out[ODDS_COLUMN].map(implied_probability)
    out["model_prob"] = pd.to_numeric(out[prob_col], errors="coerce")
    out["ev_multiplier"] = out["model_prob"] * out[ODDS_COLUMN]
    out["edge"] = out["model_prob"] - out["implied_prob"]
    out["disagreement"] = (out["model_prob"] - out["implied_prob"]).abs()
    return out


def max_drawdown(pnl_series: pd.Series) -> float:
    if pnl_series.empty:
        return 0.0
    cumulative = pnl_series.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    return float(drawdown.max())


def backtest_threshold(
    df: pd.DataFrame,
    model: str,
    ev_threshold: float,
) -> BacktestResult:
    bets_df = df[df["ev_multiplier"] > ev_threshold].copy()
    if bets_df.empty:
        return BacktestResult(
            model=model,
            ev_threshold=ev_threshold,
            bets=0,
            wins=0,
            win_rate_pct=0.0,
            profit=0.0,
            yield_per_bet=0.0,
            roi_pct=0.0,
            profit_factor=0.0,
            avg_edge=0.0,
            avg_ev_multiplier=0.0,
            max_drawdown=0.0,
        )

    bets_df = bets_df.sort_values("Date").reset_index(drop=True)
    bets_df["won"] = bets_df["FTR"] == "H"
    bets_df["pnl"] = np.where(
        bets_df["won"],
        STAKE * (bets_df[ODDS_COLUMN] - 1.0),
        -STAKE,
    )

    wins = int(bets_df["won"].sum())
    n = len(bets_df)
    profit = float(bets_df["pnl"].sum())
    staked = n * STAKE
    gross_win = float(bets_df.loc[bets_df["pnl"] > 0, "pnl"].sum())
    gross_loss = float(abs(bets_df.loc[bets_df["pnl"] < 0, "pnl"].sum()))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    return BacktestResult(
        model=model,
        ev_threshold=ev_threshold,
        bets=n,
        wins=wins,
        win_rate_pct=(wins / n * 100.0) if n else 0.0,
        profit=profit,
        yield_per_bet=(profit / n) if n else 0.0,
        roi_pct=(profit / staked * 100.0) if staked else 0.0,
        profit_factor=pf,
        avg_edge=float(bets_df["edge"].mean()),
        avg_ev_multiplier=float(bets_df["ev_multiplier"].mean()),
        max_drawdown=max_drawdown(bets_df["pnl"]),
    )


def market_edge_distribution(df: pd.DataFrame, model: str) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    rows.append(
        {
            "model": model,
            "metric": "n_matches_with_odds",
            "value": len(df),
        }
    )
    for col, label in (
        ("edge", "edge"),
        ("ev_multiplier", "ev_multiplier"),
        ("disagreement", "disagreement"),
    ):
        series = df[col]
        rows.append({"model": model, "metric": f"{label}_mean", "value": float(series.mean())})
        rows.append({"model": model, "metric": f"{label}_median", "value": float(series.median())})
        rows.append({"model": model, "metric": f"{label}_p90", "value": float(series.quantile(0.9))})
        rows.append({"model": model, "metric": f"{label}_p99", "value": float(series.quantile(0.99))})
        rows.append({"model": model, "metric": f"{label}_max", "value": float(series.max())})

    rows.append(
        {
            "model": model,
            "metric": "freq_ev_gt_1.10",
            "value": int((df["ev_multiplier"] > 1.10).sum()),
        }
    )
    rows.append(
        {
            "model": model,
            "metric": f"freq_ev_gt_{EXTREME_EV}",
            "value": int((df["ev_multiplier"] > EXTREME_EV).sum()),
        }
    )
    rows.append(
        {
            "model": model,
            "metric": f"freq_edge_gt_{EXTREME_EDGE}",
            "value": int((df["edge"] > EXTREME_EDGE).sum()),
        }
    )
    rows.append(
        {
            "model": model,
            "metric": "avg_disagreement",
            "value": float(df["disagreement"].mean()),
        }
    )
    return rows


def build_summary(
    results: list[BacktestResult],
    market_a: pd.DataFrame,
    market_b: pd.DataFrame,
    cal_a: dict[str, float],
    cal_b: dict[str, float],
) -> str:
    lines: list[str] = []
    lines.append("VISIONGOAT ROI ATTRIBUTION AUDIT")
    lines.append("=" * 72)
    lines.append(f"Data: {DEFAULT_CSV.name}")
    lines.append(f"Odds: {ODDS_COLUMN} | Flat stake: {STAKE} unit | Rule: model_prob * odds > threshold")
    lines.append("Models: A=Elo+xG (alpha=0.8), B=Elo-only (alpha=1.0) | No form / isotonic / smoothing")
    lines.append("")

    best = max(results, key=lambda r: r.profit)
    best_roi = max(results, key=lambda r: r.roi_pct if r.bets > 0 else -1e9)

    a_results = [r for r in results if r.model == "Model_A_Elo_xG"]
    b_results = [r for r in results if r.model == "Model_B_Elo_only"]

    def at_thresh(model_results: list[BacktestResult], t: float) -> BacktestResult:
        for r in model_results:
            if abs(r.ev_threshold - t) < 1e-9:
                return r
        raise KeyError(t)

    r_a_102 = at_thresh(a_results, 1.02)
    r_b_102 = at_thresh(b_results, 1.02)

    lines.append("1. Does xG improve ROI?")
    if r_a_102.bets and r_b_102.bets:
        lines.append(
            f"   At EV>1.02: Model A ROI {r_a_102.roi_pct:+.2f}% vs Model B {r_b_102.roi_pct:+.2f}% "
            f"(profit {r_a_102.profit:+.1f} vs {r_b_102.profit:+.1f})."
        )
        if r_a_102.roi_pct > r_b_102.roi_pct + 0.5:
            lines.append("   -> xG blend improves ROI at the reference threshold.")
        elif r_b_102.roi_pct > r_a_102.roi_pct + 0.5:
            lines.append("   -> Elo-only outperforms xG blend on ROI.")
        else:
            lines.append("   -> No material ROI difference between A and B at EV>1.02.")
    else:
        lines.append("   -> Insufficient bets at EV>1.02 to compare.")
    lines.append("")

    lines.append("2. Does xG improve edge quality?")
    lines.append(
        f"   Mean probability edge: A={market_a['edge'].mean():.4f}  B={market_b['edge'].mean():.4f}"
    )
    lines.append(
        f"   Mean EV multiplier:    A={market_a['ev_multiplier'].mean():.4f}  "
        f"B={market_b['ev_multiplier'].mean():.4f}"
    )
    lines.append(
        f"   Extreme edges (edge>{EXTREME_EDGE}): A={(market_a['edge'] > EXTREME_EDGE).sum()}  "
        f"B={(market_b['edge'] > EXTREME_EDGE).sum()}"
    )
    if market_a["edge"].mean() > market_b["edge"].mean():
        lines.append("   -> xG model shows higher average edge vs market (not necessarily profitable).")
    else:
        lines.append("   -> xG does not improve average edge vs Elo-only.")
    lines.append("")

    lines.append("3. Which model generates the highest ROI?")
    lines.append(
        f"   Best profit: {best.model} @ EV>{best.ev_threshold:.2f} -> "
        f"profit {best.profit:+.2f} units, ROI {best.roi_pct:+.2f}% ({best.bets} bets)."
    )
    lines.append(
        f"   Best ROI% (among configs with bets): {best_roi.model} @ EV>{best_roi.ev_threshold:.2f} "
        f"-> {best_roi.roi_pct:+.2f}%."
    )
    lines.append("")

    lines.append("4. Is calibration quality translating into betting profitability?")
    lines.append(
        f"   Full-sample ECE:  A={cal_a['ece']:.4f}  B={cal_b['ece']:.4f}"
    )
    lines.append(
        f"   Full-sample LL:   A={cal_a['log_loss']:.4f}  B={cal_b['log_loss']:.4f}"
    )
    better_cal = "A" if cal_a["ece"] < cal_b["ece"] else "B"
    better_roi_102 = "A" if r_a_102.roi_pct > r_b_102.roi_pct else "B"
    if better_cal != better_roi_102:
        lines.append(
            "   -> Better calibration (lower ECE) does NOT align with better ROI at EV>1.02."
        )
    else:
        lines.append(
            "   -> Lower ECE and higher ROI align for the same model at EV>1.02 (correlation only)."
        )
    if best.profit < 0:
        lines.append(
            "   -> Overall: no configuration is profitable; calibration quality alone "
            "does not yield positive flat-stake returns on this sample."
        )
    lines.append("")

    lines.append("5. At which edge threshold does the model perform best?")
    for model_name, subset in (
        ("Model_A_Elo_xG", a_results),
        ("Model_B_Elo_only", b_results),
    ):
        viable = [r for r in subset if r.bets > 0]
        if not viable:
            lines.append(f"   {model_name}: no bets at any threshold.")
            continue
        best_m = max(viable, key=lambda r: r.profit)
        lines.append(
            f"   {model_name}: best profit at EV>{best_m.ev_threshold:.2f} "
            f"(profit {best_m.profit:+.1f}, ROI {best_m.roi_pct:+.2f}%, {best_m.bets} bets)."
        )
    lines.append("")

    lines.append("DETAILED RESULTS BY THRESHOLD")
    lines.append("-" * 72)
    for r in results:
        lines.append(
            f"  {r.model} EV>{r.ev_threshold:.2f}: bets={r.bets} win%={r.win_rate_pct:.1f} "
            f"profit={r.profit:+.2f} ROI={r.roi_pct:+.2f}% PF={r.profit_factor:.2f} "
            f"avg_edge={r.avg_edge:+.4f} maxDD={r.max_drawdown:.2f}"
        )
    lines.append("")
    lines.append("END OF REPORT")
    return "\n".join(lines)


def plot_roi_vs_threshold(results: list[BacktestResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, color in (
        ("Model_A_Elo_xG", "#2563eb"),
        ("Model_B_Elo_only", "#dc2626"),
    ):
        subset = [r for r in results if r.model == model]
        xs = [r.ev_threshold for r in subset]
        ys = [r.roi_pct for r in subset]
        ax.plot(xs, ys, "o-", label=model, color=color, linewidth=2)
    ax.axhline(0, color="#666", linestyle="--", linewidth=1)
    ax.set_xlabel("EV threshold (model_prob * odds)")
    ax.set_ylabel("ROI (%)")
    ax.set_title("Flat-stake ROI vs EV threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
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
    y = (valid["FTR"] == "H").astype(int).to_numpy()

    configs = {
        "Model_A_Elo_xG": audit_config(alpha=HYBRID_ALPHA),
        "Model_B_Elo_only": audit_config(alpha=1.0),
    }

    market_frames: dict[str, pd.DataFrame] = {}
    cal_metrics: dict[str, dict[str, float]] = {}

    print("=" * 72)
    print("ROI ATTRIBUTION AUDIT")
    print("=" * 72)

    for name, cfg in configs.items():
        pred = PredictionEngine(cfg).attach_predictions(valid)
        work = attach_market_fields(pred, prob_col="model_prob_raw")
        market_frames[name] = work
        y_work = (work["FTR"] == "H").astype(int).to_numpy()
        m = evaluate_probs(y_work, work["model_prob"].to_numpy())
        cal_metrics[name] = {
            "log_loss": m.log_loss,
            "ece": m.ece,
            "brier": m.brier_score,
        }
        print(f"{name}: {len(work)} matches with odds | ECE={m.ece:.4f} LL={m.log_loss:.4f}")

    results: list[BacktestResult] = []
    for model_name, frame in market_frames.items():
        for thr in EV_THRESHOLDS:
            res = backtest_threshold(frame, model_name, thr)
            results.append(res)
            print(
                f"  {model_name} EV>{thr:.2f}: bets={res.bets} "
                f"ROI={res.roi_pct:+.2f}% profit={res.profit:+.1f}"
            )

    roi_df = pd.DataFrame([r.__dict__ for r in results])
    roi_df.to_csv(OUTPUT_DIR / "roi_by_threshold.csv", index=False)

    edge_rows: list[dict[str, float | str | int]] = []
    for model_name, frame in market_frames.items():
        edge_rows.extend(market_edge_distribution(frame, model_name))
    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(OUTPUT_DIR / "edge_distribution.csv", index=False)

    summary = build_summary(
        results,
        market_frames["Model_A_Elo_xG"],
        market_frames["Model_B_Elo_only"],
        cal_metrics["Model_A_Elo_xG"],
        cal_metrics["Model_B_Elo_only"],
    )
    (OUTPUT_DIR / "model_comparison_summary.txt").write_text(summary, encoding="utf-8")

    plot_roi_vs_threshold(results, OUTPUT_DIR / "roi_vs_threshold.png")

    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
