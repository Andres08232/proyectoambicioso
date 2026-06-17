#!/usr/bin/env python3
"""
Calibration audit: decompose miscalibration across Elo, xG-Elo, HFA, and market.

Usage:
    python scripts/calibration_audit.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.ml.config import default_engine_config  # noqa: E402
from app.ml.prediction_engine import PredictionEngine  # noqa: E402
from app.ml.value_bets import implied_probability  # noqa: E402
from detect_value_bets import load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "VisionGoat_Master_WithOdds.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "calibration_audit"
REPORT_PATH = OUTPUT_DIR / "calibration_report.txt"

SCALE_S = 530.0
PROB_EPS = 1e-15
N_CALIBRATION_BINS = 10
HIGH_CONF_THRESHOLD = 0.7
HYBRID_ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 1)
HFA_MULTIPLIERS = {
    "no_hfa": 0.0,
    "hfa_minus_50pct": 0.5,
    "hfa_baseline": 1.0,
    "hfa_plus_50pct": 1.5,
}


@dataclass
class CalibrationMetrics:
    log_loss: float
    brier_score: float
    ece: float
    oci: float
    tail_error: float
    n_samples: int


def _derive_ftr_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "FTR" not in out.columns:
        out["FTR"] = pd.NA
    out["FTR"] = out["FTR"].astype(object)
    if "FTHG" in out.columns and "FTAG" in out.columns:
        goals_ok = out["FTHG"].notna() & out["FTAG"].notna()
        needs = out["FTR"].isna() & goals_ok
        out.loc[needs & (out["FTHG"] > out["FTAG"]), "FTR"] = "H"
        out.loc[needs & (out["FTHG"] < out["FTAG"]), "FTR"] = "A"
        out.loc[needs & (out["FTHG"] == out["FTAG"]), "FTR"] = "D"
    return out


def build_audit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct Elo/xG/HFA fields via existing PredictionEngine (read-only)."""
    engine_cfg = default_engine_config()
    engine_cfg.use_probability_smoothing = False
    predicted = PredictionEngine(engine_cfg).attach_predictions(df)

    league_col = engine_cfg.league_column
    hfa_by_league = {
        str(code): engine_cfg.for_league(str(code)).home_advantage_elo
        for code in predicted[league_col].dropna().unique()
    }
    default_hfa = engine_cfg.default.home_advantage_elo

    out = predicted.copy()
    out["Home_Elo"] = pd.to_numeric(out["home_elo_traditional"], errors="coerce")
    out["Away_Elo"] = pd.to_numeric(out["away_elo_traditional"], errors="coerce")
    out["xG_Elo_Home"] = pd.to_numeric(out["home_elo_xg"], errors="coerce")
    out["xG_Elo_Away"] = pd.to_numeric(out["away_elo_xg"], errors="coerce")
    out["HFA_Value"] = out[league_col].astype(str).map(
        lambda x: hfa_by_league.get(x, default_hfa)
    )
    out["FTR"] = out[engine_cfg.result_column].astype(str)

    if "B365H" in out.columns:
        odds = pd.to_numeric(out["B365H"], errors="coerce")
        valid_odds = odds.gt(0)
        market = pd.Series(np.nan, index=out.index, dtype=float)
        market.loc[valid_odds] = odds.loc[valid_odds].map(implied_probability)
        out["Market_Prob_Home"] = market
    return out


def prepare_dataset(csv_path: Path) -> pd.DataFrame:
    raw = load_matches(csv_path)
    raw = _derive_ftr_if_needed(raw)
    featured = build_audit_features(raw)

    featured["HomeWin"] = (featured["FTR"] == "H").astype(int)
    featured["Elo_diff"] = featured["Home_Elo"] - featured["Away_Elo"]
    featured["xG_diff"] = featured["xG_Elo_Home"] - featured["xG_Elo_Away"]

    required = [
        "Home_Elo",
        "Away_Elo",
        "xG_Elo_Home",
        "xG_Elo_Away",
        "HFA_Value",
        "FTR",
        "HomeWin",
        "Elo_diff",
        "xG_diff",
    ]
    clean = featured.dropna(subset=required)
    clean = clean[clean["FTR"].isin(["H", "D", "A"])].copy()
    return clean


def logistic_prob(diff: np.ndarray, scale: float = SCALE_S) -> np.ndarray:
    prob = 1.0 / (1.0 + np.power(10.0, -diff / scale))
    return np.clip(prob, PROB_EPS, 1.0 - PROB_EPS)


def prob_model_a(df: pd.DataFrame, hfa_mult: float = 1.0) -> np.ndarray:
    diff = df["Elo_diff"].to_numpy() + (df["HFA_Value"].to_numpy() * hfa_mult)
    return logistic_prob(diff)


def prob_model_b(df: pd.DataFrame, hfa_mult: float = 1.0) -> np.ndarray:
    diff = df["xG_diff"].to_numpy() + (df["HFA_Value"].to_numpy() * hfa_mult)
    return logistic_prob(diff)


def prob_model_c(
    df: pd.DataFrame, alpha: float, hfa_mult: float = 1.0
) -> np.ndarray:
    elo_d = df["Elo_diff"].to_numpy()
    xg_d = df["xG_diff"].to_numpy()
    hfa = df["HFA_Value"].to_numpy() * hfa_mult
    diff = (alpha * elo_d) + ((1.0 - alpha) * xg_d) + hfa
    return logistic_prob(diff)


def prob_model_d(df: pd.DataFrame) -> np.ndarray:
    prob = pd.to_numeric(df["Market_Prob_Home"], errors="coerce").to_numpy(dtype=float)
    return np.clip(prob, PROB_EPS, 1.0 - PROB_EPS)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def overconfidence_index(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    high = y_prob > HIGH_CONF_THRESHOLD
    if not high.any():
        return float("nan")
    empirical = y_true[high].mean()
    return float(np.mean(np.abs(y_prob[high] - empirical)))


def tail_error(y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.10) -> float:
    if len(y_prob) == 0:
        return float("nan")
    cutoff = np.quantile(y_prob, 1.0 - top_frac)
    tail = y_prob >= cutoff
    if not tail.any():
        return float("nan")
    return float(np.mean(np.abs(y_prob[tail] - y_true[tail])))


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> CalibrationMetrics:
    valid = np.isfinite(y_prob)
    y = y_true[valid]
    p = y_prob[valid]
    return CalibrationMetrics(
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier_score=float(brier_score_loss(y, p)),
        ece=expected_calibration_error(y, p, N_CALIBRATION_BINS),
        oci=overconfidence_index(y, p),
        tail_error=tail_error(y, p),
        n_samples=int(len(y)),
    )


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int
) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            rows.append(
                {
                    "bin": b,
                    "bin_center": (bins[b] + bins[b + 1]) / 2,
                    "mean_predicted": np.nan,
                    "empirical_rate": np.nan,
                    "count": 0,
                    "gap": np.nan,
                }
            )
            continue
        mean_p = y_prob[mask].mean()
        emp = y_true[mask].mean()
        rows.append(
            {
                "bin": b,
                "bin_center": (bins[b] + bins[b + 1]) / 2,
                "mean_predicted": mean_p,
                "empirical_rate": emp,
                "count": int(mask.sum()),
                "gap": abs(mean_p - emp),
            }
        )
    return pd.DataFrame(rows)


def plot_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    rel = reliability_bins(y_true, y_prob, N_CALIBRATION_BINS)
    fig, ax = plt.subplots(figsize=(7, 7))
    valid = rel["count"] > 0
    ax.plot([0, 1], [0, 1], "--", color="#666666", label="Perfect calibration")
    ax.plot(
        rel.loc[valid, "mean_predicted"],
        rel.loc[valid, "empirical_rate"],
        marker="o",
        linewidth=2,
        color="#2563eb",
        label="Model",
    )
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_overconfidence_by_bin(
    model_probs: dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(N_CALIBRATION_BINS)
    width = 0.8 / max(len(model_probs), 1)

    for i, (name, probs) in enumerate(model_probs.items()):
        rel = reliability_bins(y_true, probs, N_CALIBRATION_BINS)
        gaps = rel["gap"].fillna(0.0)
        ax.bar(x + i * width, gaps, width=width, label=name)

    ax.set_title("|Predicted - Empirical| by confidence bin")
    ax.set_xlabel("Probability bin")
    ax.set_ylabel("Calibration gap")
    ax.set_xticks(x + width * (len(model_probs) - 1) / 2)
    ax.set_xticklabels([f"B{i}" for i in range(N_CALIBRATION_BINS)])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_hfa_sensitivity(results: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["log_loss", "ece", "oci"]
    titles = ["Log Loss", "ECE", "OCI"]
    labels = {
        "no_hfa": "0× HFA",
        "hfa_minus_50pct": "0.5× HFA",
        "hfa_baseline": "1.0× HFA",
        "hfa_plus_50pct": "1.5× HFA",
    }

    for ax, metric, title in zip(axes, metrics, titles, strict=True):
        for scenario, group in results.groupby("scenario"):
            ax.plot(
                group["alpha"],
                group[metric],
                marker="o",
                label=labels.get(scenario, scenario),
            )
        ax.set_title(f"HFA sensitivity — {title}")
        ax.set_xlabel("Hybrid alpha (Elo weight)")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_metrics_table(results: dict[str, CalibrationMetrics]) -> None:
    print("\n" + "=" * 72)
    print("CALIBRATION AUDIT — MODEL COMPARISON (s = 530)")
    print("=" * 72)
    header = f"{'Model':<12} {'LogLoss':>10} {'Brier':>10} {'ECE':>10} {'OCI':>10} {'TailErr':>10} {'N':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<12} {m.log_loss:>10.4f} {m.brier_score:>10.4f} "
            f"{m.ece:>10.4f} {m.oci:>10.4f} {m.tail_error:>10.4f} {m.n_samples:>8}"
        )


def find_best_hybrid_alpha(df: pd.DataFrame, y: np.ndarray) -> tuple[float, CalibrationMetrics]:
    best_alpha = 0.0
    best_metrics = evaluate_probs(y, prob_model_c(df, 0.0))
    rows = []
    for alpha in HYBRID_ALPHAS:
        probs = prob_model_c(df, float(alpha))
        metrics = evaluate_probs(y, probs)
        rows.append({"alpha": float(alpha), **metrics.__dict__})
        if metrics.log_loss < best_metrics.log_loss:
            best_alpha = float(alpha)
            best_metrics = metrics
    return best_alpha, best_metrics, pd.DataFrame(rows)


def build_report(
    model_metrics: dict[str, CalibrationMetrics],
    best_alpha: float,
    hybrid_search: pd.DataFrame,
    hfa_results: pd.DataFrame,
    baseline_hfa: CalibrationMetrics,
) -> str:
    lines: list[str] = []
    lines.append("VISIONGOAT CALIBRATION AUDIT REPORT")
    lines.append("=" * 60)
    lines.append(f"Scale parameter (fixed): s = {SCALE_S}")
    lines.append("")

    ranked_ll = sorted(model_metrics.items(), key=lambda x: x[1].log_loss)
    ranked_ece = sorted(model_metrics.items(), key=lambda x: x[1].ece)
    ranked_oci = sorted(model_metrics.items(), key=lambda x: x[1].oci)

    lines.append("1. MODEL RANKING (lower is better)")
    lines.append(f"   Best Log Loss: {ranked_ll[0][0]} ({ranked_ll[0][1].log_loss:.4f})")
    lines.append(f"   Best ECE:      {ranked_ece[0][0]} ({ranked_ece[0][1].ece:.4f})")
    lines.append(f"   Best OCI:      {ranked_oci[0][0]} ({ranked_oci[0][1].oci:.4f})")
    lines.append("")

    lines.append("2. COMPONENT ATTRIBUTION")
    elo_only = model_metrics["A_Elo"]
    xg_only = model_metrics["B_xG"]
    hybrid = model_metrics[f"C_Hybrid_a{best_alpha:.1f}"]
    market = model_metrics.get("D_Market")

    lines.append(
        f"   Elo-only vs xG-only log loss delta: {elo_only.log_loss - xg_only.log_loss:+.4f}"
    )
    lines.append(
        f"   Hybrid (α={best_alpha:.1f}) vs Elo log loss delta: {hybrid.log_loss - elo_only.log_loss:+.4f}"
    )
    if market:
        lines.append(
            f"   Elo-only vs Market log loss delta: {elo_only.log_loss - market.log_loss:+.4f}"
        )
    lines.append("")

    lines.append("3. OVERCONFIDENCE PROFILE")
    tail_rank = sorted(model_metrics.items(), key=lambda x: x[1].tail_error)
    lines.append(f"   Largest tail error (top 10% conf): {tail_rank[-1][0]}")
    lines.append(f"   Smallest tail error: {tail_rank[0][0]}")
    if market and elo_only.oci > market.oci:
        lines.append("   OCI suggests overconfidence is WORSE than market in high-confidence zone.")
    else:
        lines.append("   OCI vs market: model high-confidence zone is not worse than market.")
    if elo_only.tail_error > elo_only.ece:
        lines.append("   Tail error > global ECE gap → miscalibration is TAIL-DRIVEN.")
    else:
        lines.append("   Tail error comparable to ECE → miscalibration is more GLOBAL.")
    lines.append("")

    lines.append("4. HFA STRUCTURAL TEST (Model C, α grid)")
    for scenario in HFA_MULTIPLIERS:
        sub = hfa_results[hfa_results["scenario"] == scenario]
        if sub.empty:
            continue
        best_row = sub.loc[sub["log_loss"].idxmin()]
        lines.append(
            f"   {scenario}: best α={best_row['alpha']:.1f}, "
            f"LL={best_row['log_loss']:.4f}, ECE={best_row['ece']:.4f}, OCI={best_row['oci']:.4f}"
        )
    base = hfa_results[hfa_results["scenario"] == "hfa_baseline"]
    no_hfa = hfa_results[hfa_results["scenario"] == "no_hfa"]
    if not base.empty and not no_hfa.empty:
        ll_shift = no_hfa["log_loss"].min() - base["log_loss"].min()
        lines.append(f"   Removing HFA changes best log loss by {ll_shift:+.4f}")
        if abs(ll_shift) > 0.01:
            lines.append("   → HFA materially affects calibration (potential overweighting).")
        else:
            lines.append("   → HFA has limited effect on global log loss.")
    lines.append("")

    lines.append("5. xG CONTRIBUTION")
    if xg_only.log_loss < elo_only.log_loss:
        lines.append("   xG-only BEATS Elo-only on log loss → xG adds predictive signal.")
    else:
        lines.append("   xG-only does NOT beat Elo-only on log loss.")
    if xg_only.ece < elo_only.ece:
        lines.append("   xG improves calibration (lower ECE).")
    else:
        lines.append("   xG does NOT improve calibration (ECE); may only shift variance.")
    lines.append("")

    lines.append("6. RECOMMENDATIONS (diagnostic only — no config changes)")
    if ranked_ll[0][0].startswith("D"):
        lines.append("   - Market baseline wins → model must shrink toward implied probs.")
    elif ranked_ll[0][0].startswith("C"):
        lines.append(f"   - Hybrid α≈{best_alpha:.1f} is strongest internal structure.")
    elif ranked_ll[0][0].startswith("B"):
        lines.append("   - xG track alone is strongest; consider higher xG weight in blend.")
    else:
        lines.append("   - Traditional Elo track is strongest baseline component.")

    if elo_only.oci > 0.08:
        lines.append("   - Apply probability damping / stronger HFA caps before betting edges.")
    if best_alpha < 0.5:
        lines.append("   - Favor xG-Elo in next architecture iteration.")
    elif best_alpha > 0.8:
        lines.append("   - Favor traditional Elo; xG hybrid adds limited value.")
    else:
        lines.append("   - Balanced hybrid justified; tune α and separate scale per track.")

    lines.append("")
    lines.append("END OF REPORT")
    return "\n".join(lines)


def main() -> None:
    if not DEFAULT_CSV.exists():
        raise SystemExit(f"Dataset not found: {DEFAULT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(DEFAULT_CSV)
    y = df["HomeWin"].to_numpy(dtype=int)

    print(f"Matches in audit: {len(df)}")

    probs_a = prob_model_a(df)
    probs_b = prob_model_b(df)
    best_alpha, best_hybrid_metrics, hybrid_search = find_best_hybrid_alpha(df, y)
    probs_c = prob_model_c(df, best_alpha)

    model_metrics: dict[str, CalibrationMetrics] = {
        "A_Elo": evaluate_probs(y, probs_a),
        "B_xG": evaluate_probs(y, probs_b),
        f"C_Hybrid_a{best_alpha:.1f}": best_hybrid_metrics,
    }

    market_mask = df["Market_Prob_Home"].notna()
    if market_mask.any():
        y_m = y[market_mask.to_numpy()]
        probs_d = prob_model_d(df.loc[market_mask])
        model_metrics["D_Market"] = evaluate_probs(y_m, probs_d)
    else:
        probs_d = None
        print("Warning: no bookmaker odds — Model D skipped.")

    print_metrics_table(model_metrics)
    print(f"\nBest hybrid alpha (Model C, by log loss): {best_alpha:.1f}")
    print(
        f"  Log Loss={best_hybrid_metrics.log_loss:.4f}, "
        f"ECE={best_hybrid_metrics.ece:.4f}, OCI={best_hybrid_metrics.oci:.4f}"
    )

    plot_reliability_curve(y, probs_a, "Model A — Pure Elo", OUTPUT_DIR / "reliability_curve_model_A.png")
    plot_reliability_curve(y, probs_b, "Model B — Pure xG-Elo", OUTPUT_DIR / "reliability_curve_model_B.png")
    plot_reliability_curve(
        y,
        probs_c,
        f"Model C — Hybrid (alpha={best_alpha:.1f})",
        OUTPUT_DIR / "reliability_curve_model_C.png",
    )

    plot_probs = {
        "A_Elo": probs_a,
        "B_xG": probs_b,
        f"C_a{best_alpha:.1f}": probs_c,
    }
    if probs_d is not None:
        plot_probs["D_Market"] = np.full(len(y), np.nan)
        plot_probs["D_Market"][market_mask.to_numpy()] = probs_d

    # Overconfidence chart uses models with full-length vectors
    plot_overconfidence_by_bin(
        {k: v for k, v in plot_probs.items() if k != "D_Market"},
        y,
        OUTPUT_DIR / "overconfidence_by_bin.png",
    )

    hfa_rows: list[dict[str, float | str]] = []
    for scenario, mult in HFA_MULTIPLIERS.items():
        for alpha in HYBRID_ALPHAS:
            p = prob_model_c(df, float(alpha), hfa_mult=mult)
            m = evaluate_probs(y, p)
            hfa_rows.append(
                {
                    "scenario": scenario,
                    "hfa_multiplier": mult,
                    "alpha": float(alpha),
                    **m.__dict__,
                }
            )
    hfa_results = pd.DataFrame(hfa_rows)
    hfa_results.to_csv(OUTPUT_DIR / "hfa_sensitivity_results.csv", index=False)
    plot_hfa_sensitivity(hfa_results, OUTPUT_DIR / "hfa_sensitivity_plot.png")

    hybrid_search.to_csv(OUTPUT_DIR / "hybrid_alpha_search.csv", index=False)

    report = build_report(
        model_metrics,
        best_alpha,
        hybrid_search,
        hfa_results,
        model_metrics["A_Elo"],
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"\nArtifacts written to: {OUTPUT_DIR}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
