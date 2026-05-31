#!/usr/bin/env python3
"""
Find optimal logistic scale parameter (s) for Elo -> home win probability.

Minimizes log loss on VisionGoat master data with walk-forward Elo ratings.

Usage:
    python scripts/optimize_scale_parameter.py
"""

from __future__ import annotations

import sys
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
from detect_value_bets import load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "VisionGoat_Master_WithOdds.csv"
RESULTS_CSV = REPO_ROOT / "data" / "processed" / "scale_parameter_results.csv"
PLOT_PATH = REPO_ROOT / "data" / "processed" / "scale_optimization.png"

SCALE_MIN = 200
SCALE_MAX = 800
SCALE_STEP = 10
PROB_EPS = 1e-15


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


def build_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach Elo inputs required for scale optimization.

    Maps engine outputs to expected names:
      Home_Elo, Away_Elo, HFA_Value (home advantage in Elo points), FTR
    """
    engine_cfg = default_engine_config()
    engine_cfg.use_probability_smoothing = False
    predicted = PredictionEngine(engine_cfg).attach_predictions(df)

    league_col = engine_cfg.league_column
    hfa_by_league = {
        code: engine_cfg.for_league(code).home_advantage_elo
        for code in predicted[league_col].dropna().unique()
    }
    default_hfa = engine_cfg.default.home_advantage_elo

    out = predicted.copy()
    out["Home_Elo"] = pd.to_numeric(out["home_elo"], errors="coerce")
    out["Away_Elo"] = pd.to_numeric(out["away_elo"], errors="coerce")
    out["HFA_Value"] = out[league_col].map(lambda x: hfa_by_league.get(x, default_hfa))
    out["FTR"] = out[engine_cfg.result_column].astype(str)
    return out


def prepare_dataset(csv_path: Path) -> pd.DataFrame:
    raw = load_matches(csv_path)
    raw = _derive_ftr_if_needed(raw)
    featured = build_elo_features(raw)

    featured["HomeWin"] = (featured["FTR"] == "H").astype(int)
    required = ["Home_Elo", "Away_Elo", "HFA_Value", "FTR", "HomeWin"]
    clean = featured.dropna(subset=["Home_Elo", "Away_Elo", "HFA_Value", "FTR"])
    clean = clean[clean["FTR"].isin(["H", "D", "A"])].copy()
    return clean


def elo_home_prob(
    home_elo: np.ndarray,
    away_elo: np.ndarray,
    hfa_value: np.ndarray,
    scale: float,
) -> np.ndarray:
    adjusted_diff = (home_elo + hfa_value) - away_elo
    prob = 1.0 / (1.0 + np.power(10.0, -adjusted_diff / scale))
    return np.clip(prob, PROB_EPS, 1.0 - PROB_EPS)


def evaluate_scales(
    df: pd.DataFrame,
    scales: np.ndarray,
) -> pd.DataFrame:
    home_elo = df["Home_Elo"].to_numpy(dtype=float)
    away_elo = df["Away_Elo"].to_numpy(dtype=float)
    hfa = df["HFA_Value"].to_numpy(dtype=float)
    y_true = df["HomeWin"].to_numpy(dtype=int)

    rows: list[dict[str, float]] = []
    for scale in scales:
        prob = elo_home_prob(home_elo, away_elo, hfa, float(scale))
        ll = log_loss(y_true, prob, labels=[0, 1])
        brier = brier_score_loss(y_true, prob)
        rows.append(
            {
                "scale_parameter": float(scale),
                "log_loss": float(ll),
                "brier_score": float(brier),
            }
        )
    return pd.DataFrame(rows)


def print_summary(best_row: pd.Series, n_matches: int) -> None:
    print("\n===================================")
    print("SCALE PARAMETER OPTIMIZATION")
    print("===================================\n")
    print(f"Best Scale: {int(best_row['scale_parameter'])}")
    print(f"Best Log Loss: {best_row['log_loss']:.4f}")
    print(f"Brier Score: {best_row['brier_score']:.4f}")
    print(f"Matches Used: {n_matches}")


def save_plot(results: pd.DataFrame, best_scale: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        results["scale_parameter"],
        results["log_loss"],
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="#2563eb",
        label="Log Loss",
    )
    best_row = results.loc[results["log_loss"].idxmin()]
    ax.scatter(
        [best_row["scale_parameter"]],
        [best_row["log_loss"]],
        color="#dc2626",
        s=120,
        zorder=5,
        label=f"Best s={int(best_scale)}",
    )
    ax.set_title("Scale Parameter Optimization (Log Loss)")
    ax.set_xlabel("Scale Parameter (s)")
    ax.set_ylabel("Log Loss")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not DEFAULT_CSV.exists():
        raise SystemExit(f"Dataset not found: {DEFAULT_CSV}")

    df = prepare_dataset(DEFAULT_CSV)
    if df.empty:
        raise SystemExit("No matches with complete Home_Elo, Away_Elo, HFA_Value, and FTR.")

    scales = np.arange(SCALE_MIN, SCALE_MAX + 1, SCALE_STEP)
    results = evaluate_scales(df, scales)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)

    best_idx = results["log_loss"].idxmin()
    best_row = results.loc[best_idx]
    best_scale = float(best_row["scale_parameter"])

    print_summary(best_row, len(df))
    save_plot(results, best_scale, PLOT_PATH)
    print(f"\nResults saved to: {RESULTS_CSV}")
    print(f"Plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
