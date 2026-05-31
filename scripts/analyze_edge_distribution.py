#!/usr/bin/env python3
"""
Analyze EV edge distribution: Edge = model_prob * decimal_odds (home).

Applies probability smoothing (damping toward market) by default.

Usage:
    python scripts/analyze_edge_distribution.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.ml.config import default_engine_config  # noqa: E402
from app.ml.prediction_engine import PredictionEngine  # noqa: E402
from app.ml.probability_smoothing import apply_probability_smoothing  # noqa: E402
from detect_value_bets import load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "VisionGoat_Master_WithOdds.csv"
DEFAULT_HISTOGRAM = REPO_ROOT / "data" / "processed" / "edge_histogram.png"
DEFAULT_HISTOGRAM_RAW = REPO_ROOT / "data" / "processed" / "edge_histogram_raw.png"
ODDS_COLUMN = "B365H"
EDGE_THRESHOLDS = (1.00, 1.01, 1.02, 1.05)


def build_edge_frame(df: pd.DataFrame, odds_column: str) -> pd.DataFrame:
    work = df.copy()
    work[odds_column] = pd.to_numeric(work[odds_column], errors="coerce")
    work = work.dropna(subset=["model_prob", odds_column])
    work = work[work[odds_column] > 0]
    work["edge"] = work["model_prob"] * work[odds_column]
    return work


def print_distribution_stats(edges: pd.Series, title: str) -> None:
    print(f"\n--- {title} ---")
    print(f"Matches with valid edge: {len(edges)}")
    print(f"Min:    {edges.min():.4f}")
    print(f"Max:    {edges.max():.4f}")
    print(f"Mean:   {edges.mean():.4f}")
    print(f"Median: {edges.median():.4f}")

    print("\n--- Count above thresholds ---")
    for threshold in EDGE_THRESHOLDS:
        count = int((edges > threshold).sum())
        pct = (count / len(edges) * 100) if len(edges) else 0.0
        print(f"  Edge > {threshold:.2f}: {count:,} ({pct:.1f}%)")


def print_top_edges(work: pd.DataFrame, odds_column: str, top_n: int, title: str) -> None:
    display_cols = ["Date", "HomeTeam", "AwayTeam", "model_prob", odds_column, "edge"]
    if "model_prob_raw" in work.columns:
        display_cols.insert(4, "model_prob_raw")
    if "implied_prob" in work.columns:
        display_cols.insert(5, "implied_prob")
    if "FTR" in work.columns:
        display_cols.append("FTR")

    top = work.nlargest(top_n, "edge")[[c for c in display_cols if c in work.columns]]
    print(f"\n--- {title} ---\n")
    print(top.to_string(index=False))


def save_edge_histogram(
    edges: pd.Series, output_path: Path, title: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(edges, bins=50, color="#2563eb", edgecolor="white", alpha=0.85)
    ax.axvline(1.0, color="#dc2626", linestyle="--", linewidth=1.5, label="Edge = 1.00")
    for threshold in (1.02, 1.05):
        ax.axvline(
            threshold,
            color="#f59e0b",
            linestyle=":",
            linewidth=1.2,
            label=f"Edge = {threshold:.2f}",
        )
    ax.set_title(title)
    ax.set_xlabel("Edge")
    ax.set_ylabel("Number of matches")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Histogram saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model edge distribution.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Master CSV with odds (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--odds-column",
        default=ODDS_COLUMN,
        help=f"Home decimal odds column (default: {ODDS_COLUMN})",
    )
    parser.add_argument(
        "--histogram",
        type=Path,
        default=DEFAULT_HISTOGRAM,
        help="Output histogram (smoothed)",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=default_engine_config().probability_smoothing_alpha,
        help="Model weight in damping (default: 0.7)",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Skip probability smoothing (raw model only)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top-edge matches to print (default: 10)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    print(f"Loading: {args.csv}")
    df = load_matches(args.csv)
    print(f"Running PredictionEngine on {len(df)} matches...")
    engine_cfg = default_engine_config()
    engine_cfg.use_probability_smoothing = False
    predicted = PredictionEngine(engine_cfg).attach_predictions(df)

    raw_work = build_edge_frame(predicted, args.odds_column)
    print_distribution_stats(
        raw_work["edge"],
        "Edge distribution — RAW model (no smoothing)",
    )
    save_edge_histogram(
        raw_work["edge"],
        DEFAULT_HISTOGRAM_RAW,
        "Edge distribution — RAW (model_prob × B365H)",
    )

    if args.no_smoothing:
        print_top_edges(
            raw_work,
            args.odds_column,
            args.top,
            f"Top {args.top} edges (raw)",
        )
        return

    smoothed = apply_probability_smoothing(
        predicted,
        odds_column=args.odds_column,
        alpha=args.smoothing_alpha,
    )
    smooth_work = build_edge_frame(smoothed, args.odds_column)
    print(
        f"\nSmoothing: model_prob = {args.smoothing_alpha:.2f} * raw "
        f"+ {1 - args.smoothing_alpha:.2f} * implied_market_prob"
    )
    print_distribution_stats(
        smooth_work["edge"],
        "Edge distribution — SMOOTHED model",
    )
    print_top_edges(
        smooth_work,
        args.odds_column,
        args.top,
        f"Top {args.top} edges (smoothed)",
    )
    save_edge_histogram(
        smooth_work["edge"],
        args.histogram,
        f"Edge distribution — SMOOTHED (α={args.smoothing_alpha})",
    )


if __name__ == "__main__":
    main()
