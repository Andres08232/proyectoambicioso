#!/usr/bin/env python3
"""
Refined calibration check: raw Elo model (s=530, HFA in Elo space) vs post-hoc isotonic.

Usage:
    python scripts/check_calibration_refined.py
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

from app.ml.calibration_metrics import (  # noqa: E402
    CalibrationMetrics,
    evaluate_probs,
    reliability_bins,
)
from app.ml.config import EngineConfig, LeagueConfig, default_engine_config  # noqa: E402
from app.ml.prediction_engine import PredictionEngine  # noqa: E402
from app.ml.probability_calibration import WalkForwardIsotonicCalibrator  # noqa: E402
from detect_value_bets import load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "VisionGoat_Master_WithOdds.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "calibration_refined"


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


def audit_engine_config(
    *,
    home_advantage_elo: float | None = None,
    alpha: float | None = None,
    use_form: bool = False,
) -> EngineConfig:
    cfg = deepcopy(default_engine_config())
    cfg.use_probability_smoothing = False
    cfg.use_post_hoc_calibration = False
    cfg.probability_scale = 530.0
    cfg.use_form_modifier = use_form
    if alpha is not None:
        cfg.alpha = alpha
    if home_advantage_elo is not None:
        e0 = cfg.for_league("E0")
        cfg.leagues["E0"] = LeagueConfig(
            k_factor=e0.k_factor,
            home_advantage_elo=home_advantage_elo,
            initial_rating=e0.initial_rating,
            league_hfa=e0.league_hfa,
            neutral_prob=e0.neutral_prob,
            prob_floor=e0.prob_floor,
            prob_ceiling=e0.prob_ceiling,
        )
    return cfg


def walk_forward_calibrated(
    raw_probs: np.ndarray,
    y: np.ndarray,
    min_samples: int = 150,
    refit_every: int = 50,
) -> np.ndarray:
    calibrator = WalkForwardIsotonicCalibrator(
        min_samples=min_samples, refit_every=refit_every
    )
    out = np.empty_like(raw_probs, dtype=float)
    for i, raw in enumerate(raw_probs):
        out[i] = calibrator.calibrate_probability(float(raw))
        calibrator.observe(float(raw), int(y[i]))
    return out


def run_predictions(cfg: EngineConfig, df: pd.DataFrame) -> pd.DataFrame:
    return PredictionEngine(cfg).attach_predictions(df)


def print_metrics(label: str, m: CalibrationMetrics) -> None:
    print(
        f"  {label:<22} LL={m.log_loss:.4f}  Brier={m.brier_score:.4f}  "
        f"ECE={m.ece:.4f}  TailErr={m.tail_error:.4f}  N={m.n_samples}"
    )


def plot_reliability_comparison(
    y: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, probs, title in zip(
        axes,
        [raw, calibrated],
        ["Raw model (s=530, HFA in Elo)", "Post-hoc isotonic"],
    ):
        rows = reliability_bins(y, probs)
        centers = [r["bin_center"] for r in rows if r["count"] > 0]
        pred = [r["mean_predicted"] for r in rows if r["count"] > 0]
        emp = [r["empirical_rate"] for r in rows if r["count"] > 0]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        if centers:
            ax.plot(pred, emp, "o-", label="Bins")
        ax.set_title(title)
        ax.set_xlabel("Mean predicted P(home win)")
        ax.set_ylabel("Empirical home win rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    if not DEFAULT_CSV.exists():
        raise SystemExit(f"Dataset not found: {DEFAULT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_matches(DEFAULT_CSV)
    raw_df = _derive_ftr(raw_df)
    valid = raw_df[raw_df["FTR"].isin(["H", "D", "A"])].copy()
    y = (valid["FTR"] == "H").astype(int).to_numpy()

    baseline_hfa = default_engine_config().for_league("E0").home_advantage_elo

    print("=" * 72)
    print("REFINED CALIBRATION CHECK (HFA in Elo space, s=530)")
    print("=" * 72)
    print(f"Matches: {len(valid)}")
    print(f"E0 home_advantage_elo (HFA in Elo): {baseline_hfa:.1f}")
    print()

    # --- Main: baseline Elo (alpha=1) ---
    cfg = audit_engine_config()
    pred = run_predictions(cfg, valid)
    raw_probs = pred["model_prob_raw"].to_numpy(dtype=float)
    cal_probs = walk_forward_calibrated(raw_probs, y)

    m_raw = evaluate_probs(y, raw_probs)
    m_cal = evaluate_probs(y, cal_probs)

    print("1. RAW vs POST-HOC ISOTONIC (Elo alpha=1.0, form OFF, HFA in Elo)")
    print_metrics("Raw", m_raw)
    print_metrics("Isotonic", m_cal)
    print(
        f"     Delta LL={m_cal.log_loss - m_raw.log_loss:+.4f}  "
        f"Delta ECE={m_cal.ece - m_raw.ece:+.4f}  "
        f"Delta TailErr={m_cal.tail_error - m_raw.tail_error:+.4f}"
    )
    print()

    plot_reliability_comparison(
        y, raw_probs, cal_probs, OUTPUT_DIR / "reliability_pre_post.png"
    )

    # --- HFA isolation (Elo-space only) ---
    cfg_no_hfa = audit_engine_config(home_advantage_elo=0.0)
    pred_no_hfa = run_predictions(cfg_no_hfa, valid)
    raw_no_hfa = pred_no_hfa["model_prob_raw"].to_numpy(dtype=float)
    cal_no_hfa = walk_forward_calibrated(raw_no_hfa, y)

    m_raw_no = evaluate_probs(y, raw_no_hfa)
    m_cal_no = evaluate_probs(y, cal_no_hfa)

    print("2. HFA ISOLATION (home_advantage_elo = 0 vs baseline)")
    print_metrics(f"Raw HFA={baseline_hfa:.0f}", m_raw)
    print_metrics("Raw HFA=0", m_raw_no)
    print(
        f"     Tail error: HFA={baseline_hfa:.0f} -> {m_raw.tail_error:.4f}, "
        f"HFA=0 -> {m_raw_no.tail_error:.4f}"
    )
    print(
        "     -> HFA in Elo space "
        + (
            "contributes to tail miscalibration."
            if m_raw.tail_error > m_raw_no.tail_error + 0.01
            else "is not the dominant tail driver (vs removing HFA)."
        )
    )
    print()

    # --- xG after calibration ---
    cfg_xg = audit_engine_config(alpha=0.8)
    pred_xg = run_predictions(cfg_xg, valid)
    raw_xg = pred_xg["model_prob_raw"].to_numpy(dtype=float)
    cal_xg = walk_forward_calibrated(raw_xg, y)

    m_raw_xg = evaluate_probs(y, raw_xg)
    m_cal_xg = evaluate_probs(y, cal_xg)

    print("3. xG PREDICTIVE VALUE (alpha=0.8: 80% Elo / 20% xG)")
    print_metrics("Raw hybrid", m_raw_xg)
    print_metrics("Calibrated hybrid", m_cal_xg)
    print(
        f"     After calibration, hybrid vs Elo-only LL: "
        f"{m_cal_xg.log_loss - m_cal.log_loss:+.4f}"
    )
    print(
        "     -> xG adds predictive value after calibration."
        if m_cal_xg.log_loss < m_cal.log_loss - 0.0005
        else "     -> xG does not improve log loss after calibration."
    )
    print()

    # --- Tail bucket (top 10% confidence, same matches for raw vs calibrated) ---
    cutoff = np.quantile(raw_probs, 0.9)
    tail = raw_probs >= cutoff
    if tail.any():
        tail_bias_raw = float(np.mean(raw_probs[tail] - y[tail]))
        tail_bias_cal = float(np.mean(cal_probs[tail] - y[tail]))
        print("4. TAIL BUCKET (top 10% raw confidence, same fixtures)")
        print(f"     Raw mean bias (pred - outcome):        {tail_bias_raw:+.4f}")
        print(f"     Calibrated on same fixtures:           {tail_bias_cal:+.4f}")
        print(
            "     -> Isotonic "
            + (
                "reduces tail overconfidence."
                if abs(tail_bias_cal) < abs(tail_bias_raw)
                else "does not reduce tail bias on these fixtures."
            )
        )
    print()

    cfg_form = audit_engine_config(use_form=True)
    pred_form = run_predictions(cfg_form, valid)
    raw_form = pred_form["model_prob_raw"].to_numpy(dtype=float)
    m_form = evaluate_probs(y, raw_form)
    print("5. PRODUCTION-LIKE RAW (form modifier ON)")
    print_metrics("Raw + form", m_form)
    print()

    summary_path = OUTPUT_DIR / "refined_calibration_summary.txt"
    lines = [
        "REFINED CALIBRATION SUMMARY",
        "=" * 50,
        f"N matches: {len(valid)}",
        f"Scale s: 530 (HFA only in Elo space)",
        "",
        "Raw model:",
        f"  Log Loss={m_raw.log_loss:.4f}  Brier={m_raw.brier_score:.4f}",
        f"  ECE={m_raw.ece:.4f}  TailErr={m_raw.tail_error:.4f}",
        "",
        "Post-hoc isotonic:",
        f"  Log Loss={m_cal.log_loss:.4f}  Brier={m_cal.brier_score:.4f}",
        f"  ECE={m_cal.ece:.4f}  TailErr={m_cal.tail_error:.4f}",
        "",
        "HFA=0 raw tail error: "
        f"{m_raw_no.tail_error:.4f} vs HFA={baseline_hfa:.0f}: {m_raw.tail_error:.4f}",
        "",
        "Hybrid calibrated LL vs Elo-only: "
        f"{m_cal_xg.log_loss:.4f} vs {m_cal.log_loss:.4f}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Artifacts: {OUTPUT_DIR}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()
