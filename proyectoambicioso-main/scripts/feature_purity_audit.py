#!/usr/bin/env python3
"""
Feature space purity audit: ablation of Elo/xG/form and calibration stability.

Usage:
    python scripts/feature_purity_audit.py
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

from app.ml.calibration_metrics import (  # noqa: E402
    CalibrationMetrics,
    N_CALIBRATION_BINS,
    evaluate_probs,
    reliability_bins,
)
from app.ml.config import EngineConfig, default_engine_config  # noqa: E402
from app.ml.prediction_engine import PredictionEngine  # noqa: E402
from app.ml.probability_calibration import WalkForwardIsotonicCalibrator  # noqa: E402
from detect_value_bets import load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "VisionGoat_Master_WithOdds.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "feature_purity_audit"
REPORT_PATH = REPO_ROOT / "data" / "processed" / "feature_purity_audit.txt"

HYBRID_ALPHA = 0.8
CAL_MIN_SAMPLES = 150
CAL_REFIT_EVERY = 50


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    space: str  # A, B, C
    description: str
    applies_to: str
    risk: str


FEATURE_REGISTRY: list[FeatureSpec] = [
    FeatureSpec(
        "traditional_elo_rating",
        "A",
        "Walk-forward Elo from match results (FTR) with optional goal-margin K",
        "Rating update + win probability input",
        "Low — core strength signal",
    ),
    FeatureSpec(
        "xg_elo_rating",
        "A",
        "Walk-forward Elo from Home_xG / Away_xG ordering",
        "Rating update + blended win probability when alpha < 1",
        "Low — core strength signal if kept in Elo blend",
    ),
    FeatureSpec(
        "elo_blend_alpha",
        "A",
        "Linear blend of traditional and xG Elo in rating space before logistic",
        "pre-logistic rating",
        "Low — valid Elo-space composition",
    ),
    FeatureSpec(
        "home_advantage_elo",
        "A",
        "HFA added to home rating in Elo space; P = logistic((H+HFA-A)/530)",
        "pre-logistic rating",
        "Low — structurally correct HFA",
    ),
    FeatureSpec(
        "probability_scale",
        "A",
        "Logistic scale s=530 (10-base), single probability transform",
        "logistic output",
        "Low — calibrated scale parameter",
    ),
    FeatureSpec(
        "goal_margin_multiplier",
        "A",
        "log(margin+1) scales K on Elo updates only",
        "Rating update magnitude",
        "Low — does not enter probability directly",
    ),
    FeatureSpec(
        "form_ppg_window",
        "C",
        "Rolling points-per-game over last N matches (momentum / temporal)",
        "probability_shift after logistic",
        "HIGH — probability-space leakage",
    ),
    FeatureSpec(
        "form_probability_shift",
        "C",
        "P_final = P_elo + shift_per_point * (home_form - away_form)",
        "post-logistic additive offset",
        "HIGH — breaks logit-space purity; non-monotonic w.r.t. Elo",
    ),
    FeatureSpec(
        "walk_forward_league_hfa_rate",
        "C",
        "Walk-forward league home-win rate (clamped)",
        "Diagnostic only in current engine (not applied to P)",
        "Medium if re-enabled in probability — currently inert",
    ),
    FeatureSpec(
        "probability_smoothing",
        "C",
        "0.7*model + 0.3*implied (detect_value_bets layer)",
        "post-model probability blend",
        "HIGH — mixes market into model; disable for purity tests",
    ),
    FeatureSpec(
        "isotonic_calibration",
        "C",
        "Post-hoc monotonic map on P",
        "separate calibration layer",
        "Medium — valid only if base features are pure; cannot fix bad features",
    ),
]


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


def purity_config(
    *,
    alpha: float,
    use_form: bool,
) -> EngineConfig:
    cfg = deepcopy(default_engine_config())
    cfg.use_probability_smoothing = False
    cfg.use_post_hoc_calibration = False
    cfg.probability_scale = 530.0
    cfg.alpha = alpha
    cfg.use_form_modifier = use_form
    return cfg


def walk_forward_isotonic(raw: np.ndarray, y: np.ndarray) -> np.ndarray:
    cal = WalkForwardIsotonicCalibrator(
        min_samples=CAL_MIN_SAMPLES, refit_every=CAL_REFIT_EVERY
    )
    out = np.empty_like(raw, dtype=float)
    for i, p in enumerate(raw):
        out[i] = cal.calibrate_probability(float(p))
        cal.observe(float(p), int(y[i]))
    return out


def midrange_ece(y: np.ndarray, p: np.ndarray, lo: float = 0.35, hi: float = 0.65) -> float:
    mask = (p >= lo) & (p <= hi)
    if not mask.any():
        return float("nan")
    return expected_calibration_error(y[mask], p[mask], n_bins=5)


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
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def high_bin_curve_distortion(y: np.ndarray, pre: np.ndarray, post: np.ndarray) -> float:
    """Mean |gap| change in bins with center >= 0.65 (upper / tail region)."""
    rows_pre = reliability_bins(y, pre, N_CALIBRATION_BINS)
    rows_post = reliability_bins(y, post, N_CALIBRATION_BINS)
    deltas = []
    for rp, rpo in zip(rows_pre, rows_post):
        if rp["count"] == 0 or rpo["count"] == 0:
            continue
        if rp["bin_center"] < 0.65:
            continue
        gap_pre = abs(rp["mean_predicted"] - rp["empirical_rate"])
        gap_post = abs(rpo["mean_predicted"] - rpo["empirical_rate"])
        deltas.append(gap_post - gap_pre)
    return float(np.mean(deltas)) if deltas else float("nan")


def tail_bias(y: np.ndarray, p: np.ndarray, top_frac: float = 0.10) -> float:
    cutoff = np.quantile(p, 1.0 - top_frac)
    tail = p >= cutoff
    if not tail.any():
        return float("nan")
    return float(np.mean(p[tail] - y[tail]))


def plot_reliability(
    y: np.ndarray, p: np.ndarray, title: str, path: Path
) -> None:
    rows = reliability_bins(y, p, N_CALIBRATION_BINS)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="#666", label="Perfect")
    valid = [r for r in rows if r["count"] > 0]
    if valid:
        ax.plot(
            [r["mean_predicted"] for r in valid],
            [r["empirical_rate"] for r in valid],
            "o-",
            color="#2563eb",
            label="Model",
        )
    ax.set_title(title)
    ax.set_xlabel("Mean predicted P(home win)")
    ax.set_ylabel("Empirical home win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation_comparison(
    metrics: dict[str, CalibrationMetrics], path: Path
) -> None:
    names = list(metrics.keys())
    tail = [metrics[n].tail_error for n in names]
    ece = [metrics[n].ece for n in names]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, tail, w, label="Tail error (top 10%)", color="#dc2626")
    ax.bar(x + w / 2, ece, w, label="ECE", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Ablation: tail error vs global ECE")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_metrics(m: CalibrationMetrics) -> str:
    return (
        f"LL={m.log_loss:.4f}  Brier={m.brier_score:.4f}  "
        f"ECE={m.ece:.4f}  TailErr={m.tail_error:.4f}  N={m.n_samples}"
    )


def build_report(
    *,
    metrics: dict[str, CalibrationMetrics],
    mid_ece: dict[str, float],
    cal_rows: dict[str, tuple[CalibrationMetrics, CalibrationMetrics, float, float]],
    form_shift_stats: dict[str, float],
) -> str:
    a, b, c = metrics["Model_A"], metrics["Model_B"], metrics["Model_C"]
    lines: list[str] = []
    lines.append("VISIONGOAT FEATURE SPACE PURITY AUDIT")
    lines.append("=" * 72)
    lines.append(f"Dataset: {DEFAULT_CSV.name}")
    lines.append(f"Scale s=530 | HFA in Elo space | No smoothing in ablation")
    lines.append("")

    lines.append("TASK 1 — FEATURE SPACE CLASSIFICATION")
    lines.append("-" * 72)
    lines.append("A) Elo-space (allowed in rating / single logistic)")
    lines.append("B) Logit-space additive (allowed ONLY before sigmoid — none implemented)")
    lines.append("C) Invalid / contaminating (probability-space or post-hoc misuse)")
    lines.append("")
    for spec in FEATURE_REGISTRY:
        lines.append(f"  [{spec.space}] {spec.name}")
        lines.append(f"      {spec.description}")
        lines.append(f"      Used for: {spec.applies_to}")
        lines.append(f"      Risk: {spec.risk}")
        lines.append("")
    lines.append(
        "  NOTE: No dedicated streak columns exist; form_ppg_window is the momentum proxy."
    )
    lines.append(
        "  Rolling league_hfa is computed but NOT applied to P in the current engine."
    )
    lines.append("")

    lines.append("TASK 2 — ABLATION (raw, no calibration)")
    lines.append("-" * 72)
    lines.append("  Model A: Elo + xG blend (alpha=0.8), no form")
    lines.append(f"    {format_metrics(a)}")
    lines.append(f"    Mid-range ECE [0.35,0.65]: {mid_ece['Model_A']:.4f}")
    lines.append("")
    lines.append("  Model B: Elo + xG path + form (alpha=1.0 production default, form ON)")
    lines.append(f"    {format_metrics(b)}")
    lines.append(f"    Mid-range ECE: {mid_ece['Model_B']:.4f}")
    lines.append("")
    lines.append("  Model C: Elo only (alpha=1.0), no form")
    lines.append(f"    {format_metrics(c)}")
    lines.append(f"    Mid-range ECE: {mid_ece['Model_C']:.4f}")
    lines.append("")
    lines.append("  Feature deltas:")
    lines.append("    Form effect (B vs C, same alpha=1.0):")
    lines.append(
        f"      tail error: {b.tail_error - c.tail_error:+.4f}  "
        f"log loss: {b.log_loss - c.log_loss:+.4f}  ECE: {b.ece - c.ece:+.4f}"
    )
    lines.append("    xG blend effect (A vs C, form off):")
    lines.append(
        f"      tail error: {a.tail_error - c.tail_error:+.4f}  "
        f"log loss: {a.log_loss - c.log_loss:+.4f}  ECE: {a.ece - c.ece:+.4f}"
    )
    lines.append("    Form on dual-Elo path (B vs A, alpha differs — indicative only):")
    lines.append(
        f"      tail error: {b.tail_error - a.tail_error:+.4f}  "
        f"log loss: {b.log_loss - a.log_loss:+.4f}"
    )
    if form_shift_stats:
        lines.append(
            f"    |form_shift| mean: {form_shift_stats['mean_abs']:.4f}  "
            f"max: {form_shift_stats['max_abs']:.4f}  "
            f"frac |shift|>0.05: {form_shift_stats['frac_large']:.1%}"
        )
    lines.append("")

    lines.append("TASK 3 — CALIBRATION STABILITY (isotonic on A and C only)")
    lines.append("-" * 72)
    for label in ("Model_A", "Model_C"):
        pre, post, dll, dtail = cal_rows[label]
        lines.append(f"  {label}:")
        lines.append(f"    Pre:  {format_metrics(pre)}")
        lines.append(f"    Post: {format_metrics(post)}")
        lines.append(f"    Delta LL: {dll:+.4f}  Delta tail error: {dtail:+.4f}")
        lines.append(
            f"    Upper-bin curve distortion (post-pre |gap|): "
            f"{cal_rows[label + '_distortion']:.4f}"
        )
        lines.append("")
    lines.append("  Model B: isotonic NOT run (per audit protocol).")
    lines.append(
        "  Prior evidence: form pushes P outside logistic manifold; isotonic on "
        "contaminated features degrades log loss while patching ECE."
    )
    lines.append("")

    lines.append("TASK 4 — FEATURE IMPACT CONCLUSIONS")
    lines.append("-" * 72)

    form_worsens_tail = b.tail_error > c.tail_error + 0.005
    form_worsens_ll = b.log_loss > c.log_loss + 0.005
    elo_more_stable = c.tail_error <= a.tail_error and c.log_loss <= a.log_loss + 0.01

    lines.append("1. Tail overconfidence drivers:")
    if form_worsens_tail:
        lines.append("   -> FORM MODIFIER is the primary tail-error amplifier in this ablation.")
    else:
        lines.append("   -> Form does not dominate tail error vs Model A; check xG blend / base Elo.")
    if a.tail_error > c.tail_error + 0.005:
        lines.append("   -> xG blend (A vs C) increases tail error slightly.")
    lines.append("")

    lines.append("2. Form vs probabilistic calibration:")
    if form_worsens_ll or form_worsens_tail:
        lines.append(
            "   -> Form is INCOMPATIBLE with a single logistic + monotonic calibration stack."
        )
        lines.append(
            "      It applies a linear shift in probability space after the sigmoid."
        )
    else:
        lines.append("   -> Form impact on tail/LL is modest in this sample; still invalid space.")
    lines.append("")

    lines.append("3. Elo-only stability:")
    if elo_more_stable:
        lines.append("   -> Model C (Elo-only) is more stable than feature-augmented A on LL/tail.")
    else:
        lines.append("   -> Hybrid Model A matches or beats Elo-only C on LL/tail in this run.")
    lines.append("")

    lines.append("4. Calibration layer role:")
    for label in ("Model_A", "Model_C"):
        dll = cal_rows[label][2]
        if dll > 0.05:
            lines.append(
                f"   -> Isotonic on {label} INCREASES log loss ({dll:+.4f}) — "
                "calibration is compensating for structural misfit, not refining signal."
            )
        else:
            lines.append(
                f"   -> Isotonic on {label} does not harm log loss materially."
            )
    lines.append("")
    lines.append(
        "RECOMMENDATION: Remove or relocate form into Elo-space before enabling "
        "post-hoc calibration. Do not stack isotonic on Model B until probability "
        "space is clean."
    )
    lines.append("")
    lines.append("END OF REPORT")
    return "\n".join(lines)


def main() -> None:
    if not DEFAULT_CSV.exists():
        raise SystemExit(f"Dataset not found: {DEFAULT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_matches(DEFAULT_CSV)
    raw_df = _derive_ftr(raw_df)
    valid = raw_df[raw_df["FTR"].isin(["H", "D", "A"])].copy()
    y = (valid["FTR"] == "H").astype(int).to_numpy()

    configs = {
        "Model_A": purity_config(alpha=HYBRID_ALPHA, use_form=False),
        "Model_B": purity_config(alpha=1.0, use_form=True),
        "Model_C": purity_config(alpha=1.0, use_form=False),
    }

    probs: dict[str, np.ndarray] = {}
    metrics: dict[str, CalibrationMetrics] = {}
    mid_ece: dict[str, float] = {}

    print("=" * 72)
    print("FEATURE PURITY AUDIT")
    print("=" * 72)
    print(f"Matches: {len(valid)}\n")

    pred_b = None
    for name, cfg in configs.items():
        pred = PredictionEngine(cfg).attach_predictions(valid)
        p = pred["model_prob_raw"].to_numpy(dtype=float)
        probs[name] = p
        metrics[name] = evaluate_probs(y, p)
        mid_ece[name] = midrange_ece(y, p)
        print(f"{name}: {format_metrics(metrics[name])}")
        if name == "Model_B":
            pred_b = pred

    plot_ablation_comparison(metrics, OUTPUT_DIR / "ablation_tail_ece.png")
    for name, p in probs.items():
        plot_reliability(
            y,
            p,
            f"{name} — raw reliability",
            OUTPUT_DIR / f"reliability_{name.lower()}.png",
        )

    form_shift_stats: dict[str, float] = {}
    if pred_b is not None and "form_shift" in pred_b.columns:
        shifts = pred_b["form_shift"].to_numpy(dtype=float)
        form_shift_stats = {
            "mean_abs": float(np.mean(np.abs(shifts))),
            "max_abs": float(np.max(np.abs(shifts))),
            "frac_large": float(np.mean(np.abs(shifts) > 0.05)),
        }

    print("\n--- Calibration stability (A and C only) ---")
    cal_rows: dict[str, tuple] = {}
    for name in ("Model_A", "Model_C"):
        raw = probs[name]
        cal = walk_forward_isotonic(raw, y)
        m_pre = metrics[name]
        m_post = evaluate_probs(y, cal)
        dll = m_post.log_loss - m_pre.log_loss
        dtail = m_post.tail_error - m_pre.tail_error
        dist = high_bin_curve_distortion(y, raw, cal)
        cal_rows[name] = (m_pre, m_post, dll, dtail)
        cal_rows[name + "_distortion"] = dist
        print(f"{name}  LL delta {dll:+.4f}  tail delta {dtail:+.4f}  distortion {dist:+.4f}")
        plot_reliability(
            y,
            cal,
            f"{name} — post-isotonic",
            OUTPUT_DIR / f"reliability_{name.lower()}_isotonic.png",
        )
        fig, ax = plt.subplots(figsize=(7, 7))
        for p_arr, label, color in (
            (raw, "Pre", "#2563eb"),
            (cal, "Post isotonic", "#dc2626"),
        ):
            rows = reliability_bins(y, p_arr, N_CALIBRATION_BINS)
            vv = [r for r in rows if r["count"] > 0]
            if vv:
                ax.plot(
                    [r["mean_predicted"] for r in vv],
                    [r["empirical_rate"] for r in vv],
                    "o-",
                    label=label,
                    color=color,
                )
        ax.plot([0, 1], [0, 1], "--", color="#666")
        ax.set_title(f"{name} — pre vs post isotonic")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"calibration_stability_{name.lower()}.png", dpi=150)
        plt.close(fig)

    # Tail amplification on same top-10% raw fixtures
    for name in ("Model_A", "Model_C"):
        raw = probs[name]
        cal = walk_forward_isotonic(raw, y)
        cutoff = np.quantile(raw, 0.9)
        tail = raw >= cutoff
        amp = float(np.mean(np.abs(cal[tail] - y[tail])) - np.mean(np.abs(raw[tail] - y[tail])))
        cal_rows[name + "_tail_amp"] = amp

    report = build_report(
        metrics=metrics,
        mid_ece=mid_ece,
        cal_rows=cal_rows,
        form_shift_stats=form_shift_stats,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "model": k,
                **metrics[k].__dict__,
                "midrange_ece": mid_ece[k],
            }
            for k in metrics
        ]
    )
    summary.to_csv(OUTPUT_DIR / "ablation_metrics.csv", index=False)

    print(f"\nReport: {REPORT_PATH}")
    print(f"Plots:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
