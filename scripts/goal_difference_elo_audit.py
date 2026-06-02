#!/usr/bin/env python3
"""
Goal Difference Elo (G-Elo) research audit — dominance multipliers vs baseline Elo.

Usage:
    python scripts/goal_difference_elo_audit.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.ml.calibration_metrics import evaluate_probs  # noqa: E402
from app.ml.config import default_engine_config  # noqa: E402
from app.ml.value_bets import implied_probability  # noqa: E402
from detect_value_bets import load_matches  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "VisionGoat_Master_WithOdds.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "goal_difference_elo"
ODDS_COLUMN = "B365H"
STAKE = 1.0
EV_THRESHOLD = 1.02
PROBABILITY_SCALE = 530.0
EXTREME_EDGE = 0.15
EXTREME_EV = 1.20
K_FACTOR = 40.0  # fixed (best log-loss region from K sweep; isolates G effect)


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    g_fn: Callable[[int], float]
    description: str


def g_baseline(goal_diff: int) -> float:
    """Standard Elo: no margin scaling."""
    return 1.0


def g_production_log(goal_diff: int) -> float:
    """Current engine: log(margin+1)."""
    if goal_diff <= 0:
        return 1.0
    return math.log(goal_diff + 1)


def g_method_a(goal_diff: int) -> float:
    """G = 1 + ln(goal_difference); draws -> G=1."""
    if goal_diff <= 0:
        return 1.0
    return 1.0 + math.log(goal_diff)


def g_method_b(goal_diff: int) -> float:
    """G = sqrt(goal_difference); draws -> G=1."""
    if goal_diff <= 0:
        return 1.0
    return math.sqrt(float(goal_diff))


def g_method_538(goal_diff: int) -> float:
    """FiveThirtyEight-style margin curve (goals-adapted)."""
    if goal_diff <= 0:
        return 1.0
    return math.log(goal_diff + 1) * (2.2 / (1.0 + math.exp(-0.1 * goal_diff)))


VARIANTS: list[VariantSpec] = [
    VariantSpec("baseline", "Baseline Elo (G=1)", g_baseline, "Control: no goal-difference multiplier"),
    VariantSpec(
        "production_log",
        "Production log margin",
        g_production_log,
        "G = ln(goal_diff+1) — current goal-adjusted Elo",
    ),
    VariantSpec("method_a", "Method A: 1+ln(gd)", g_method_a, "G = 1 + ln(goal_difference)"),
    VariantSpec("method_b", "Method B: sqrt(gd)", g_method_b, "G = sqrt(goal_difference)"),
    VariantSpec(
        "method_c_538",
        "Method C: 538-style",
        g_method_538,
        "G = ln(gd+1) * 2.2/(1+exp(-0.1*gd))",
    ),
]


@dataclass
class AuditResult:
    variant: str
    label: str
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
    freq_extreme_edge: float
    freq_extreme_ev: float
    rating_change_mean: float
    rating_change_std: float
    rating_change_p95: float
    clv_proxy_mean_edge: float
    clv_beat_implied_pp: float
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


def match_scores(ftr: str) -> tuple[float, float]:
    if ftr == "H":
        return 1.0, 0.0
    if ftr == "D":
        return 0.5, 0.5
    if ftr == "A":
        return 0.0, 1.0
    raise ValueError(f"Unsupported FTR: {ftr!r}")


def expected_home_win_prob(
    home_rating: float,
    away_rating: float,
    home_advantage_elo: float,
    *,
    scale: float = PROBABILITY_SCALE,
) -> float:
    adjusted_diff = (home_rating + home_advantage_elo) - away_rating
    return 1.0 / (1.0 + math.pow(10.0, -adjusted_diff / scale))


def goal_diff(home_goals: int, away_goals: int) -> int:
    return abs(int(home_goals) - int(away_goals))


class WalkForwardGElo:
    """Minimal walk-forward Elo with configurable G(goal_diff) multiplier."""

    def __init__(
        self,
        g_fn: Callable[[int], float],
        *,
        k_factor: float = K_FACTOR,
        home_advantage_elo: float,
        initial_rating: float,
    ) -> None:
        self.g_fn = g_fn
        self.k_factor = k_factor
        self.home_advantage_elo = home_advantage_elo
        self.initial_rating = initial_rating
        self._ratings: dict[str, float] = {}
        self._rating_changes: list[float] = []

    def _rating(self, team: str) -> float:
        return self._ratings.get(team, self.initial_rating)

    def predict(self, home_team: str, away_team: str) -> float:
        return expected_home_win_prob(
            self._rating(home_team),
            self._rating(away_team),
            self.home_advantage_elo,
        )

    def update(
        self,
        home_team: str,
        away_team: str,
        ftr: str,
        home_goals: int,
        away_goals: int,
    ) -> None:
        home_r = self._rating(home_team)
        away_r = self._rating(away_team)
        e_home = expected_home_win_prob(home_r, away_r, self.home_advantage_elo)
        e_away = 1.0 - e_home
        s_home, s_away = match_scores(ftr)
        g_mult = self.g_fn(goal_diff(home_goals, away_goals))

        delta_home = self.k_factor * g_mult * (s_home - e_home)
        delta_away = self.k_factor * g_mult * (s_away - e_away)
        self._ratings[home_team] = home_r + delta_home
        self._ratings[away_team] = away_r + delta_away
        self._rating_changes.append(abs(delta_home))
        self._rating_changes.append(abs(delta_away))

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = default_engine_config()
        sort_cols = ["Date", "Time"] if "Time" in df.columns else ["Date"]
        work = df.sort_values(sort_cols).reset_index(drop=True)

        probs: list[float] = []
        home_elos: list[float] = []
        away_elos: list[float] = []
        g_values: list[float] = []

        for row in work.itertuples(index=False):
            home = str(row.HomeTeam)
            away = str(row.AwayTeam)
            probs.append(self.predict(home, away))
            home_elos.append(self._rating(home))
            away_elos.append(self._rating(away))

            ftr = getattr(row, "FTR", None)
            if pd.isna(ftr):
                g_values.append(float("nan"))
                continue
            hg = int(row.FTHG) if pd.notna(row.FTHG) else 0
            ag = int(row.FTAG) if pd.notna(row.FTAG) else 0
            gd = goal_diff(hg, ag)
            g_values.append(self.g_fn(gd))
            self.update(home, away, str(ftr), hg, ag)

        out = work.copy()
        out["model_prob_raw"] = probs
        out["home_elo"] = home_elos
        out["away_elo"] = away_elos
        out["g_multiplier"] = g_values
        return out


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
    }, bets


def clv_analysis(bets: pd.DataFrame) -> tuple[float, float]:
    """
    CLV proxy (no closing odds in dataset — B365H used as reference line).

    - clv_proxy_mean_edge: mean(model_prob - implied) on bets
    - clv_beat_implied_pp: actual home win rate minus mean implied prob (pp)
    """
    if bets.empty:
        return 0.0, 0.0
    mean_edge = float(bets["edge"].mean())
    actual_rate = float((bets["FTR"] == "H").mean())
    beat_implied = (actual_rate - float(bets["implied_prob"].mean())) * 100.0
    return mean_edge, beat_implied


def evaluate_variant(spec: VariantSpec, valid: pd.DataFrame) -> AuditResult:
    league_cfg = default_engine_config().for_league("E0")
    engine = WalkForwardGElo(
        spec.g_fn,
        k_factor=K_FACTOR,
        home_advantage_elo=league_cfg.home_advantage_elo,
        initial_rating=league_cfg.initial_rating,
    )
    pred = engine.run(valid)
    work = attach_market(pred)

    y = (work["FTR"] == "H").astype(int).to_numpy()
    cal = evaluate_probs(y, work["model_prob"].to_numpy())
    bt, bets_df = simulate_bets(work)
    clv_edge, clv_beat = clv_analysis(bets_df)

    changes = np.array(engine._rating_changes, dtype=float)
    return AuditResult(
        variant=spec.key,
        label=spec.label,
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
        freq_extreme_edge=float((work["edge"] > EXTREME_EDGE).mean()),
        freq_extreme_ev=float((work["ev_multiplier"] > EXTREME_EV).mean()),
        rating_change_mean=float(changes.mean()) if len(changes) else 0.0,
        rating_change_std=float(changes.std()) if len(changes) else 0.0,
        rating_change_p95=float(np.quantile(changes, 0.95)) if len(changes) else 0.0,
        clv_proxy_mean_edge=clv_edge,
        clv_beat_implied_pp=clv_beat,
        n_matches=len(work),
    )


def build_summary(results: list[AuditResult]) -> str:
    df = pd.DataFrame([asdict(r) for r in results])
    base = df[df["variant"] == "baseline"].iloc[0]
    best_roi = df.loc[df["roi_pct"].idxmax()]
    best_ll = df.loc[df["log_loss"].idxmin()]
    lowest_extreme = df.loc[df["freq_extreme_edge"].idxmin()]

    lines = [
        "VISIONGOAT GOAL DIFFERENCE ELO (G-ELO) AUDIT",
        "=" * 72,
        f"Dataset: {DEFAULT_CSV.name}",
        f"Update: R_new = R_old + K * G * (W - W_e) | K={K_FACTOR} | s={PROBABILITY_SCALE}",
        f"HFA in Elo | No xG/form/isotonic/smoothing | EV>{EV_THRESHOLD}",
        "",
        "DATA NOTE: No closing (CL) odds column — CLV uses B365H as reference-line proxy only.",
        "",
        "1. Does G-Elo outperform baseline Elo?",
    ]
    for metric, better_low in [
        ("log_loss", True),
        ("brier_score", True),
        ("ece", True),
        ("roi_pct", False),
    ]:
        best = df.loc[df[metric].idxmin() if better_low else df[metric].idxmax()]
        base_val = base[metric]
        lines.append(
            f"   Best {metric}: {best['variant']} ({best[metric]:.4f}) "
            f"vs baseline ({base_val:.4f})."
        )
    lines.extend(
        [
            "",
            "2. Does it improve ROI?",
            f"   Best ROI: {best_roi['variant']} ({best_roi['roi_pct']:+.2f}%, "
            f"profit {best_roi['profit']:+.1f}).",
            f"   Baseline ROI: {base['roi_pct']:+.2f}% (profit {base['profit']:+.1f}).",
        ]
    )
    if best_roi["roi_pct"] > base["roi_pct"] + 0.3:
        lines.append("   -> Some G formulation improves ROI vs baseline (still negative overall).")
    else:
        lines.append("   -> Goal-difference multipliers do not materially improve ROI.")
    lines.extend(
        [
            "",
            "3. Does it improve calibration?",
            f"   Best log loss: {best_ll['variant']} (LL={best_ll['log_loss']:.4f}).",
            f"   Baseline LL={base['log_loss']:.4f}, ECE={base['ece']:.4f}.",
        ]
    )
    if best_ll["variant"] != "baseline":
        lines.append("   -> Margin-aware updates can improve probabilistic fit vs G=1.")
    else:
        lines.append("   -> Baseline Elo is already best calibrated on this sample.")
    lines.extend(
        [
            "",
            "4. Does it reduce extreme false edges?",
            f"   Lowest freq(edge>{EXTREME_EDGE}): {lowest_extreme['variant']} "
            f"({lowest_extreme['freq_extreme_edge']:.1%}) vs baseline "
            f"({base['freq_extreme_edge']:.1%}).",
            f"   Lowest freq(ev>{EXTREME_EV}): "
            f"{df.loc[df['freq_extreme_ev'].idxmin(), 'variant']} "
            f"({df['freq_extreme_ev'].min():.1%}).",
        ]
    )
    lines.extend(
        [
            "",
            "5. Is match-dominance economically useful?",
        ]
    )
    if best_roi["roi_pct"] < 0 and base["roi_pct"] < 0:
        lines.append(
            "   -> NO for flat-stake home-win EV betting: all variants remain unprofitable."
        )
        if best_ll["log_loss"] < base["log_loss"] - 0.002:
            lines.append(
                "   -> PARTIAL for rating system: dominance info may help prediction (LL) "
                "without translating to ROI."
            )
    lines.extend(
        [
            "",
            "RATING STABILITY (mean |delta_rating| per team update)",
        ]
    )
    for _, row in df.iterrows():
        lines.append(
            f"   {row['variant']:<16} mean={row['rating_change_mean']:.3f}  "
            f"std={row['rating_change_std']:.3f}  p95={row['rating_change_p95']:.3f}"
        )
    lines.extend(
        [
            "",
            "CLV PROXY ON BETS (B365H reference)",
        ]
    )
    for _, row in df.iterrows():
        lines.append(
            f"   {row['variant']:<16} mean_edge={row['clv_proxy_mean_edge']:+.4f}  "
            f"beat_implied={row['clv_beat_implied_pp']:+.2f}pp"
        )
    lines.extend(["", "VARIANT TABLE", "-" * 72])
    for _, row in df.iterrows():
        lines.append(
            f"  {row['label']:<28} ROI {row['roi_pct']:+7.2f}%  LL {row['log_loss']:.4f}  "
            f"ECE {row['ece']:.4f}  extreme_edge {row['freq_extreme_edge']:.1%}"
        )
    lines.extend(["", "END OF REPORT"])
    return "\n".join(lines)


def plot_roi_comparison(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    colors = ["#666666" if v == "baseline" else "#2563eb" for v in df["variant"]]
    ax.bar(x, df["roi_pct"], color=colors)
    ax.axhline(0, color="#333", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=25, ha="right")
    ax.set_ylabel("ROI (%)")
    ax.set_title(f"ROI by G-Elo variant (EV>{EV_THRESHOLD}, K={K_FACTOR:.0f})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rating_stability(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["rating_change_mean"], 0.4, label="Mean |ΔR|", color="#2563eb")
    ax.bar(x + 0.2, df["rating_change_p95"], 0.4, label="P95 |ΔR|", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=25, ha="right")
    ax.set_ylabel("Elo points")
    ax.set_title("Rating change magnitude by variant")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_rating_change_histograms(
    valid: pd.DataFrame, path: Path
) -> None:
    """Distribution of per-update |rating change| for each variant."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes.flatten()
    league_cfg = default_engine_config().for_league("E0")

    for ax, spec in zip(axes_flat, VARIANTS):
        engine = WalkForwardGElo(
            spec.g_fn,
            k_factor=K_FACTOR,
            home_advantage_elo=league_cfg.home_advantage_elo,
            initial_rating=league_cfg.initial_rating,
        )
        engine.run(valid)
        changes = engine._rating_changes
        ax.hist(changes, bins=40, color="#2563eb", alpha=0.85, edgecolor="white")
        ax.set_title(spec.key, fontsize=10)
        ax.set_xlabel("|Δ rating|")
        ax.set_ylabel("Count")

    for ax in axes_flat[len(VARIANTS) :]:
        ax.axis("off")

    fig.suptitle("Distribution of rating changes per update", y=1.02)
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
    if "FTHG" not in valid.columns or "FTAG" not in valid.columns:
        raise SystemExit("FTHG/FTAG required for goal-difference multipliers.")

    print("=" * 72)
    print("GOAL DIFFERENCE ELO AUDIT")
    print("=" * 72)
    print(f"Matches: {len(valid)} | K={K_FACTOR} | EV>{EV_THRESHOLD}")
    print()

    results: list[AuditResult] = []
    for spec in VARIANTS:
        res = evaluate_variant(spec, valid)
        results.append(res)
        print(
            f"{spec.key:<16} ROI={res.roi_pct:+7.2f}%  LL={res.log_loss:.4f}  "
            f"ECE={res.ece:.4f}  extreme_edge={res.freq_extreme_edge:.1%}  "
            f"|dR|_mean={res.rating_change_mean:.2f}"
        )

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(OUTPUT_DIR / "goal_difference_results.csv", index=False)

    summary = build_summary(results)
    (OUTPUT_DIR / "goal_difference_summary.txt").write_text(summary, encoding="utf-8")

    plot_roi_comparison(df, OUTPUT_DIR / "roi_comparison.png")
    plot_rating_stability(df, OUTPUT_DIR / "rating_stability.png")
    save_rating_change_histograms(
        valid, OUTPUT_DIR / "rating_change_distributions.png"
    )

    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
