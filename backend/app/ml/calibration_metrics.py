"""Shared calibration diagnostics (log loss, ECE, tail error)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

PROB_EPS = 1e-15
N_CALIBRATION_BINS = 10
HIGH_CONF_THRESHOLD = 0.7


@dataclass
class CalibrationMetrics:
    log_loss: float
    brier_score: float
    ece: float
    tail_error: float
    n_samples: int


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p, PROB_EPS, 1.0 - PROB_EPS)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_CALIBRATION_BINS
) -> float:
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


def tail_error(
    y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.10
) -> float:
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
    p = _clip_probs(y_prob[valid].astype(float))
    return CalibrationMetrics(
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier_score=float(brier_score_loss(y, p)),
        ece=expected_calibration_error(y, p),
        tail_error=tail_error(y, p),
        n_samples=int(len(y)),
    )


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_CALIBRATION_BINS
) -> list[dict[str, float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            rows.append(
                {
                    "bin_center": (bins[b] + bins[b + 1]) / 2,
                    "mean_predicted": float("nan"),
                    "empirical_rate": float("nan"),
                    "count": 0.0,
                }
            )
            continue
        rows.append(
            {
                "bin_center": (bins[b] + bins[b + 1]) / 2,
                "mean_predicted": float(y_prob[mask].mean()),
                "empirical_rate": float(y_true[mask].mean()),
                "count": float(mask.sum()),
            }
        )
    return rows
