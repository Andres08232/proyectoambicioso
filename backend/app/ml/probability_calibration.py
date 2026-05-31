"""Post-hoc monotonic probability calibration (separate from the Elo model)."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicProbabilityCalibrator:
    """
    Non-parametric monotonic map from raw model probability to calibrated probability.

    Preserves ranking, is piecewise-constant (no backward steps), and clips out-of-sample
    inputs via IsotonicRegression out_of_bounds='clip'.
    """

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> IsotonicProbabilityCalibrator:
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y) & np.isfinite(p)
        y = y[mask]
        p = p[mask]
        if len(y) < 2:
            self._model = None
            return self
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._model.fit(p, y)
        return self

    def calibrate_probability(
        self, p: float | np.ndarray
    ) -> float | np.ndarray:
        """Apply fitted isotonic map; returns inputs unchanged if not fitted."""
        if self._model is None:
            if np.isscalar(p):
                return float(p)
            return np.asarray(p, dtype=float)

        arr = np.atleast_1d(np.asarray(p, dtype=float))
        out = self._model.predict(arr)
        if np.isscalar(p):
            return float(out[0])
        return out


class WalkForwardIsotonicCalibrator:
    """
    Expanding-window isotonic fit for honest out-of-sample calibration.

    Refits only every ``refit_every`` new observations after ``min_samples`` to
    avoid unstable maps from refitting on every single row.
    """

    def __init__(self, min_samples: int = 150, refit_every: int = 50) -> None:
        self.min_samples = min_samples
        self.refit_every = max(1, refit_every)
        self._raw_history: list[float] = []
        self._y_history: list[int] = []
        self._calibrator = IsotonicProbabilityCalibrator()
        self._since_refit = 0

    def _maybe_refit(self) -> None:
        if len(self._raw_history) < self.min_samples:
            return
        if self._calibrator.is_fitted and self._since_refit < self.refit_every:
            return
        self._calibrator.fit(
            np.asarray(self._y_history, dtype=float),
            np.asarray(self._raw_history, dtype=float),
        )
        self._since_refit = 0

    def calibrate_probability(self, raw_p: float) -> float:
        if len(self._raw_history) < self.min_samples:
            return float(raw_p)
        self._maybe_refit()
        return float(self._calibrator.calibrate_probability(raw_p))

    def observe(self, raw_p: float, outcome: int) -> None:
        self._raw_history.append(float(raw_p))
        self._y_history.append(int(outcome))
        self._since_refit += 1


def calibrate_probability(
    p: float | np.ndarray,
    calibrator: IsotonicProbabilityCalibrator,
) -> float | np.ndarray:
    """Apply a fitted post-hoc calibrator to raw model probability(ies)."""
    return calibrator.calibrate_probability(p)
