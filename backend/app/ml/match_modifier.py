"""Walk-forward form scoring for match probability adjustments."""

from __future__ import annotations

from collections import defaultdict


class MatchModifier:
    """
    Recent-form scores from points per game (W=3, D=1, L=0).

    History is walk-forward: only matches already played count toward form.
    """

    def __init__(
        self,
        window: int = 3,
        neutral_ppg: float = 1.0,
    ) -> None:
        self.window = window
        self.neutral_ppg = neutral_ppg
        self._points_history: dict[tuple[str, str], list[int]] = defaultdict(list)

    def reset(self) -> None:
        self._points_history.clear()

    @staticmethod
    def points_for_team(ftr: str, *, is_home: bool) -> int:
        if ftr == "D":
            return 1
        if ftr == "H":
            return 3 if is_home else 0
        if ftr == "A":
            return 0 if is_home else 3
        raise ValueError(f"Unsupported result code: {ftr!r}")

    def form_score(self, league: str, team: str) -> float:
        """Average points per game over the last `window` matches (or fewer)."""
        history = self._points_history[(str(league), str(team))]
        if not history:
            return self.neutral_ppg
        recent = history[-self.window :]
        return sum(recent) / len(recent)

    def probability_shift(self, home_form: float, away_form: float, shift_per_point: float) -> float:
        """Shift home-win probability by shift_per_point * (home_form - away_form)."""
        return shift_per_point * (home_form - away_form)

    def record_match(self, league: str, home_team: str, away_team: str, ftr: str) -> None:
        """Append result points for both teams after a match is played."""
        league_key = str(league)
        self._points_history[(league_key, str(home_team))].append(
            self.points_for_team(ftr, is_home=True)
        )
        self._points_history[(league_key, str(away_team))].append(
            self.points_for_team(ftr, is_home=False)
        )
