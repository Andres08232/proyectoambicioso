"""Configuration for the universal prediction engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeagueConfig:
    """Per-league parameters; override only what differs from defaults."""

    k_factor: float = 20.0
    home_advantage_elo: float = 100.0
    initial_rating: float = 1500.0
    league_hfa: float | None = None
    neutral_prob: float = 0.5
    use_hfa_normalization: bool = True
    prob_floor: float = 0.05
    prob_ceiling: float = 0.95


@dataclass
class EngineConfig:
    """Engine-wide settings with optional per-league overrides."""

    default: LeagueConfig = field(default_factory=LeagueConfig)
    leagues: dict[str, LeagueConfig] = field(default_factory=dict)
    league_column: str = "Div"
    home_team_column: str = "HomeTeam"
    away_team_column: str = "AwayTeam"
    result_column: str = "FTR"
    date_column: str = "Date"
    time_column: str = "Time"
    home_goals_column: str = "FTHG"
    away_goals_column: str = "FTAG"
    home_xg_column: str = "Home_xG"
    away_xg_column: str = "Away_xG"
    goal_adjusted_elo: bool = True
    use_form_modifier: bool = True
    form_window: int = 3
    form_shift_per_point: float = 0.05
    form_neutral_ppg: float = 1.0
    alpha: float = 1.0

    def for_league(self, league: str) -> LeagueConfig:
        return self.leagues.get(str(league), self.default)


def default_engine_config() -> EngineConfig:
    """Sensible defaults for common Football-Data league codes."""
    fallback = LeagueConfig()
    return EngineConfig(
        default=fallback,
        leagues={
            "E0": LeagueConfig(
                k_factor=17.0,
                home_advantage_elo=51.0,
                league_hfa=0.426,
            ),
            "D1": LeagueConfig(
                k_factor=22.0,
                home_advantage_elo=90.0,
                league_hfa=0.438,
            ),
            "I1": LeagueConfig(
                k_factor=18.0,
                home_advantage_elo=75.0,
                league_hfa=0.389,
            ),
            "SP1": LeagueConfig(
                k_factor=20.0,
                home_advantage_elo=85.0,
                league_hfa=0.45,
            ),
            "F1": LeagueConfig(
                k_factor=20.0,
                home_advantage_elo=80.0,
                league_hfa=0.45,
            ),
        },
        form_shift_per_point=0.02,
        alpha=1.0,
    )
