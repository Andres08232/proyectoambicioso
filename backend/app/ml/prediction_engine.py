"""Walk-forward Elo prediction engine for any league CSV."""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from app.ml.config import EngineConfig, LeagueConfig


class PredictionEngine:
    """
    Universal walk-forward engine: Elo ratings + optional league HFA normalization.

    Supports goal-adjusted Elo updates: margin multiplier log(goal_difference + 1).
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Clear in-memory ratings and walk-forward league history."""
        self._ratings: dict[tuple[str, str], float] = {}
        self._league_home_results: dict[str, list[int]] = defaultdict(list)

    def _team_key(self, league: str, team: str) -> tuple[str, str]:
        return str(league), str(team)

    def _get_rating(self, league: str, team: str, league_cfg: LeagueConfig) -> float:
        return self._ratings.get(
            self._team_key(league, team), league_cfg.initial_rating
        )

    def _set_rating(self, league: str, team: str, rating: float) -> None:
        self._ratings[self._team_key(league, team)] = rating

    @staticmethod
    def expected_home_win_prob(
        home_rating: float,
        away_rating: float,
        home_advantage_elo: float,
    ) -> float:
        exponent = (away_rating - home_rating - home_advantage_elo) / 400.0
        return 1.0 / (1.0 + math.pow(10.0, exponent))

    @staticmethod
    def _match_scores(ftr: str) -> tuple[float, float]:
        if ftr == "H":
            return 1.0, 0.0
        if ftr == "D":
            return 0.5, 0.5
        if ftr == "A":
            return 0.0, 1.0
        raise ValueError(f"Unsupported result code: {ftr!r}")

    @staticmethod
    def goal_margin_multiplier(home_goals: int, away_goals: int) -> float:
        """
        Scale Elo shift by winning margin: log(margin + 1).

        Draws (margin 0) use multiplier 1.0 so rating still updates on the result.
        """
        margin = abs(int(home_goals) - int(away_goals))
        if margin == 0:
            return 1.0
        return math.log(margin + 1)

    def _walk_forward_league_hfa(
        self, league: str, league_cfg: LeagueConfig
    ) -> float:
        history = self._league_home_results[league]
        if history:
            return sum(history) / len(history)
        if league_cfg.league_hfa is not None:
            return league_cfg.league_hfa
        return league_cfg.neutral_prob

    def _normalize_with_hfa(
        self,
        elo_prob: float,
        league_hfa: float,
        league_cfg: LeagueConfig,
    ) -> float:
        if not league_cfg.use_hfa_normalization:
            adjusted = elo_prob
        else:
            adjusted = league_hfa + (elo_prob - league_cfg.neutral_prob)
        return max(
            league_cfg.prob_floor,
            min(league_cfg.prob_ceiling, adjusted),
        )

    def predict_home_win(
        self, league: str, home_team: str, away_team: str
    ) -> tuple[float, float, float, float]:
        """
        Return (model_prob, elo_prob, league_hfa, home_elo) before the match is played.
        """
        league_cfg = self.config.for_league(league)
        home_elo = self._get_rating(league, home_team, league_cfg)
        away_elo = self._get_rating(league, away_team, league_cfg)
        elo_prob = self.expected_home_win_prob(
            home_elo, away_elo, league_cfg.home_advantage_elo
        )
        league_hfa = self._walk_forward_league_hfa(league, league_cfg)
        model_prob = self._normalize_with_hfa(elo_prob, league_hfa, league_cfg)
        return model_prob, elo_prob, league_hfa, home_elo

    def _update_after_match(
        self,
        league: str,
        home_team: str,
        away_team: str,
        ftr: str,
        home_goals: int | None = None,
        away_goals: int | None = None,
    ) -> None:
        league_cfg = self.config.for_league(league)
        home_elo = self._get_rating(league, home_team, league_cfg)
        away_elo = self._get_rating(league, away_team, league_cfg)

        e_home = self.expected_home_win_prob(
            home_elo, away_elo, league_cfg.home_advantage_elo
        )
        e_away = 1.0 / (
            1.0
            + math.pow(
                10.0,
                (home_elo - away_elo + league_cfg.home_advantage_elo) / 400.0,
            )
        )
        s_home, s_away = self._match_scores(ftr)
        k = league_cfg.k_factor

        margin_mult = 1.0
        if self.config.goal_adjusted_elo and home_goals is not None and away_goals is not None:
            margin_mult = self.goal_margin_multiplier(home_goals, away_goals)

        self._set_rating(
            league, home_team, home_elo + k * margin_mult * (s_home - e_home)
        )
        self._set_rating(
            league, away_team, away_elo + k * margin_mult * (s_away - e_away)
        )

        self._league_home_results[league].append(1 if ftr == "H" else 0)

    def attach_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Walk-forward: predict each row, then update Elo from that row's result.

        Required columns: league, home/away teams, date, result (per EngineConfig).
        FTHG/FTAG used when goal_adjusted_elo is enabled.
        """
        cfg = self.config
        league_col = cfg.league_column
        required = {
            league_col,
            cfg.home_team_column,
            cfg.away_team_column,
            cfg.date_column,
            cfg.result_column,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        has_goals = (
            cfg.home_goals_column in df.columns
            and cfg.away_goals_column in df.columns
        )
        if cfg.goal_adjusted_elo and not has_goals:
            print(
                "Warning: goal-adjusted Elo enabled but goal columns missing; "
                "using standard Elo updates (margin multiplier = 1.0)."
            )

        sort_cols = [cfg.date_column]
        if cfg.time_column in df.columns:
            sort_cols.append(cfg.time_column)
        sort_cols.append(league_col)

        out = df.sort_values(sort_cols).reset_index(drop=True)
        self.reset()

        model_probs: list[float] = []
        elo_probs: list[float] = []
        league_hfas: list[float] = []
        home_elos: list[float] = []
        away_elos: list[float] = []

        for row in out.itertuples(index=False):
            league = getattr(row, league_col)
            home = getattr(row, cfg.home_team_column)
            away = getattr(row, cfg.away_team_column)
            ftr = getattr(row, cfg.result_column)

            home_elo = self._get_rating(
                str(league), str(home), cfg.for_league(str(league))
            )
            away_elo = self._get_rating(
                str(league), str(away), cfg.for_league(str(league))
            )

            model_prob, elo_prob, league_hfa, _ = self.predict_home_win(
                str(league), str(home), str(away)
            )

            model_probs.append(model_prob)
            elo_probs.append(elo_prob)
            league_hfas.append(league_hfa)
            home_elos.append(home_elo)
            away_elos.append(away_elo)

            if pd.notna(ftr):
                home_goals: int | None = None
                away_goals: int | None = None
                if has_goals:
                    hg = getattr(row, cfg.home_goals_column)
                    ag = getattr(row, cfg.away_goals_column)
                    if pd.notna(hg) and pd.notna(ag):
                        home_goals = int(hg)
                        away_goals = int(ag)

                self._update_after_match(
                    str(league),
                    str(home),
                    str(away),
                    str(ftr),
                    home_goals=home_goals,
                    away_goals=away_goals,
                )

        out["model_prob"] = model_probs
        out["elo_prob"] = elo_probs
        out["league_hfa"] = league_hfas
        out["home_elo"] = home_elos
        out["away_elo"] = away_elos
        return out
