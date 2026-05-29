"""Walk-forward Elo prediction engine for any league CSV."""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from app.ml.config import EngineConfig, LeagueConfig


class PredictionEngine:
    """
    Universal walk-forward engine: Elo ratings + optional league HFA normalization.

  Processes matches chronologically. Before each match it emits model_prob from
    current Elo; after the result it updates ratings match-by-match.
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
        self, league: str, home_team: str, away_team: str, ftr: str
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

        self._set_rating(league, home_team, home_elo + k * (s_home - e_home))
        self._set_rating(league, away_team, away_elo + k * (s_away - e_away))

        self._league_home_results[league].append(1 if ftr == "H" else 0)

    def attach_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Walk-forward: predict each row, then update Elo from that row's result.

        Required columns: league, home/away teams, date, result (per EngineConfig).
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

            league_cfg = cfg.for_league(str(league))
            home_elo = self._get_rating(str(league), str(home), league_cfg)
            away_elo = self._get_rating(str(league), str(away), league_cfg)

            model_prob, elo_prob, league_hfa, _ = self.predict_home_win(
                str(league), str(home), str(away)
            )

            model_probs.append(model_prob)
            elo_probs.append(elo_prob)
            league_hfas.append(league_hfa)
            home_elos.append(home_elo)
            away_elos.append(away_elo)

            if pd.notna(ftr):
                self._update_after_match(str(league), str(home), str(away), str(ftr))

        out["model_prob"] = model_probs
        out["elo_prob"] = elo_probs
        out["league_hfa"] = league_hfas
        out["home_elo"] = home_elos
        out["away_elo"] = away_elos
        return out
