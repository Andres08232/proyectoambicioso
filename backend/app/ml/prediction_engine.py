"""Walk-forward Elo prediction engine for any league CSV."""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from app.ml.config import EngineConfig, LeagueConfig
from app.ml.match_modifier import MatchModifier


class PredictionEngine:
    """
    Universal walk-forward engine: Elo ratings + optional league HFA normalization.

    Supports goal-adjusted Elo updates and optional recent-form probability shifts.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._form_modifier = MatchModifier(
            window=config.form_window,
            neutral_ppg=config.form_neutral_ppg,
        )
        self.reset()

    def reset(self) -> None:
        """Clear in-memory ratings, form history, and walk-forward league history."""
        self._ratings: dict[tuple[str, str], float] = {}
        self._league_home_results: dict[str, list[int]] = defaultdict(list)
        self._form_modifier.reset()

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

    def _clamp_prob(self, prob: float, league_cfg: LeagueConfig) -> float:
        return max(league_cfg.prob_floor, min(league_cfg.prob_ceiling, prob))

    def _normalize_with_hfa(
        self,
        prob: float,
        league_hfa: float,
        league_cfg: LeagueConfig,
    ) -> float:
        if not league_cfg.use_hfa_normalization:
            adjusted = prob
        else:
            adjusted = league_hfa + (prob - league_cfg.neutral_prob)
        return self._clamp_prob(adjusted, league_cfg)

    def _apply_form_modifier(
        self,
        elo_prob: float,
        league: str,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float, float, float]:
        """
        Shift Elo probability using walk-forward form (last N matches PPG).

        Returns (adjusted_prob, home_form, away_form, form_shift).
        """
        cfg = self.config
        home_form = self._form_modifier.form_score(league, home_team)
        away_form = self._form_modifier.form_score(league, away_team)
        form_shift = self._form_modifier.probability_shift(
            home_form, away_form, cfg.form_shift_per_point
        )
        adjusted = elo_prob + form_shift
        league_cfg = cfg.for_league(league)
        return self._clamp_prob(adjusted, league_cfg), home_form, away_form, form_shift

    def predict_home_win(
        self,
        league: str,
        home_team: str,
        away_team: str,
    ) -> dict[str, float]:
        """
        Build pre-match probabilities: Elo -> optional form -> HFA normalization.
        """
        league_cfg = self.config.for_league(league)
        home_elo = self._get_rating(league, home_team, league_cfg)
        away_elo = self._get_rating(league, away_team, league_cfg)
        elo_prob = self.expected_home_win_prob(
            home_elo, away_elo, league_cfg.home_advantage_elo
        )

        home_form = self._form_modifier.form_score(league, home_team)
        away_form = self._form_modifier.form_score(league, away_team)
        form_shift = 0.0
        prob_after_form = elo_prob

        if self.config.use_form_modifier:
            prob_after_form, home_form, away_form, form_shift = self._apply_form_modifier(
                elo_prob, league, home_team, away_team
            )

        league_hfa = self._walk_forward_league_hfa(league, league_cfg)
        model_prob = self._normalize_with_hfa(prob_after_form, league_hfa, league_cfg)

        return {
            "model_prob": model_prob,
            "elo_prob": elo_prob,
            "prob_after_form": prob_after_form,
            "league_hfa": league_hfa,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_form": home_form,
            "away_form": away_form,
            "form_shift": form_shift,
        }

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
        self._form_modifier.record_match(league, home_team, away_team, ftr)

    def attach_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Walk-forward: predict each row, then update Elo and form from that result.
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

        columns: dict[str, list[float]] = {
            "model_prob": [],
            "elo_prob": [],
            "prob_after_form": [],
            "league_hfa": [],
            "home_elo": [],
            "away_elo": [],
            "home_form": [],
            "away_form": [],
            "form_shift": [],
        }

        for row in out.itertuples(index=False):
            league = getattr(row, league_col)
            home = getattr(row, cfg.home_team_column)
            away = getattr(row, cfg.away_team_column)
            ftr = getattr(row, cfg.result_column)

            prediction = self.predict_home_win(str(league), str(home), str(away))
            for key, values in columns.items():
                values.append(prediction[key])

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

        for key, values in columns.items():
            out[key] = values
        return out
