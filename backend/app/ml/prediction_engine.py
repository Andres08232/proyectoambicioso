"""Walk-forward Elo prediction engine for any league CSV."""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from app.ml.config import EngineConfig, LeagueConfig
from app.ml.match_modifier import MatchModifier
from app.ml.probability_calibration import WalkForwardIsotonicCalibrator


class PredictionEngine:
    """
    Walk-forward dual Elo: traditional (goals) + xG-based ratings blended by alpha.

    final_rating = (alpha * traditional_elo) + ((1 - alpha) * xg_elo)
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._form_modifier = MatchModifier(
            window=config.form_window,
            neutral_ppg=config.form_neutral_ppg,
        )
        self.reset()

    def reset(self) -> None:
        """Clear traditional Elo, xG Elo, form, and league HFA history."""
        self._elo_ratings: dict[tuple[str, str], float] = {}
        self._elo_xg_ratings: dict[tuple[str, str], float] = {}
        self._league_home_results: dict[str, list[int]] = defaultdict(list)
        self._form_modifier.reset()

    def _team_key(self, league: str, team: str) -> tuple[str, str]:
        return str(league), str(team)

    def _get_traditional_rating(
        self, league: str, team: str, league_cfg: LeagueConfig
    ) -> float:
        return self._elo_ratings.get(
            self._team_key(league, team), league_cfg.initial_rating
        )

    def _get_xg_rating(
        self, league: str, team: str, league_cfg: LeagueConfig
    ) -> float:
        return self._elo_xg_ratings.get(
            self._team_key(league, team), league_cfg.initial_rating
        )

    def _set_traditional_rating(self, league: str, team: str, rating: float) -> None:
        self._elo_ratings[self._team_key(league, team)] = rating

    def _set_xg_rating(self, league: str, team: str, rating: float) -> None:
        self._elo_xg_ratings[self._team_key(league, team)] = rating

    def _blended_rating(
        self, league: str, team: str, league_cfg: LeagueConfig
    ) -> float:
        alpha = self.config.alpha
        trad = self._get_traditional_rating(league, team, league_cfg)
        if alpha >= 1.0:
            return trad
        xg = self._get_xg_rating(league, team, league_cfg)
        if alpha <= 0.0:
            return xg
        return (alpha * trad) + ((1.0 - alpha) * xg)

    @staticmethod
    def expected_home_win_prob(
        home_rating: float,
        away_rating: float,
        home_advantage_elo: float,
        *,
        scale: float = 530.0,
    ) -> float:
        """
        Logistic win probability with HFA applied only in Elo space.

        adjusted_diff = (home_rating + home_advantage_elo) - away_rating
        P_home = 1 / (1 + 10 ** (-adjusted_diff / scale))
        """
        adjusted_diff = (home_rating + home_advantage_elo) - away_rating
        return 1.0 / (1.0 + math.pow(10.0, -adjusted_diff / scale))

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
    def _xg_match_scores(home_xg: float, away_xg: float) -> tuple[float, float]:
        if home_xg > away_xg:
            return 1.0, 0.0
        if away_xg > home_xg:
            return 0.0, 1.0
        return 0.5, 0.5

    @staticmethod
    def goal_margin_multiplier(home_goals: int, away_goals: int) -> float:
        margin = abs(int(home_goals) - int(away_goals))
        if margin == 0:
            return 1.0
        return math.log(margin + 1)

    @staticmethod
    def xg_margin_multiplier(home_xg: float, away_xg: float) -> float:
        margin = abs(float(home_xg) - float(away_xg))
        if margin < 1e-9:
            return 1.0
        return math.log(margin + 1)

    def _walk_forward_league_hfa(
        self, league: str, league_cfg: LeagueConfig
    ) -> float:
        """
        Walk-forward home-win rate, blended toward historical league HFA, then clamped.

        final_hfa = (w * calculated_hfa) + ((1 - w) * historical_average)
        Returns value in [hfa_floor, hfa_ceiling] to avoid early-season spikes.
        """
        cfg = self.config
        history = self._league_home_results[league]

        if history:
            calculated_hfa = sum(history) / len(history)
        elif league_cfg.league_hfa is not None:
            calculated_hfa = league_cfg.league_hfa
        else:
            calculated_hfa = league_cfg.neutral_prob

        historical_avg = (
            league_cfg.league_hfa
            if league_cfg.league_hfa is not None
            else league_cfg.neutral_prob
        )
        weight = cfg.hfa_calculated_weight
        blended = (weight * calculated_hfa) + ((1.0 - weight) * historical_avg)

        return max(cfg.hfa_floor, min(cfg.hfa_ceiling, blended))

    def _apply_form_modifier(
        self,
        elo_prob: float,
        league: str,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float, float, float]:
        cfg = self.config
        home_form = self._form_modifier.form_score(league, home_team)
        away_form = self._form_modifier.form_score(league, away_team)
        form_shift = self._form_modifier.probability_shift(
            home_form, away_form, cfg.form_shift_per_point
        )
        adjusted = elo_prob + form_shift
        return adjusted, home_form, away_form, form_shift

    def _update_elo_pair(
        self,
        league: str,
        home_team: str,
        away_team: str,
        *,
        home_rating: float,
        away_rating: float,
        s_home: float,
        s_away: float,
        margin_mult: float,
        set_home,
        set_away,
    ) -> None:
        league_cfg = self.config.for_league(league)
        scale = self.config.probability_scale
        e_home = self.expected_home_win_prob(
            home_rating, away_rating, league_cfg.home_advantage_elo, scale=scale
        )
        e_away = 1.0 - e_home
        k = league_cfg.k_factor
        set_home(home_rating + k * margin_mult * (s_home - e_home))
        set_away(away_rating + k * margin_mult * (s_away - e_away))

    def predict_home_win(
        self,
        league: str,
        home_team: str,
        away_team: str,
    ) -> dict[str, float]:
        league_cfg = self.config.for_league(league)
        home_trad = self._get_traditional_rating(league, home_team, league_cfg)
        away_trad = self._get_traditional_rating(league, away_team, league_cfg)
        home_xg_elo = self._get_xg_rating(league, home_team, league_cfg)
        away_xg_elo = self._get_xg_rating(league, away_team, league_cfg)
        home_blended = self._blended_rating(league, home_team, league_cfg)
        away_blended = self._blended_rating(league, away_team, league_cfg)

        scale = self.config.probability_scale
        elo_prob = self.expected_home_win_prob(
            home_blended,
            away_blended,
            league_cfg.home_advantage_elo,
            scale=scale,
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
        model_prob_raw = prob_after_form

        return {
            "model_prob": model_prob_raw,
            "model_prob_raw": model_prob_raw,
            "elo_prob": elo_prob,
            "prob_after_form": prob_after_form,
            "league_hfa": league_hfa,
            "home_elo": home_blended,
            "away_elo": away_blended,
            "home_elo_traditional": home_trad,
            "away_elo_traditional": away_trad,
            "home_elo_xg": home_xg_elo,
            "away_elo_xg": away_xg_elo,
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
        home_xg: float | None = None,
        away_xg: float | None = None,
    ) -> None:
        league_cfg = self.config.for_league(league)

        home_trad = self._get_traditional_rating(league, home_team, league_cfg)
        away_trad = self._get_traditional_rating(league, away_team, league_cfg)
        s_home, s_away = self._match_scores(ftr)
        trad_margin = 1.0
        if self.config.goal_adjusted_elo and home_goals is not None and away_goals is not None:
            trad_margin = self.goal_margin_multiplier(home_goals, away_goals)

        self._update_elo_pair(
            league,
            home_team,
            away_team,
            home_rating=home_trad,
            away_rating=away_trad,
            s_home=s_home,
            s_away=s_away,
            margin_mult=trad_margin,
            set_home=lambda r: self._set_traditional_rating(league, home_team, r),
            set_away=lambda r: self._set_traditional_rating(league, away_team, r),
        )

        if home_xg is not None and away_xg is not None:
            home_xg_elo = self._get_xg_rating(league, home_team, league_cfg)
            away_xg_elo = self._get_xg_rating(league, away_team, league_cfg)
            xg_s_home, xg_s_away = self._xg_match_scores(home_xg, away_xg)
            xg_margin = (
                self.xg_margin_multiplier(home_xg, away_xg)
                if self.config.goal_adjusted_elo
                else 1.0
            )
            self._update_elo_pair(
                league,
                home_team,
                away_team,
                home_rating=home_xg_elo,
                away_rating=away_xg_elo,
                s_home=xg_s_home,
                s_away=xg_s_away,
                margin_mult=xg_margin,
                set_home=lambda r: self._set_xg_rating(league, home_team, r),
                set_away=lambda r: self._set_xg_rating(league, away_team, r),
            )

        self._league_home_results[league].append(1 if ftr == "H" else 0)
        self._form_modifier.record_match(league, home_team, away_team, ftr)

    def attach_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Walk-forward: predict each row, then update both Elo tracks."""
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
        has_xg = (
            cfg.home_xg_column in df.columns and cfg.away_xg_column in df.columns
        )

        if cfg.goal_adjusted_elo and not has_goals:
            print(
                "Warning: goal-adjusted Elo enabled but goal columns missing; "
                "traditional Elo uses margin multiplier = 1.0."
            )
        if cfg.alpha < 1.0 and not has_xg:
            print(
                "Warning: alpha < 1.0 but xG columns missing; "
                "only traditional Elo will affect blended ratings."
            )

        sort_cols = [cfg.date_column]
        if cfg.time_column in df.columns:
            sort_cols.append(cfg.time_column)
        sort_cols.append(league_col)

        out = df.sort_values(sort_cols).reset_index(drop=True)
        self.reset()

        calibrator: WalkForwardIsotonicCalibrator | None = None
        if self.config.use_post_hoc_calibration:
            calibrator = WalkForwardIsotonicCalibrator(
                min_samples=self.config.calibration_min_samples,
                refit_every=self.config.calibration_refit_every,
            )

        columns: dict[str, list[float]] = {
            "model_prob": [],
            "model_prob_raw": [],
            "elo_prob": [],
            "prob_after_form": [],
            "league_hfa": [],
            "home_elo": [],
            "away_elo": [],
            "home_elo_traditional": [],
            "away_elo_traditional": [],
            "home_elo_xg": [],
            "away_elo_xg": [],
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
            raw_prob = prediction["model_prob_raw"]
            if calibrator is not None:
                calibrated = calibrator.calibrate_probability(raw_prob)
                prediction["model_prob"] = calibrated
            else:
                prediction["model_prob"] = raw_prob

            for key, values in columns.items():
                values.append(prediction[key])

            if pd.notna(ftr):
                outcome = 1 if str(ftr) == "H" else 0
                if calibrator is not None:
                    calibrator.observe(raw_prob, outcome)
                home_goals: int | None = None
                away_goals: int | None = None
                if has_goals:
                    hg = getattr(row, cfg.home_goals_column)
                    ag = getattr(row, cfg.away_goals_column)
                    if pd.notna(hg) and pd.notna(ag):
                        home_goals = int(hg)
                        away_goals = int(ag)

                home_xg_val: float | None = None
                away_xg_val: float | None = None
                if has_xg:
                    hxg = getattr(row, cfg.home_xg_column)
                    axg = getattr(row, cfg.away_xg_column)
                    if pd.notna(hxg) and pd.notna(axg):
                        home_xg_val = float(hxg)
                        away_xg_val = float(axg)

                self._update_after_match(
                    str(league),
                    str(home),
                    str(away),
                    str(ftr),
                    home_goals=home_goals,
                    away_goals=away_goals,
                    home_xg=home_xg_val,
                    away_xg=away_xg_val,
                )

        for key, values in columns.items():
            out[key] = values
        return out
