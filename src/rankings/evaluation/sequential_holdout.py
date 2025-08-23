# evaluation/sequential_holdout.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from rankings.analysis.engine import RatingEngine

from rankings.analysis.engine import RatingEngine
from rankings.analysis.utils.summaries import derive_team_ratings_from_players
from rankings.core.constants import MIN_TOURNAMENTS_BEFORE_CV
from rankings.evaluation.loss import (
    compute_match_loss,
    compute_match_probability,
    fit_alpha_parameter,
)


def sequential_holdout_loss(
    *,
    engine_class: type[RatingEngine] = RatingEngine,
    engine_params: dict[str, Any],
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    teams_df: pl.DataFrame,
    ranking_entity: str = "player",  # "player" or "team"
    prediction_entity: str = "team",  # what we evaluate on
    agg_func: str = "mean",  # player -> team aggregation
    min_history_tournaments: int = MIN_TOURNAMENTS_BEFORE_CV,
    fit_alpha: bool = True,
    alpha_bounds: tuple[float, float] = (0.1, 10.0),
    n_splits: int
    | None = None,  # NEW: number of spaced-out holdouts (None=all)
    weighting_scheme: str = "none",
) -> tuple[float, pl.DataFrame]:
    """
    Rolling one-tournament-ahead evaluation, but can subsample holdout tournaments.

    Parameters
    ----------
    engine_class, engine_params
        How to build the `RatingEngine`.
    matches_df / players_df / teams_df
        Parsed tables from `parse_tournaments_data`.
    ranking_entity
        Which entity we *rate* ("player" or "team").
    prediction_entity
        Which entity we *predict* matches for ("team" or "player").
    min_history_tournaments
        Warm-up length before the first evaluation split.
    fit_alpha
        Whether to MLE-fit the logistic temperature on training data.
    alpha_bounds
        Search bounds for `alpha` when fitting.
    n_splits
        If set, only evaluate on this many spaced-out tournaments (default: all).
    weighting_scheme
        Weighting scheme: "var_inv", "entropy", or "none"
    Returns
    -------
    (avg_loss, per_tournament_df)
        - `avg_loss`: mean held-out log-loss
        - `per_tournament_df`: one row per test tournament with detailed metrics
    """
    # Preparation
    # Tournament start times (earliest finished game == proxy for start)
    tourney_times = (
        matches_df.group_by("tournament_id")
        .agg(pl.col("last_game_finished_at").min().alias("start_ts"))
        .sort("start_ts")
    )
    all_tournaments = tourney_times["tournament_id"].to_list()

    if len(all_tournaments) <= min_history_tournaments:
        raise ValueError(
            f"Need >{min_history_tournaments} tournaments, "
            f"found only {len(all_tournaments)}."
        )

    # Determine which tournaments to hold out
    possible_indices = list(
        range(min_history_tournaments, len(all_tournaments))
    )
    if n_splits is not None and n_splits < len(possible_indices):
        # Evenly space the holdout tournaments
        # Always include the last tournament
        if n_splits == 1:
            chosen_indices = [possible_indices[-1]]
        else:
            # linspace returns floats, so round and deduplicate
            linspace = np.linspace(0, len(possible_indices) - 1, n_splits)
            chosen_indices = sorted(
                {possible_indices[int(round(idx))] for idx in linspace}
            )
    else:
        chosen_indices = possible_indices

    # Helper to grab matches by tournament id list
    def _matches_for(tids: list[int]) -> pl.DataFrame:
        return matches_df.filter(pl.col("tournament_id").is_in(tids))

    per_tournament_rows = []
    rolling_losses: list[float] = []

    # Walk forward, but only for selected tournaments
    for i in chosen_indices:
        test_tid = all_tournaments[i]
        train_tids = all_tournaments[:i]

        train_df = _matches_for(train_tids)
        test_df = _matches_for([test_tid])

        # 1) Train ratings on historical matches
        engine = engine_class(**engine_params)

        if ranking_entity == "player":
            rating_df = engine.rank_players(train_df, players_df)
            if prediction_entity == "team":
                rating_df = derive_team_ratings_from_players(
                    players_df, rating_df, agg=agg_func
                ).rename({"team_id": "id", "team_rating": "score"})
        else:  # ranking_entity == "team"
            rating_df = engine.rank_teams(train_df)
            rating_df = rating_df.rename(
                {"team_rank": "score", "team_id": "id"}
            )
            if prediction_entity == "player":
                raise ValueError(
                    "Cannot predict on players when ranking teams."
                )

        rating_map = dict(zip(rating_df["id"], rating_df["score"]))

        # 2) Optional alpha tuning on training data
        if fit_alpha:
            winner_col = (
                "winner_team_id"
                if prediction_entity == "team"
                else "winner_user_id"
            )
            loser_col = (
                "loser_team_id"
                if prediction_entity == "team"
                else "loser_user_id"
            )
            alpha = fit_alpha_parameter(
                train_df,
                rating_map,
                alpha_bounds=alpha_bounds,
                winner_id_col=winner_col,
                loser_id_col=loser_col,
                weighting_scheme=weighting_scheme,
            )
        else:
            alpha = 1.0

        # 3) Predict each held-out match & accumulate loss
        match_losses = []
        probs = []

        for row in test_df.iter_rows(named=True):
            wid = (
                row["winner_team_id"]
                if prediction_entity == "team"
                else row["winner_user_id"]
            )
            lid = (
                row["loser_team_id"]
                if prediction_entity == "team"
                else row["loser_user_id"]
            )

            p = compute_match_probability(
                rating_map.get(wid, 0.0),
                rating_map.get(lid, 0.0),
                alpha=alpha,
            )
            loss = compute_match_loss(p, actual_winner_is_a=True)
            match_losses.append(loss)
            probs.append(p)

        tournament_loss = float(np.mean(match_losses))
        rolling_losses.append(tournament_loss)

        per_tournament_rows.append(
            {
                "tournament_id": test_tid,
                "n_matches": len(match_losses),
                "loss": tournament_loss,
                "mean_pred": float(np.mean(probs)),
                "alpha": alpha,
                "train_size": train_df.height,
            }
        )

    per_tournament_df = pl.DataFrame(per_tournament_rows).sort("tournament_id")
    avg_loss = float(np.mean(rolling_losses))

    return avg_loss, per_tournament_df
