"""
Match analysis functions for tournament rankings.

This module provides functions for analyzing individual matches and their
impact on player rankings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rankings.analysis.engine import RatingEngine

import polars as pl


def get_most_influential_matches(
    player_id: int,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    engine: "RatingEngine",
    top_n: int = 10,
) -> dict[str, pl.DataFrame]:
    """
    Get the most influential wins and losses for a specific player.

    This function uses network-aware edge flux (how much PageRank mass flows through
    an edge) and attributes it to individual matches, providing a better measure
    of match importance than simple match weights.

    Parameters
    ----------
    player_id : int
        The user_id of the player to analyze
    matches_df : pl.DataFrame
        Matches DataFrame with tournament_id, winner_team_id, loser_team_id, etc.
    players_df : pl.DataFrame
        Players DataFrame with user_id, team_id, tournament_id
    engine : RatingEngine
        The RatingEngine instance that has already computed tournament influences and edge flux
    top_n : int, optional
        Number of top matches to return for wins and losses

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary with 'wins' and 'losses' DataFrames containing:
        - match details (tournament_id, opponent info, scores)
        - match_flux (network-aware influence metric)
        - share_incoming/share_outgoing: share among the player's returned wins/losses
        - influence_rank (1 = most influential)
    """
    import math
    from datetime import timezone

    # Get tournament influences
    if (
        not hasattr(engine, "tournament_influence_")
        or engine.tournament_influence_ is None
    ):
        raise ValueError(
            "Engine must have computed tournament influences. Run rank_players() first."
        )

    tournament_influence = engine.tournament_influence_

    # Convert tournament influences to DataFrame for joining
    influence_df = pl.DataFrame(
        {
            "tournament_id": list(tournament_influence.keys()),
            "tournament_influence": list(tournament_influence.values()),
        }
    ).with_columns(pl.col("tournament_id").cast(pl.Int64))

    # Get all matches with tournament influences
    matches_with_influence = matches_df.join(
        influence_df, on="tournament_id", how="left"
    ).fill_null(
        1.0
    )  # Default influence for unseen tournaments

    # Exclude byes/forfeits to match engine filtering
    if "is_bye" in matches_with_influence.columns:
        matches_with_influence = matches_with_influence.filter(
            ~pl.col("is_bye").fill_null(False)
        )

    # Add time decay weight
    # First add event timestamp
    matches_with_influence = matches_with_influence.with_columns(
        pl.when(pl.col("last_game_finished_at").is_not_null())
        .then(pl.col("last_game_finished_at"))
        .otherwise(pl.col("match_created_at"))
        .fill_null(int(engine.now.timestamp()))
        .alias("event_ts")
    )

    # Calculate time decay
    matches_with_influence = matches_with_influence.with_columns(
        (
            (int(engine.now.timestamp()) - pl.col("event_ts").cast(pl.Float64))
            / 86400.0
        )
        .mul(-engine.decay_rate)
        .exp()
        .alias("time_decay")
    )

    # Ensure column types are correct
    matches_with_influence = matches_with_influence.with_columns(
        [
            pl.col("tournament_id").cast(pl.Int64),
            pl.col("winner_team_id").cast(pl.Int64),
            pl.col("loser_team_id").cast(pl.Int64),
        ]
    )

    # Calculate raw match weight w_m (time decay * tournament strength^beta)
    matches_with_influence = matches_with_influence.with_columns(
        (
            pl.col("time_decay")
            * (pl.col("tournament_influence") ** engine.beta)
        ).alias("w_m")
    )

    # Compute pair-level sums W_ij
    pair_sum = matches_with_influence.group_by(
        ["loser_team_id", "winner_team_id"]
    ).agg(pl.col("w_m").sum().alias("W_ij"))

    # Join pair sums back to matches
    matches_with_influence = matches_with_influence.join(
        pair_sum, on=["loser_team_id", "winner_team_id"], how="left"
    )

    # Detect engine mode (player vs team) based on what IDs were ranked
    mode = "unknown"
    rated_ids: set[int] = set()
    if (
        hasattr(engine, "ratings_df")
        and engine.ratings_df is not None
        and "id" in engine.ratings_df.columns
    ):
        rated_ids = set(engine.ratings_df["id"].to_list())
        # Check presence of team IDs and player IDs in rated IDs
        has_team_hits = False
        if "winner_team_id" in matches_df.columns:
            # Build a sample set from both team columns
            team_ids_col = (
                matches_df.select(["winner_team_id"])
                .rename({"winner_team_id": "id"})
                .vstack(
                    matches_df.select(["loser_team_id"]).rename(
                        {"loser_team_id": "id"}
                    )
                )
                .get_column("id")
                .drop_nulls()
            )
            sample_team_ids = (
                set(team_ids_col.head(1000).to_list())
                if matches_df.height > 0
                else set()
            )
            has_team_hits = any(tid in rated_ids for tid in sample_team_ids)

        has_player_hits = False
        if "user_id" in players_df.columns:
            sample_player_ids = (
                set(players_df["user_id"].drop_nulls().head(2000).to_list())
                if players_df.height > 0
                else set()
            )
            has_player_hits = any(uid in rated_ids for uid in sample_player_ids)

        if has_player_hits and not has_team_hits:
            mode = "player"
        elif has_team_hits and not has_player_hits:
            mode = "team"
        else:
            # Fallback: prefer player mode if any player IDs appear
            mode = "player" if has_player_hits else "team"

    # Calculate match flux depending on detected mode
    alpha = engine.damping_factor if hasattr(engine, "damping_factor") else 0.85
    if mode == "player":
        # Build player-pair rows per match
        pl_sel = players_df.select(
            ["tournament_id", "team_id", "user_id"]
        ).with_columns(
            [
                pl.col("tournament_id").cast(pl.Int64),
                pl.col("team_id").cast(pl.Int64),
                # Do not cast user_id to int; may be string IDs in synthetic data
            ]
        )

        winners = (
            matches_with_influence.select(
                ["match_id", "tournament_id", "winner_team_id", "w_m"]
            )  # carry w_m per match
            .with_columns(
                [
                    pl.col("tournament_id").cast(pl.Int64),
                    pl.col("winner_team_id").cast(pl.Int64),
                ]
            )
            .join(
                pl_sel,
                left_on=["tournament_id", "winner_team_id"],
                right_on=["tournament_id", "team_id"],
                how="inner",
            )
            .rename({"user_id": "winner_user_id"})
            .select(
                ["match_id", "winner_user_id", "w_m"]
            )  # keep w_m for pairing
        )

        losers = (
            matches_with_influence.select(
                ["match_id", "tournament_id", "loser_team_id", "w_m"]
            )
            .with_columns(
                [
                    pl.col("tournament_id").cast(pl.Int64),
                    pl.col("loser_team_id").cast(pl.Int64),
                ]
            )
            .join(
                pl_sel,
                left_on=["tournament_id", "loser_team_id"],
                right_on=["tournament_id", "team_id"],
                how="inner",
            )
            .rename({"user_id": "loser_user_id"})
            .select(
                ["match_id", "loser_user_id", "w_m"]
            )  # keep w_m for pairing
        )

        pairs = losers.join(
            winners, on=["match_id", "w_m"], how="inner"
        ).select(["match_id", "loser_user_id", "winner_user_id", "w_m"])

        # Map r_loser and join denominators if available
        denom_df = None
        if (
            hasattr(engine, "denominators_df_")
            and engine.denominators_df_ is not None
        ):
            # Expect denominators for player mode (id == loser_user_id)
            denom_df = (
                engine.denominators_df_.select(["id", "denom"])
                .rename({"id": "loser_user_id"})
                .with_columns(pl.col("loser_user_id").cast(pl.Int64))
            )

        rating_map = {}
        if hasattr(engine, "ratings_df") and engine.ratings_df is not None:
            rating_map = dict(
                zip(engine.ratings_df["id"], engine.ratings_df["score"])
            )
        default_score = (
            engine.global_prior_
            if getattr(engine, "global_prior_", None) is not None
            else 0.0
        )

        if denom_df is not None:
            pairs = (
                pairs.join(denom_df, on="loser_user_id", how="left")
                .with_columns(
                    [
                        pl.col("loser_user_id")
                        .map_elements(
                            lambda x: rating_map.get(x, default_score),
                            return_dtype=pl.Float64,
                        )
                        .alias("r_loser"),
                    ]
                )
                .with_columns(
                    pl.when(pl.col("denom") > 0)
                    .then(
                        alpha
                        * pl.col("r_loser")
                        * (pl.col("w_m") / pl.col("denom"))
                    )
                    .otherwise(0.0)
                    .alias("pair_flux")
                )
            )
        else:
            # Fallback to row-sum denominator if engine didn't expose denoms
            loser_out = pairs.group_by("loser_user_id").agg(
                pl.col("w_m").sum().alias("W_loser_total")
            )
            pairs = (
                pairs.join(loser_out, on="loser_user_id", how="left")
                .with_columns(
                    [
                        pl.col("loser_user_id")
                        .map_elements(
                            lambda x: rating_map.get(x, default_score),
                            return_dtype=pl.Float64,
                        )
                        .alias("r_loser"),
                    ]
                )
                .with_columns(
                    pl.when(pl.col("W_loser_total") > 0)
                    .then(
                        alpha
                        * pl.col("r_loser")
                        * (pl.col("w_m") / pl.col("W_loser_total"))
                    )
                    .otherwise(0.0)
                    .alias("pair_flux")
                )
            )

        # For selected player, aggregate by match as incoming and outgoing
        wins_pairs = pairs.filter(pl.col("winner_user_id") == player_id)
        losses_pairs = pairs.filter(pl.col("loser_user_id") == player_id)

        wins_flux = wins_pairs.group_by("match_id").agg(
            pl.col("pair_flux").sum().alias("match_flux")
        )
        losses_flux = losses_pairs.group_by("match_id").agg(
            pl.col("pair_flux").sum().alias("match_flux")
        )

        # Attach to match details table
        matches_with_influence = matches_with_influence.join(
            wins_flux, on="match_id", how="left"
        ).with_columns(pl.col("match_flux").fill_null(0.0))
        # For losses, we will build a separate losses table below using join
        losses_match_flux = (
            matches_with_influence.select(["match_id"])
            .join(losses_flux, on="match_id", how="left")
            .with_columns(pl.col("match_flux").fill_null(0.0))
        )
    else:
        # Team mode
        denom_df = None
        if (
            hasattr(engine, "denominators_df_")
            and engine.denominators_df_ is not None
        ):
            # Expect denominators for team mode (id == loser_team_id)
            denom_df = (
                engine.denominators_df_.select(["id", "denom"])
                .rename({"id": "loser_team_id"})
                .with_columns(pl.col("loser_team_id").cast(pl.Int64))
            )
        if denom_df is not None:
            matches_with_influence = matches_with_influence.join(
                denom_df, on="loser_team_id", how="left"
            )
        else:
            # Fallback to total outgoing weight if no denominators provided
            loser_total_weights = matches_with_influence.group_by(
                "loser_team_id"
            ).agg(pl.col("w_m").sum().alias("W_loser_total"))

            matches_with_influence = matches_with_influence.join(
                loser_total_weights, on="loser_team_id", how="left"
            )

        rating_map = {}
        if hasattr(engine, "ratings_df") and engine.ratings_df is not None:
            rating_map = dict(
                zip(engine.ratings_df["id"], engine.ratings_df["score"])
            )
        default_score = (
            engine.global_prior_
            if getattr(engine, "global_prior_", None) is not None
            else 0.0
        )

        matches_with_influence = matches_with_influence.with_columns(
            [
                pl.col("loser_team_id")
                .map_elements(
                    lambda x: rating_map.get(x, default_score),
                    return_dtype=pl.Float64,
                )
                .alias("r_loser"),
                pl.col("winner_team_id")
                .map_elements(
                    lambda x: rating_map.get(x, default_score),
                    return_dtype=pl.Float64,
                )
                .alias("r_winner"),
            ]
        )

        if "denom" in matches_with_influence.columns:
            matches_with_influence = matches_with_influence.with_columns(
                pl.when(pl.col("denom") > 0)
                .then(
                    alpha
                    * pl.col("r_loser")
                    * (pl.col("w_m") / pl.col("denom"))
                )
                .otherwise(0.0)
                .alias("match_flux")
            )
        else:
            matches_with_influence = matches_with_influence.with_columns(
                pl.when(pl.col("W_loser_total") > 0)
                .then(
                    alpha
                    * pl.col("r_loser")
                    * (pl.col("w_m") / pl.col("W_loser_total"))
                )
                .otherwise(0.0)
                .alias("match_flux")
            )

    # Ensure types are still correct after all operations
    matches_with_influence = matches_with_influence.with_columns(
        [
            pl.col("tournament_id").cast(pl.Int64),
            pl.col("winner_team_id").cast(pl.Int64),
            pl.col("loser_team_id").cast(pl.Int64),
        ]
    )

    # Get player's teams across all tournaments
    player_teams = (
        players_df.filter(pl.col("user_id") == player_id)
        .select(["tournament_id", "team_id"])
        .with_columns(
            [
                pl.col("tournament_id").cast(pl.Int64),
                pl.col("team_id").cast(pl.Int64),
            ]
        )
    )

    # Find matches where player's team won (or where player contributed incoming flux in player mode)
    wins = matches_with_influence.join(
        player_teams,
        left_on=["tournament_id", "winner_team_id"],
        right_on=["tournament_id", "team_id"],
        how="inner",
    ).filter(pl.col("loser_team_id").is_not_null())

    # In player mode, we may not have r_loser/r_winner present
    if "r_loser" not in wins.columns:
        wins = wins.with_columns(
            [
                pl.lit(None).alias("r_loser"),
                pl.lit(None).alias("r_winner"),
            ]
        )

    # Calculate share of incoming flux for wins
    if not wins.is_empty():
        # Calculate share relative to total flux in this result set
        # This shows the relative importance among the player's wins
        total_win_flux = (
            wins["match_flux"].sum() if "match_flux" in wins.columns else 0
        )

        if total_win_flux > 0:
            wins = wins.with_columns(
                (pl.col("match_flux") / total_win_flux).alias("share_incoming")
            )
        else:
            wins = wins.with_columns(pl.lit(0.0).alias("share_incoming"))

    # Find matches where player's team lost
    losses = matches_with_influence.join(
        player_teams,
        left_on=["tournament_id", "loser_team_id"],
        right_on=["tournament_id", "team_id"],
        how="inner",
    ).filter(pl.col("winner_team_id").is_not_null())

    # If player mode, replace match_flux for losses using the per-player losses aggregation
    if mode == "player":
        # Join per-player loss flux and ensure it becomes the active match_flux
        losses = losses.join(losses_match_flux, on="match_id", how="left")
        # Handle potential column collision from join (e.g., match_flux_right)
        new_flux_col = (
            pl.coalesce(
                [
                    pl.col("match_flux_right").alias("__tmp__"),
                    pl.col("match_flux").alias("__tmp__"),
                ]
            )
            if "match_flux_right" in losses.columns
            else pl.col("match_flux")
        )
        losses = losses.with_columns(
            new_flux_col.alias("match_flux")
        ).with_columns(pl.col("match_flux").fill_null(0.0))
        if "match_flux_right" in losses.columns:
            losses = losses.drop("match_flux_right")

    if "r_loser" not in losses.columns:
        losses = losses.with_columns(
            [
                pl.lit(None).alias("r_loser"),
                pl.lit(None).alias("r_winner"),
            ]
        )

    # Calculate share of outgoing flux for losses
    if not losses.is_empty():
        # Calculate share relative to total flux in this result set
        # This shows the relative importance among the player's losses
        total_loss_flux = (
            losses["match_flux"].sum() if "match_flux" in losses.columns else 0
        )

        if total_loss_flux > 0:
            losses = losses.with_columns(
                (pl.col("match_flux") / total_loss_flux).alias("share_outgoing")
            )
        else:
            losses = losses.with_columns(pl.lit(0.0).alias("share_outgoing"))

    # Get opponent player info for wins
    if not wins.is_empty():
        # Get losing team players
        opponent_players_wins = players_df.group_by(
            ["tournament_id", "team_id"]
        ).agg(pl.col("username").str.concat(", ").alias("opponent_players"))

        wins = wins.join(
            opponent_players_wins,
            left_on=["tournament_id", "loser_team_id"],
            right_on=["tournament_id", "team_id"],
            how="left",
        )

        # Sort by match flux and add rank
        wins = (
            wins.sort("match_flux", descending=True)
            .with_columns(pl.arange(1, wins.height + 1).alias("influence_rank"))
            .select(
                [
                    "influence_rank",
                    "tournament_id",
                    "match_id",
                    "loser_team_id",
                    "opponent_players",
                    "team1_score",
                    "team2_score",
                    "total_games",
                    "match_flux",
                    "share_incoming",
                    "w_m",
                    "tournament_influence",
                    "time_decay",
                    "event_ts",
                    "r_loser",
                    "r_winner",
                ]
            )
            .head(top_n)
        )
    else:
        wins = pl.DataFrame()

    # Get opponent player info for losses
    if not losses.is_empty():
        # Get winning team players
        opponent_players_losses = players_df.group_by(
            ["tournament_id", "team_id"]
        ).agg(pl.col("username").str.concat(", ").alias("opponent_players"))

        losses = losses.join(
            opponent_players_losses,
            left_on=["tournament_id", "winner_team_id"],
            right_on=["tournament_id", "team_id"],
            how="left",
        )

        # Sort by match flux and add rank
        losses = (
            losses.sort("match_flux", descending=True)
            .with_columns(
                pl.arange(1, losses.height + 1).alias("influence_rank")
            )
            .select(
                [
                    "influence_rank",
                    "tournament_id",
                    "match_id",
                    "winner_team_id",
                    "opponent_players",
                    "team1_score",
                    "team2_score",
                    "total_games",
                    "match_flux",
                    "share_outgoing",
                    "w_m",
                    "tournament_influence",
                    "time_decay",
                    "event_ts",
                    "r_loser",
                    "r_winner",
                ]
            )
            .head(top_n)
        )
    else:
        losses = pl.DataFrame()

    return {"wins": wins, "losses": losses}


def get_player_match_history(
    player_id: int,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    teams_df: pl.DataFrame,
    tournament_data: list[dict],
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Get a player's complete match history with names and context.

    Parameters
    ----------
    player_id : int
        The user_id of the player
    matches_df : pl.DataFrame
        Matches DataFrame
    players_df : pl.DataFrame
        Players DataFrame
    teams_df : pl.DataFrame
        Teams DataFrame
    tournament_data : list[dict]
        Raw tournament data
    limit : int, optional
        Maximum number of matches to return

    Returns
    -------
    pl.DataFrame
        Player's match history with full context
    """
    from rankings.analysis.utils.names import create_match_summary_with_names

    # Get player's teams
    player_teams = players_df.filter(pl.col("user_id") == player_id).select(
        ["tournament_id", "team_id"]
    )

    # Get all matches for those teams
    wins = matches_df.join(
        player_teams,
        left_on=["tournament_id", "winner_team_id"],
        right_on=["tournament_id", "team_id"],
        how="inner",
    ).with_columns(pl.lit("win").alias("result"))

    losses = matches_df.join(
        player_teams,
        left_on=["tournament_id", "loser_team_id"],
        right_on=["tournament_id", "team_id"],
        how="inner",
    ).with_columns(pl.lit("loss").alias("result"))

    # Combine wins and losses
    all_matches = pl.concat([wins, losses])

    # Add names and context
    result = create_match_summary_with_names(
        all_matches, players_df, teams_df, tournament_data
    )

    # Sort by date (most recent first)
    result = result.sort("event_ts", descending=True)

    if limit:
        result = result.head(limit)

    return result
