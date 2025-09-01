"""
Match analysis functions for tournament rankings.

This module provides functions for analyzing individual matches and their
impact on player rankings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rankings.analysis.engine import RatingEngine

import polars as pl

# Support for new engine
try:
    from rankings.algorithms import TTLEngine
except ImportError:
    TTLEngine = None


def _get_engine_attributes(engine):
    """
    Extract necessary attributes from either old or new engine type.

    Returns a dict with keys:
    - tournament_influence
    - now
    - decay_rate
    - damping_factor
    - beta
    - ratings_df
    - win_pagerank
    - loss_pagerank
    - denominators_df
    - global_prior
    """
    from datetime import datetime, timezone

    attrs = {}

    # Check if it's the new TTLEngine (by checking for specific attributes)
    if hasattr(engine, "last_result") and hasattr(
        engine, "tournament_influence"
    ):
        attrs["tournament_influence"] = engine.tournament_influence
        # For TTLEngine, get now from clock or use current time
        if hasattr(engine, "clock") and hasattr(engine.clock, "now"):
            now_value = engine.clock.now
            # Check if it's already a datetime or needs conversion from timestamp
            if isinstance(now_value, (int, float)):
                attrs["now"] = datetime.fromtimestamp(
                    now_value, tz=timezone.utc
                )
            else:
                attrs["now"] = now_value
        else:
            attrs["now"] = datetime.now(timezone.utc)
        attrs["decay_rate"] = (
            engine.config.decay.decay_rate if hasattr(engine, "config") else 0.0
        )
        attrs["damping_factor"] = (
            engine.config.pagerank.alpha if hasattr(engine, "config") else 0.85
        )
        attrs["beta"] = (
            engine.config.engine.beta
            if hasattr(engine, "config") and hasattr(engine.config, "engine")
            else 0.0
        )
        attrs["ratings_df"] = None  # May need to build from last_result
        attrs["win_pagerank"] = None
        attrs["loss_pagerank"] = None
        attrs["denominators_df"] = None
        # Use a small positive prior to avoid zero flux on unmapped IDs
        prior = 1e-12
        if hasattr(engine, "last_result") and engine.last_result:
            if (
                hasattr(engine.last_result, "exposure")
                and engine.last_result.exposure is not None
            ):
                exp = (
                    engine.last_result.exposure.tolist()
                    if hasattr(engine.last_result.exposure, "tolist")
                    else list(engine.last_result.exposure)
                )
                if exp:
                    # Use median exposure as prior (robust and strictly positive)
                    exp_sorted = sorted(float(x) for x in exp if x is not None)
                    if exp_sorted:
                        prior = max(exp_sorted[len(exp_sorted) // 2], 1e-12)
        attrs["global_prior"] = prior

        # Build ratings_df from last_result if available
        if hasattr(engine, "last_result") and engine.last_result:
            result = engine.last_result
            if hasattr(result, "ids") and hasattr(result, "scores"):
                # Handle scores as either numpy array or list
                scores_list = (
                    result.scores.tolist()
                    if hasattr(result.scores, "tolist")
                    else result.scores
                )
                data = {"id": result.ids, "score": scores_list}
                if result.win_pagerank is not None:
                    attrs["win_pagerank"] = result.win_pagerank
                    win_pr_list = (
                        result.win_pagerank.tolist()
                        if hasattr(result.win_pagerank, "tolist")
                        else result.win_pagerank
                    )
                    data["win_pr"] = win_pr_list
                if result.loss_pagerank is not None:
                    attrs["loss_pagerank"] = result.loss_pagerank
                    loss_pr_list = (
                        result.loss_pagerank.tolist()
                        if hasattr(result.loss_pagerank, "tolist")
                        else result.loss_pagerank
                    )
                    data["loss_pr"] = loss_pr_list
                attrs["ratings_df"] = pl.DataFrame(data)

    # Check if it's the old RatingEngine
    else:
        attrs["tournament_influence"] = getattr(
            engine, "tournament_influence_", None
        )
        now_value = getattr(engine, "now", None)
        # Handle different types of now values
        if now_value is not None:
            if isinstance(now_value, (int, float)):
                attrs["now"] = datetime.fromtimestamp(
                    now_value, tz=timezone.utc
                )
            else:
                attrs["now"] = now_value
        else:
            attrs["now"] = datetime.now(timezone.utc)
        attrs["decay_rate"] = getattr(engine, "decay_rate", 0.0)
        attrs["damping_factor"] = getattr(engine, "damping_factor", 0.85)
        attrs["beta"] = getattr(engine, "beta", 0.0)
        attrs["ratings_df"] = getattr(engine, "ratings_df", None)
        attrs["win_pagerank"] = getattr(engine, "win_pagerank_", None)
        attrs["loss_pagerank"] = getattr(engine, "loss_pagerank_", None)
        attrs["denominators_df"] = getattr(engine, "denominators_df_", None)
        attrs["global_prior"] = getattr(engine, "global_prior_", 0.0)

    return attrs


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

    # Get engine attributes
    engine_attrs = _get_engine_attributes(engine)

    # Get tournament influences
    if engine_attrs["tournament_influence"] is None:
        raise ValueError(
            "Engine must have computed tournament influences. Run rank_players() first."
        )

    tournament_influence = engine_attrs["tournament_influence"]

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
    # Handle different timestamp columns that might exist
    if "last_game_finished_at" in matches_with_influence.columns:
        timestamp_expr = pl.col("last_game_finished_at")
    elif "match_created_at" in matches_with_influence.columns:
        timestamp_expr = pl.col("match_created_at")
    elif "ts" in matches_with_influence.columns:
        timestamp_expr = pl.col("ts")
    else:
        # Default to now if no timestamp column exists
        timestamp_expr = pl.lit(
            int(
                engine_attrs["now"].timestamp()
                if hasattr(engine_attrs["now"], "timestamp")
                else engine_attrs["now"]
            )
            if engine_attrs["now"]
            else 0
        )

    matches_with_influence = matches_with_influence.with_columns(
        timestamp_expr.fill_null(
            int(
                engine_attrs["now"].timestamp()
                if hasattr(engine_attrs["now"], "timestamp")
                else engine_attrs["now"]
            )
            if engine_attrs["now"]
            else 0
        ).alias("event_ts")
    )

    # Calculate time decay
    decay_rate = engine_attrs["decay_rate"]
    # Handle now as either datetime or timestamp
    if engine_attrs["now"] is not None:
        if hasattr(engine_attrs["now"], "timestamp"):
            now_ts = int(engine_attrs["now"].timestamp())
        else:
            now_ts = int(engine_attrs["now"])
    else:
        now_ts = 0
    matches_with_influence = matches_with_influence.with_columns(
        ((now_ts - pl.col("event_ts").cast(pl.Float64)) / 86400.0)
        .mul(-decay_rate)
        .exp()
        .alias("time_decay")
    )

    # Ensure column types are correct and add total_games if needed
    columns_to_cast = [
        pl.col("tournament_id").cast(pl.Int64),
        pl.col("winner_team_id").cast(pl.Int64),
        pl.col("loser_team_id").cast(pl.Int64),
    ]

    # Add total_games if we have score columns
    if (
        "team1_score" in matches_with_influence.columns
        and "team2_score" in matches_with_influence.columns
    ):
        if "total_games" not in matches_with_influence.columns:
            columns_to_cast.append(
                (pl.col("team1_score") + pl.col("team2_score")).alias(
                    "total_games"
                )
            )
    elif "total_games" not in matches_with_influence.columns:
        # Default to 0 if no score information available
        columns_to_cast.append(pl.lit(0).alias("total_games"))

    matches_with_influence = matches_with_influence.with_columns(
        columns_to_cast
    )

    # Calculate raw match weight w_m (time decay * tournament strength^beta)
    beta = engine_attrs["beta"]
    matches_with_influence = matches_with_influence.with_columns(
        (pl.col("time_decay") * (pl.col("tournament_influence") ** beta)).alias(
            "w_m"
        )
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
    # Default to player mode for ExposureLogOdds-like engines
    mode = "player"
    if (
        engine_attrs["ratings_df"] is not None
        and "id" in engine_attrs["ratings_df"].columns
    ):
        # Convert all IDs to strings for robust comparison
        to_str_set = lambda seq: set(map(str, seq))
        rated_str = to_str_set(engine_attrs["ratings_df"]["id"].to_list())

        # Get sample team IDs
        team_str = set()
        if "winner_team_id" in matches_df.columns and matches_df.height > 0:
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
            team_str = to_str_set(team_ids_col.head(1000).to_list())

        # Get sample player IDs
        player_str = set()
        if players_df.height > 0 and "user_id" in players_df.columns:
            player_str = to_str_set(
                players_df["user_id"].drop_nulls().head(2000).to_list()
            )

        # Check for overlaps
        has_player_hits = bool(rated_str & player_str)
        has_team_hits = bool(rated_str & team_str)

        # Prefer player mode on ties/ambiguity (since ExposureLogOddsEngine ranks players)
        mode = "player" if has_player_hits or not has_team_hits else "team"

    # Calculate match flux depending on detected mode
    alpha = engine_attrs["damping_factor"]
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

        pairs = losers.join(winners, on=["match_id"], how="inner").select(
            ["match_id", "loser_user_id", "winner_user_id", "w_m"]
        )

        # Map r_loser and join denominators if available
        denom_df = None
        if engine_attrs["denominators_df"] is not None:
            # Expect denominators for player mode (id == loser_user_id)
            denom_df = (
                engine_attrs["denominators_df"]
                .select(["id", "denom"])
                .rename({"id": "loser_user_id"})
                .with_columns(pl.col("loser_user_id").cast(pl.Int64))
            )

        # Build PageRank maps for exposure log-odds engine
        # Ensure default_score is strictly positive to avoid zero flux
        default_score = max(float(engine_attrs["global_prior"] or 0.0), 1e-12)
        r_win_map = {}
        r_loss_map = {}

        # Check if this is an exposure log-odds engine with PageRank vectors
        if engine_attrs["win_pagerank"] is not None:
            if engine_attrs["ratings_df"] is not None:
                player_ids = engine_attrs["ratings_df"]["id"].to_list()
                for i, pid in enumerate(player_ids):
                    if i < len(engine_attrs["win_pagerank"]):
                        r_win_map[pid] = float(engine_attrs["win_pagerank"][i])
                    if engine_attrs["loss_pagerank"] is not None and i < len(
                        engine_attrs["loss_pagerank"]
                    ):
                        r_loss_map[pid] = float(
                            engine_attrs["loss_pagerank"][i]
                        )

        # Fallback to using ratings if PageRanks not available
        if not r_win_map and engine_attrs["ratings_df"] is not None:
            rating_map = dict(
                zip(
                    engine_attrs["ratings_df"]["id"],
                    engine_attrs["ratings_df"]["score"],
                )
            )
            r_win_map = rating_map
            r_loss_map = rating_map

        # Always compute winner outgoing mass (for loss graph denominator)
        winner_out = pairs.group_by("winner_user_id").agg(
            pl.col("w_m").sum().alias("W_winner_total")
        )
        pairs = pairs.join(winner_out, on="winner_user_id", how="left")

        # For losers, use engine denominators if available, else compute
        if denom_df is not None:
            pairs = pairs.join(denom_df, on="loser_user_id", how="left")
            # Use denom if available, else fall back to computed total
            loser_out = pairs.group_by("loser_user_id").agg(
                pl.col("w_m").sum().alias("W_loser_total_computed")
            )
            pairs = pairs.join(loser_out, on="loser_user_id", how="left")
            pairs = pairs.with_columns(
                pl.when(pl.col("denom").is_not_null())
                .then(pl.col("denom"))
                .otherwise(pl.col("W_loser_total_computed"))
                .alias("D_loser_final")
            )
        else:
            # No engine denominators, compute from pairs
            loser_out = pairs.group_by("loser_user_id").agg(
                pl.col("w_m").sum().alias("W_loser_total")
            )
            pairs = pairs.join(loser_out, on="loser_user_id", how="left")
            pairs = pairs.with_columns(
                pl.col("W_loser_total").alias("D_loser_final")
            )

        # Add PageRank values and compute fluxes
        pairs = pairs.with_columns(
            [
                # For win flux: use loser's win PageRank (they are source in win graph)
                pl.col("loser_user_id")
                .map_elements(
                    lambda x: r_win_map.get(x, default_score),
                    return_dtype=pl.Float64,
                )
                .alias("r_loser"),
                # For loss flux: use winner's loss PageRank (they are source in loss graph)
                pl.col("winner_user_id")
                .map_elements(
                    lambda x: r_loss_map.get(x, default_score),
                    return_dtype=pl.Float64,
                )
                .alias("r_winner"),
            ]
        ).with_columns(
            [
                # Win flux: loser's rating * edge weight / loser's denominator
                pl.when(pl.col("D_loser_final") > 0)
                .then(
                    alpha
                    * pl.col("r_loser")
                    * (pl.col("w_m") / pl.col("D_loser_final"))
                )
                .otherwise(0.0)
                .alias("win_flux"),
                # Loss flux: winner's rating * edge weight / winner's OUTGOING in loss graph
                pl.when(pl.col("W_winner_total") > 0)
                .then(
                    alpha
                    * pl.col("r_winner")
                    * (pl.col("w_m") / pl.col("W_winner_total"))
                )
                .otherwise(0.0)
                .alias("loss_flux"),
            ]
        )

        # For selected player, aggregate by match as incoming and outgoing
        wins_pairs = pairs.filter(pl.col("winner_user_id") == player_id)
        losses_pairs = pairs.filter(pl.col("loser_user_id") == player_id)

        wins_flux = wins_pairs.group_by("match_id").agg(
            pl.col("win_flux").sum().alias("match_flux")
        )
        losses_flux = losses_pairs.group_by("match_id").agg(
            pl.col("loss_flux").sum().alias("match_flux")
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
            and engine_attrs["denominators_df"] is not None
        ):
            # Expect denominators for team mode (id == loser_team_id)
            denom_df = (
                engine_attrs["denominators_df"]
                .select(["id", "denom"])
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
        if engine_attrs["ratings_df"] is not None:
            rating_map = dict(
                zip(
                    engine_attrs["ratings_df"]["id"],
                    engine_attrs["ratings_df"]["score"],
                )
            )
        default_score = (
            engine_attrs["global_prior"]
            if engine_attrs["global_prior"] is not None
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

    # Get opponent and teammate player info for wins
    if not wins.is_empty():
        # Get losing team players (opponents)
        opponent_players_wins = players_df.group_by(
            ["tournament_id", "team_id"]
        ).agg(pl.col("username").str.concat(", ").alias("opponent_players"))

        # Get winning team players (teammates)
        teammate_players_wins = players_df.group_by(
            ["tournament_id", "team_id"]
        ).agg(
            # Filter out the current player from teammates list
            pl.col("username")
            .filter(pl.col("user_id") != player_id)
            .str.concat(", ")
            .alias("teammate_players")
        )

        wins = wins.join(
            opponent_players_wins,
            left_on=["tournament_id", "loser_team_id"],
            right_on=["tournament_id", "team_id"],
            how="left",
        ).join(
            teammate_players_wins,
            left_on=["tournament_id", "winner_team_id"],
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
                    "teammate_players",
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

    # Get opponent and teammate player info for losses
    if not losses.is_empty():
        # Get winning team players (opponents)
        opponent_players_losses = players_df.group_by(
            ["tournament_id", "team_id"]
        ).agg(pl.col("username").str.concat(", ").alias("opponent_players"))

        # Get losing team players (teammates)
        teammate_players_losses = players_df.group_by(
            ["tournament_id", "team_id"]
        ).agg(
            # Filter out the current player from teammates list
            pl.col("username")
            .filter(pl.col("user_id") != player_id)
            .str.concat(", ")
            .alias("teammate_players")
        )

        losses = losses.join(
            opponent_players_losses,
            left_on=["tournament_id", "winner_team_id"],
            right_on=["tournament_id", "team_id"],
            how="left",
        ).join(
            teammate_players_losses,
            left_on=["tournament_id", "loser_team_id"],
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
                    "teammate_players",
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
    limit: int | None = None,
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


def filter_strict_4v4(
    matches: pl.DataFrame, players: pl.DataFrame, appearances: pl.DataFrame
) -> pl.DataFrame:
    """Return only matches where each side has exactly 4 participating players.

    Uses per-match player appearances from the public players route to count how
    many players actually participated on each team in each match. If
    `appearances` lacks team_id, this function derives team membership by
    joining tournament rosters from `players`.
    """
    if appearances is None or appearances.is_empty():
        return matches

    roster_df = players.select(["tournament_id", "team_id", "user_id"]).unique()

    base_cols = [
        pl.col("tournament_id").cast(pl.Int64).alias("tournament_id"),
        pl.col("match_id").cast(pl.Int64).alias("match_id"),
        pl.col("user_id").cast(pl.Int64).alias("user_id"),
    ]
    if "team_id" in appearances.columns:
        base_cols.append(pl.col("team_id").cast(pl.Int64).alias("team_id"))
    appearances_df = appearances.select(base_cols).unique(
        subset=["tournament_id", "match_id", "user_id"]
    )

    needs_team = (
        "team_id" not in appearances_df.columns
    ) or appearances_df.select(pl.col("team_id").is_null().any()).item()
    if needs_team:
        appearances_df = appearances_df.join(
            roster_df, on=["tournament_id", "user_id"], how="left"
        )
        if "team_id_right" in appearances_df.columns:
            appearances_df = appearances_df.with_columns(
                pl.coalesce([pl.col("team_id"), pl.col("team_id_right")]).alias(
                    "team_id"
                )
            ).drop(
                [c for c in ["team_id_right"] if c in appearances_df.columns]
            )

    team_counts_df = (
        appearances_df.drop_nulls(["team_id"])
        .group_by(["tournament_id", "match_id", "team_id"])
        .agg(pl.len().alias("n"))
    )

    match_keys_df = matches.select(
        [
            "tournament_id",
            "match_id",
            "winner_team_id",
            "loser_team_id",
        ]
    )
    winner_counts_df = team_counts_df.rename(
        {"team_id": "winner_team_id", "n": "winner_count"}
    )
    loser_counts_df = team_counts_df.rename(
        {"team_id": "loser_team_id", "n": "loser_count"}
    )
    match_counts_df = (
        match_keys_df.join(
            winner_counts_df,
            on=["tournament_id", "match_id", "winner_team_id"],
            how="left",
        )
        .join(
            loser_counts_df,
            on=["tournament_id", "match_id", "loser_team_id"],
            how="left",
        )
        .with_columns(
            [
                pl.col("winner_count").fill_null(0),
                pl.col("loser_count").fill_null(0),
            ]
        )
    )

    valid_keys = (
        match_counts_df.filter(
            (pl.col("winner_count") == 4) & (pl.col("loser_count") == 4)
        )
        .select(["tournament_id", "match_id"])
        .unique()
    )

    return matches.join(
        valid_keys, on=["tournament_id", "match_id"], how="inner"
    )
