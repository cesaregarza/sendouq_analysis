"""
Higher-level analysis functions for tournament rankings.

This module provides convenience functions that combine multiple utility
functions for common analysis tasks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rankings.analysis.engine import RatingEngine

import polars as pl

logger = logging.getLogger(__name__)

from rankings.analysis.engine import RatingEngine
from rankings.analysis.utils.formatting import (
    format_influential_matches,
    format_tournament_influence_summary,
)
from rankings.analysis.utils.matches import get_most_influential_matches
from rankings.analysis.utils.names import get_tournament_name_lookup

# Support for new engine
try:
    from rankings.algorithms import TTLEngine
except ImportError:
    TTLEngine = None


def _get_engine_values(engine, player_id):
    """
    Extract win_pr, loss_pr, and exposure values from either old or new engine.

    Returns:
        tuple: (win_pr, loss_pr, exposure) or (None, None, None) if not found
    """
    # Check if it's the new TTLEngine (by checking for last_result attribute)
    if hasattr(engine, "last_result") and hasattr(
        engine, "tournament_influence"
    ):
        if engine.last_result:
            result = engine.last_result
            if hasattr(result, "ids") and player_id in result.ids:
                idx = result.ids.index(player_id)
                win_pr = (
                    float(result.win_pagerank[idx])
                    if result.win_pagerank is not None
                    and idx < len(result.win_pagerank)
                    else None
                )
                loss_pr = (
                    float(result.loss_pagerank[idx])
                    if result.loss_pagerank is not None
                    and idx < len(result.loss_pagerank)
                    else None
                )
                exposure = (
                    float(result.exposure[idx])
                    if result.exposure is not None
                    and idx < len(result.exposure)
                    else None
                )
                return win_pr, loss_pr, exposure

    # Check if it's the old RatingEngine
    elif hasattr(engine, "active_players_") and hasattr(
        engine, "win_pagerank_"
    ):
        try:
            player_idx = engine.active_players_.index(player_id)
            win_pr = (
                float(engine.win_pagerank_[player_idx])
                if player_idx < len(engine.win_pagerank_)
                else None
            )
            loss_pr = (
                float(engine.loss_pagerank_[player_idx])
                if player_idx < len(engine.loss_pagerank_)
                else None
            )
            exposure = None
            if hasattr(engine, "exposure_teleport_") and player_idx < len(
                engine.exposure_teleport_
            ):
                exposure = float(engine.exposure_teleport_[player_idx])
            return win_pr, loss_pr, exposure
        except (ValueError, IndexError):
            pass

    return None, None, None


def analyze_player(
    player_id: int,
    engine: RatingEngine | None = None,
    rankings_df: pl.DataFrame | None = None,
    matches_df: pl.DataFrame | None = None,
    players_df: pl.DataFrame | None = None,
    tournament_data: list[dict] | None = None,
    top_n_matches: int = 10,
    print_output: bool = True,
    include_loo: bool = False,
    loo_matches: int = 5,
    score_multiplier: float = 1.0,
) -> dict:
    """
    Simplified player analysis function that only needs player_id.

    Automatically fetches player name and all necessary data. If engine
    is not provided, uses the data from rankings_df directly.

    Parameters
    ----------
    player_id : int
        The user_id of the player to analyze
    engine : RatingEngine, optional
        The RatingEngine instance with computed rankings
    rankings_df : pl.DataFrame, optional
        Pre-computed rankings DataFrame with player stats
    matches_df : pl.DataFrame, optional
        Matches DataFrame (required if using engine)
    players_df : pl.DataFrame, optional
        Players DataFrame (required if using engine)
    tournament_data : list[dict], optional
        Raw tournament data for name lookups
    top_n_matches : int, optional
        Number of top influential matches to analyze
    print_output : bool, optional
        Whether to print the formatted output
    include_loo : bool, optional
        Whether to include Leave-One-Out score contribution analysis for all influential matches
    loo_matches : int, optional
        Deprecated - LOO is now computed for all top_n_matches when include_loo=True
    score_multiplier : float, optional
        Multiplier for score values in output (e.g., 1000 to show as per-mille). Default is 1.0

    Returns
    -------
    dict
        Dictionary containing:
        - player_info: Basic player information
        - ranking_stats: Ranking statistics
        - match_stats: Match statistics
        - influential_matches: Top influential wins/losses (if engine provided)
        - loo_analysis: Leave-One-Out analysis results (if requested)
        - formatted_output: Formatted text output
    """
    result = {}

    # Get player name from players_df if available
    player_name = f"Player {player_id}"
    if players_df is not None:
        player_names = (
            players_df.filter(pl.col("user_id") == player_id)
            .select("username")
            .unique()
        )
        if len(player_names) > 0:
            player_name = player_names["username"][0]

    result["player_info"] = {
        "player_id": player_id,
        "player_name": player_name,
    }

    # Get ranking stats from rankings_df if available
    if rankings_df is not None:
        player_ranking = rankings_df.filter(pl.col("id") == player_id)

        if len(player_ranking) > 0:
            player_data = player_ranking.row(0, named=True)

            # Find rank position
            rankings_sorted = rankings_df.sort("player_rank", descending=True)
            rank_position = 0
            for i, row in enumerate(rankings_sorted.iter_rows(named=True), 1):
                if row["id"] == player_id:
                    rank_position = i
                    break

            # Extract all available stats
            ranking_stats = {
                "rank": rank_position,
                "total_players": len(rankings_df),
                "percentile": (1 - rank_position / len(rankings_df)) * 100,
                "score": player_data.get(
                    "score", player_data.get("player_rank", 0)
                ),
                "win_pr": player_data.get("win_pr", 0),
                "loss_pr": player_data.get("loss_pr", 0),
                "exposure": player_data.get("exposure", 0),
            }

            # If we have the engine, get the actual values from it
            if engine is not None:
                win_pr, loss_pr, exposure = _get_engine_values(
                    engine, player_id
                )
                if win_pr is not None:
                    ranking_stats["win_pr"] = win_pr
                if loss_pr is not None:
                    ranking_stats["loss_pr"] = loss_pr
                if exposure is not None:
                    ranking_stats["exposure"] = exposure

            # Calculate win/loss ratio if PageRanks available
            if ranking_stats["loss_pr"] > 0:
                ranking_stats["win_loss_ratio"] = (
                    ranking_stats["win_pr"] / ranking_stats["loss_pr"]
                )
            else:
                ranking_stats["win_loss_ratio"] = (
                    float("inf") if ranking_stats["win_pr"] > 0 else 0
                )

            result["ranking_stats"] = ranking_stats
        else:
            result["ranking_stats"] = {
                "rank": None,
                "error": f"Player {player_id} not found in rankings",
            }

    # Get match statistics if data available
    if matches_df is not None and players_df is not None:
        # Get player's teams
        player_teams = players_df.filter(pl.col("user_id") == player_id)

        # Count tournaments
        num_tournaments = player_teams["tournament_id"].n_unique()

        # Get all matches and tournament data
        total_wins = 0
        total_losses = 0
        tournament_stats = []

        # Group by tournament to get stats
        tournament_ids = player_teams["tournament_id"].unique().to_list()

        for tid in tournament_ids:
            # Get team_id for this tournament
            team_rows = player_teams.filter(pl.col("tournament_id") == tid)
            if len(team_rows) == 0:
                continue
            team_id = team_rows["team_id"][0]

            # Count wins
            wins = matches_df.filter(
                (pl.col("tournament_id") == tid)
                & (pl.col("winner_team_id") == team_id)
            )
            win_count = len(wins)
            total_wins += win_count

            # Count losses
            losses = matches_df.filter(
                (pl.col("tournament_id") == tid)
                & (pl.col("loser_team_id") == team_id)
            )
            loss_count = len(losses)
            total_losses += loss_count

            # Store tournament stats
            if win_count > 0 or loss_count > 0:
                tournament_stats.append(
                    {
                        "tournament_id": tid,
                        "wins": win_count,
                        "losses": loss_count,
                    }
                )

        total_matches = total_wins + total_losses
        win_rate = (
            (total_wins / total_matches * 100) if total_matches > 0 else 0
        )

        result["match_stats"] = {
            "tournaments_played": num_tournaments,
            "total_matches": total_matches,
            "wins": total_wins,
            "losses": total_losses,
            "win_rate": win_rate,
        }

        # Get last 3 tournaments
        if tournament_stats and tournament_data:
            # Get tournament start times for sorting
            tournament_dates = {}
            for t in tournament_data:
                if "id" in t:
                    tournament_dates[t["id"]] = t.get("startTime", 0)

            # Sort by tournament start time (most recent first)
            tournament_stats.sort(
                key=lambda x: tournament_dates.get(
                    x["tournament_id"], x["tournament_id"]
                ),
                reverse=True,
            )
            result["last_tournaments"] = tournament_stats[:3]
        elif tournament_stats:
            # If no tournament data, sort by tournament_id (most recent usually have higher ids)
            tournament_stats.sort(
                key=lambda x: x["tournament_id"], reverse=True
            )
            result["last_tournaments"] = tournament_stats[:3]

    # Get influential matches if engine provided
    if engine is not None and matches_df is not None and players_df is not None:
        try:
            influential = get_most_influential_matches(
                player_id=player_id,
                matches_df=matches_df,
                players_df=players_df,
                engine=engine,
                top_n=top_n_matches,
            )
            result["influential_matches"] = influential

            # Get tournament names if available
            tournament_names = {}
            if tournament_data:
                tournament_names = get_tournament_name_lookup(tournament_data)

            # If LOO is requested and engine supports it, compute LOO impacts for ALL influential matches
            loo_impacts = {}
            if include_loo and hasattr(engine, "analyze_match_impact"):
                try:
                    logger.info(
                        f"Computing LOO impacts for top {top_n_matches} influential matches..."
                    )
                    # Compute LOO for ALL the influential matches we're displaying
                    all_match_ids = []
                    if (
                        "wins" in influential
                        and not influential["wins"].is_empty()
                    ):
                        all_match_ids.extend(
                            influential["wins"]["match_id"].to_list()
                        )
                    if (
                        "losses" in influential
                        and not influential["losses"].is_empty()
                    ):
                        all_match_ids.extend(
                            influential["losses"]["match_id"].to_list()
                        )

                    # Compute for all matches that will be displayed (not just loo_matches)
                    for i, match_id in enumerate(all_match_ids, 1):
                        logger.debug(
                            f"  Computing LOO for match {i}/{len(all_match_ids)}: {match_id}"
                        )
                        impact = engine.analyze_match_impact(
                            match_id, player_id
                        )
                        if impact.get("ok"):
                            # Store the actual contribution (negative of removal delta) with multiplier
                            loo_impacts[match_id] = (
                                -impact["delta"]["score"] * score_multiplier
                            )

                    logger.info(
                        f"Computed LOO impacts for {len(loo_impacts)} matches"
                    )
                except Exception as e:
                    logger.warning(f"Could not compute LOO impacts: {e}")

            # Re-sort by LOO impact if available
            if loo_impacts:
                # Re-sort wins and losses by absolute LOO impact
                if "wins" in influential and not influential["wins"].is_empty():
                    wins_df = influential["wins"]
                    # Add LOO impact column
                    wins_with_loo = wins_df.with_columns(
                        pl.col("match_id")
                        .map_elements(
                            lambda x: abs(loo_impacts.get(x, 0.0)),
                            return_dtype=pl.Float64,
                        )
                        .alias("abs_loo_impact")
                    )
                    # Sort by absolute LOO impact
                    influential["wins"] = wins_with_loo.sort(
                        "abs_loo_impact", descending=True
                    )
                    # Update influence_rank based on new sorting
                    influential["wins"] = influential["wins"].with_columns(
                        pl.arange(1, len(influential["wins"]) + 1).alias(
                            "influence_rank"
                        )
                    )

                if (
                    "losses" in influential
                    and not influential["losses"].is_empty()
                ):
                    losses_df = influential["losses"]
                    # Add LOO impact column
                    losses_with_loo = losses_df.with_columns(
                        pl.col("match_id")
                        .map_elements(
                            lambda x: abs(loo_impacts.get(x, 0.0)),
                            return_dtype=pl.Float64,
                        )
                        .alias("abs_loo_impact")
                    )
                    # Sort by absolute LOO impact
                    influential["losses"] = losses_with_loo.sort(
                        "abs_loo_impact", descending=True
                    )
                    # Update influence_rank based on new sorting
                    influential["losses"] = influential["losses"].with_columns(
                        pl.arange(1, len(influential["losses"]) + 1).alias(
                            "influence_rank"
                        )
                    )

                logger.info(
                    "Re-sorted influential matches by LOO score contribution"
                )

            # Format output with LOO impacts if available
            formatted_output = format_influential_matches(
                influential,
                player_name=player_name,
                tournament_names=tournament_names,
                loo_impacts=loo_impacts,
            )
            result["formatted_output"] = formatted_output
        except Exception as e:
            logger.error(f"Failed to get influential matches: {e}")
            result["influential_matches"] = {"error": str(e)}

    # Note: LOO analysis is now integrated into influential matches above when include_loo=True

    # Create summary output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"PLAYER ANALYSIS: {player_name} (ID: {player_id})")
    output_lines.append("=" * 80)

    if "ranking_stats" in result and result["ranking_stats"].get("rank"):
        rs = result["ranking_stats"]
        output_lines.append("\nRANKING STATS:")
        output_lines.append(
            f"  Rank: #{int(rs['rank'])} out of {int(rs['total_players'])} (Top {rs['percentile']:.1f}%)"
        )
        output_lines.append(f"  Score: {rs['score'] * score_multiplier:.4f}")

    if "match_stats" in result:
        ms = result["match_stats"]
        output_lines.append("\nMATCH STATS:")
        output_lines.append(f"  Tournaments: {int(ms['tournaments_played'])}")
        output_lines.append(f"  Total Matches: {int(ms['total_matches'])}")
        output_lines.append(
            f"  Record: {int(ms['wins'])}W - {int(ms['losses'])}L"
        )
        output_lines.append(f"  Win Rate: {ms['win_rate']:.1f}%")

    if "last_tournaments" in result:
        output_lines.append("\nLAST 3 TOURNAMENTS:")
        tournament_names = {}
        if tournament_data:
            tournament_names = get_tournament_name_lookup(tournament_data)

        for i, t_stat in enumerate(result["last_tournaments"], 1):
            tid = t_stat["tournament_id"]
            t_name = tournament_names.get(tid, f"Tournament {tid}")
            output_lines.append(f"  {i}. {t_name}")
            output_lines.append(
                f"     Record: {int(t_stat['wins'])}W - {int(t_stat['losses'])}L"
            )

    if "formatted_output" in result:
        output_lines.append("\n" + result["formatted_output"])

    # LOO analysis is now integrated into influential matches display above

    summary_output = "\n".join(output_lines)
    result["summary"] = summary_output

    if print_output:
        print(summary_output)

    return result


def analyze_player_performance(
    player_id: int,
    player_name: str,
    engine: RatingEngine,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    tournament_data: list[dict],
    top_n_matches: int = 10,
    print_output: bool = True,
) -> dict:
    """
    Comprehensive analysis of a player's performance and influential matches.

    Parameters
    ----------
    player_id : int
        The user_id of the player to analyze
    player_name : str
        The username of the player
    engine : RatingEngine
        The RatingEngine instance with computed rankings
    matches_df : pl.DataFrame
        Matches DataFrame
    players_df : pl.DataFrame
        Players DataFrame
    tournament_data : list[dict]
        Raw tournament data
    top_n_matches : int, optional
        Number of top influential matches to analyze
    print_output : bool, optional
        Whether to print the formatted output

    Returns
    -------
    dict
        Dictionary containing:
        - influential_matches: dict with wins/losses DataFrames
        - tournament_names: dict of tournament ID to name mapping
        - formatted_output: str with formatted analysis
    """
    # Get tournament names
    tournament_names = get_tournament_name_lookup(tournament_data)

    # Get influential matches
    influential_matches = get_most_influential_matches(
        player_id=player_id,
        matches_df=matches_df,
        players_df=players_df,
        engine=engine,
        top_n=top_n_matches,
    )

    # Format the output
    formatted_output = format_influential_matches(
        influential_matches,
        player_name=player_name,
        tournament_names=tournament_names,
    )

    if print_output:
        print(formatted_output)

    return {
        "influential_matches": influential_matches,
        "tournament_names": tournament_names,
        "formatted_output": formatted_output,
    }


def generate_tournament_report(
    engine: "RatingEngine",
    tournament_data: list[dict],
    top_n: int = 10,
    print_output: bool = True,
) -> dict:
    """
    Generate a comprehensive tournament strength report.

    Parameters
    ----------
    engine : RatingEngine
        The RatingEngine instance with computed rankings
    tournament_data : list[dict]
        Raw tournament data
    top_n : int, optional
        Number of top tournaments to show
    print_output : bool, optional
        Whether to print the formatted output

    Returns
    -------
    dict
        Dictionary containing:
        - tournament_influence: dict of tournament influences
        - tournament_strength: DataFrame of tournament strengths
        - tournament_names: dict of tournament ID to name mapping
        - formatted_output: str with formatted analysis
    """
    # Get tournament names
    tournament_names = get_tournament_name_lookup(tournament_data)

    # Get tournament influence and strength from engine
    tournament_influence = None
    tournament_strength = None

    # Check if it's the new TTLEngine (by checking for specific attributes)
    if hasattr(engine, "last_result") and hasattr(
        engine, "tournament_influence"
    ):
        tournament_influence = engine.tournament_influence
        # New engine may not have tournament_strength
        tournament_strength = getattr(engine, "tournament_strength", None)
    else:
        # Old RatingEngine
        tournament_influence = getattr(engine, "tournament_influence", None)
        tournament_strength = getattr(engine, "tournament_strength", None)

    # Format the output
    formatted_output = format_tournament_influence_summary(
        tournament_influence=tournament_influence,
        tournament_strength=tournament_strength,
        tournament_names=tournament_names,
        top_n=top_n,
    )

    if print_output:
        print(formatted_output)

    return {
        "tournament_influence": tournament_influence,
        "tournament_strength": tournament_strength,
        "tournament_names": tournament_names,
        "formatted_output": formatted_output,
    }


def compare_player_performances(
    player_ids: list[int],
    engine: RatingEngine,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    player_summary: pl.DataFrame,
    tournament_data: list[dict],
    top_n_matches: int = 5,
) -> dict:
    """
    Compare performance metrics for multiple players.

    Parameters
    ----------
    player_ids : list[int]
        List of player IDs to compare
    engine : RatingEngine
        The RatingEngine instance
    matches_df : pl.DataFrame
        Matches DataFrame
    players_df : pl.DataFrame
        Players DataFrame
    player_summary : pl.DataFrame
        Player summary with rankings
    tournament_data : list[dict]
        Raw tournament data
    top_n_matches : int, optional
        Number of top matches to show per player

    Returns
    -------
    dict
        Dictionary with player_id as keys and analysis results as values
    """
    results = {}
    tournament_names = get_tournament_name_lookup(tournament_data)

    for player_id in player_ids:
        # Get player info
        player_info = player_summary.filter(
            pl.col("user_id") == player_id
        ).head(1)

        if player_info.is_empty():
            continue

        player_name = player_info["username"][0]
        player_rank = player_info["player_rank"][0]
        player_score = player_info["score_normalized"][0]

        # Get influential matches
        influential_matches = get_most_influential_matches(
            player_id=player_id,
            matches_df=matches_df,
            players_df=players_df,
            engine=engine,
            top_n=top_n_matches,
        )

        # Calculate win/loss statistics
        total_wins = influential_matches["wins"].height
        total_losses = influential_matches["losses"].height

        avg_win_weight = (
            influential_matches["wins"]["match_weight"].mean()
            if total_wins > 0
            else 0
        )
        avg_loss_weight = (
            influential_matches["losses"]["match_weight"].mean()
            if total_losses > 0
            else 0
        )

        results[player_id] = {
            "name": player_name,
            "rank": player_rank,
            "score": player_score,
            "influential_wins": total_wins,
            "influential_losses": total_losses,
            "avg_win_weight": avg_win_weight,
            "avg_loss_weight": avg_loss_weight,
            "win_loss_ratio": total_wins / max(total_losses, 1),
            "influential_matches": influential_matches,
        }

    return results
