"""
Higher-level analysis functions for tournament rankings.

This module provides convenience functions that combine multiple utility
functions for common analysis tasks.
"""

from __future__ import annotations

import logging
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

from rankings.analysis.engine import RatingEngine
from rankings.analysis.utils.formatting import (
    format_influential_matches,
    format_tournament_influence_summary,
)
from rankings.analysis.utils.matches import get_most_influential_matches
from rankings.analysis.utils.names import get_tournament_name_lookup


def analyze_player(
    player_id: int,
    engine: Optional[RatingEngine] = None,
    rankings_df: Optional[pl.DataFrame] = None,
    matches_df: Optional[pl.DataFrame] = None,
    players_df: Optional[pl.DataFrame] = None,
    tournament_data: Optional[list[dict]] = None,
    top_n_matches: int = 10,
    print_output: bool = True,
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

    Returns
    -------
    dict
        Dictionary containing:
        - player_info: Basic player information
        - ranking_stats: Ranking statistics
        - match_stats: Match statistics
        - influential_matches: Top influential wins/losses (if engine provided)
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
            if (
                engine is not None
                and hasattr(engine, "win_pagerank_")
                and hasattr(engine, "loss_pagerank_")
            ):
                if hasattr(engine, "active_players_"):
                    try:
                        player_idx = engine.active_players_.index(player_id)
                        if player_idx < len(engine.win_pagerank_):
                            ranking_stats["win_pr"] = float(
                                engine.win_pagerank_[player_idx]
                            )
                        if player_idx < len(engine.loss_pagerank_):
                            ranking_stats["loss_pr"] = float(
                                engine.loss_pagerank_[player_idx]
                            )
                        if hasattr(
                            engine, "exposure_teleport_"
                        ) and player_idx < len(engine.exposure_teleport_):
                            ranking_stats["exposure"] = float(
                                engine.exposure_teleport_[player_idx]
                            )
                    except (ValueError, IndexError):
                        pass  # Player not found in active players

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

            # Format output
            formatted_output = format_influential_matches(
                influential,
                player_name=player_name,
                tournament_names=tournament_names,
            )
            result["formatted_output"] = formatted_output
        except Exception as e:
            logger.error(f"Failed to get influential matches: {e}")
            result["influential_matches"] = {"error": str(e)}

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
        output_lines.append(f"  Score: {rs['score']:.4f}")

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

    # Format the output
    formatted_output = format_tournament_influence_summary(
        tournament_influence=engine.tournament_influence,
        tournament_strength=engine.tournament_strength,
        tournament_names=tournament_names,
        top_n=top_n,
    )

    if print_output:
        print(formatted_output)

    return {
        "tournament_influence": engine.tournament_influence,
        "tournament_strength": engine.tournament_strength,
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
