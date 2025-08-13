"""
Formatting functions for tournament ranking analysis.

This module provides functions to format DataFrames and results for display.
"""

from __future__ import annotations

from typing import Optional

import polars as pl


def format_top_rankings(
    rankings_df: pl.DataFrame,
    top_n: int = 10,
    title: str = "Top Rankings",
    show_score: bool = True,
    show_extra_cols: Optional[list[str]] = None,
) -> str:
    """
    Format top rankings for display.

    Parameters
    ----------
    rankings_df : pl.DataFrame
        DataFrame with rankings (must have at least username and score columns)
    top_n : int, optional
        Number of top entries to show
    title : str, optional
        Title for the output
    show_score : bool, optional
        Whether to show the score column
    show_extra_cols : list[str], optional
        Additional columns to display

    Returns
    -------
    str
        Formatted string output
    """
    output = [f"\n🏆 {title}:", "=" * 80]

    top_df = rankings_df.head(top_n)

    for i, row in enumerate(top_df.iter_rows(named=True), 1):
        # Basic info
        username = row.get("username", "Unknown")
        user_id = row.get("user_id", "N/A")

        line = f"{i:2d}. {username:<20} (ID: {user_id:<8})"

        # Add score if requested
        if show_score:
            score = row.get("score_normalized", row.get("player_rank_score", 0))
            line += f" Score: {score:6.1f}"

        # Add tournament count if available
        if "tournament_count" in row:
            line += f" ({row['tournament_count']} tournaments)"

        # Add extra columns
        if show_extra_cols:
            for col in show_extra_cols:
                if col in row:
                    line += f" {col}: {row[col]}"

        output.append(line)

    # Add summary statistics
    if rankings_df.height > 0:
        output.extend(
            [
                f"\n📊 Summary Statistics:",
                f"• Total ranked entries: {rankings_df.height}",
            ]
        )

        if "tournament_count" in rankings_df.columns:
            output.append(
                f"• Average tournaments per player: {rankings_df['tournament_count'].mean():.1f}"
            )
            output.append(
                f"• Most active player: {rankings_df['tournament_count'].max()} tournaments"
            )

    return "\n".join(output)


def display_player_rankings(
    player_summary: pl.DataFrame,
    top_n: int = 15,
    min_tournaments: Optional[int] = None,
) -> str:
    """
    Display player rankings in a formatted way.

    Parameters
    ----------
    player_summary : pl.DataFrame
        Player summary DataFrame from prepare_player_summary
    top_n : int, optional
        Number of top players to display
    min_tournaments : int, optional
        If provided, adds note about minimum tournament requirement

    Returns
    -------
    str
        Formatted player rankings
    """
    title = "Detailed Player Rankings"
    if min_tournaments:
        title += f" (min {min_tournaments} tournaments)"

    return format_top_rankings(
        player_summary, top_n=top_n, title=title, show_score=True
    )


def format_influential_matches(
    influential_matches: dict[str, pl.DataFrame],
    player_name: Optional[str] = None,
    tournament_names: Optional[dict[int, str]] = None,
    max_opponent_length: int = 50,
) -> str:
    """
    Format influential matches for display using flux-based metrics.

    Parameters
    ----------
    influential_matches : dict[str, pl.DataFrame]
        Dictionary with 'wins' and 'losses' DataFrames from get_most_influential_matches
    player_name : str, optional
        Name of the player for the title
    tournament_names : dict[int, str], optional
        Tournament ID to name mapping
    max_opponent_length : int, optional
        Maximum length for opponent names display

    Returns
    -------
    str
        Formatted match analysis with flux-based influence metrics
    """
    output = []

    if player_name:
        output.append(f"\n🎯 Most Influential Matches for {player_name}")
        output.append("=" * 70)

    # Format wins
    output.append(
        f"\n🏆 Top {influential_matches['wins'].height} Most Influential WINS:"
    )
    output.append("-" * 100)

    if not influential_matches["wins"].is_empty():
        for row in influential_matches["wins"].iter_rows(named=True):
            t_id = row["tournament_id"]
            t_name = (
                tournament_names.get(t_id, f"Tournament {t_id}")
                if tournament_names
                else f"Tournament {t_id}"
            )

            # Format the match date if available
            match_date_str = ""
            if "event_ts" in row and row["event_ts"] is not None:
                from datetime import datetime

                try:
                    match_date = datetime.fromtimestamp(row["event_ts"])
                    match_date_str = f" - {match_date.strftime('%Y-%m-%d')}"
                except:
                    pass

            output.append(
                f"\n{row['influence_rank']}. {t_name} (Match {row.get('match_id', 'Unknown')}){match_date_str}"
            )

            opponents = row.get("opponent_players", "Unknown")
            if len(opponents) > max_opponent_length:
                opponents = opponents[:max_opponent_length] + "..."
            output.append(f"   • Defeated: {opponents}")

            output.append(
                f"   • Score: {row.get('team1_score', '?')}-{row.get('team2_score', '?')} ({row.get('total_games', '?')} games)"
            )

            # Display flux-based metrics
            if "match_flux" in row:
                output.append(f"   • Match Flux: {row['match_flux']:.2e}")

                # Show share of flux if available
                if (
                    "share_incoming" in row
                    and row["share_incoming"] is not None
                ):
                    share_pct = row["share_incoming"] * 100
                    output.append(
                        f"   • Relative Importance (wins): {share_pct:.1f}%"
                    )
            else:
                # Fall back to old weight if flux not available
                output.append(
                    f"   • Match Weight: {row.get('w_m', row.get('match_weight', 0)):.6f}"
                )

            output.append(
                f"   • Tournament Influence: {row['tournament_influence']:.3f}"
            )
            output.append(f"   • Time Decay: {row['time_decay']:.3f}")
    else:
        output.append("   No wins found for this player")

    # Format losses
    output.append(
        f"\n💔 Top {influential_matches['losses'].height} Most Damaging LOSSES (worst results):"
    )
    output.append("-" * 100)

    if not influential_matches["losses"].is_empty():
        for row in influential_matches["losses"].iter_rows(named=True):
            t_id = row["tournament_id"]
            t_name = (
                tournament_names.get(t_id, f"Tournament {t_id}")
                if tournament_names
                else f"Tournament {t_id}"
            )

            # Format the match date if available
            match_date_str = ""
            if "event_ts" in row and row["event_ts"] is not None:
                from datetime import datetime

                try:
                    match_date = datetime.fromtimestamp(row["event_ts"])
                    match_date_str = f" - {match_date.strftime('%Y-%m-%d')}"
                except:
                    pass

            output.append(
                f"\n{row['influence_rank']}. {t_name} (Match {row.get('match_id', 'Unknown')}){match_date_str}"
            )

            opponents = row.get("opponent_players", "Unknown")
            if len(opponents) > max_opponent_length:
                opponents = opponents[:max_opponent_length] + "..."
            output.append(f"   • Lost to: {opponents}")

            output.append(
                f"   • Score: {row.get('team1_score', '?')}-{row.get('team2_score', '?')} ({row.get('total_games', '?')} games)"
            )

            # Display flux-based metrics
            if "match_flux" in row:
                output.append(f"   • Match Flux: {row['match_flux']:.2e}")

                # Show share of flux if available (outgoing for losses)
                if (
                    "share_outgoing" in row
                    and row["share_outgoing"] is not None
                ):
                    share_pct = row["share_outgoing"] * 100
                    output.append(
                        f"   • Damage to Rating: {share_pct:.1f}% of total loss impact"
                    )
            else:
                # Fall back to old weight if flux not available
                output.append(
                    f"   • Match Weight: {row.get('w_m', row.get('match_weight', 0)):.6f}"
                )

            output.append(
                f"   • Tournament Influence: {row['tournament_influence']:.3f}"
            )
            output.append(f"   • Time Decay: {row['time_decay']:.3f}")
    else:
        output.append("   No losses found for this player")

    # Summary statistics
    total_wins = (
        influential_matches["wins"].height
        if not influential_matches["wins"].is_empty()
        else 0
    )
    total_losses = (
        influential_matches["losses"].height
        if not influential_matches["losses"].is_empty()
        else 0
    )

    output.append(f"\n📈 Match Influence Summary:")
    output.append(f"   • Total influential wins shown: {total_wins}")
    output.append(f"   • Total influential losses shown: {total_losses}")

    if not influential_matches["wins"].is_empty():
        if "match_flux" in influential_matches["wins"].columns:
            avg_win_flux = influential_matches["wins"]["match_flux"].mean()
            output.append(f"   • Average win flux: {avg_win_flux:.2e}")
        else:
            avg_win_weight = (
                influential_matches["wins"].get_column("w_m").mean()
                if "w_m" in influential_matches["wins"].columns
                else influential_matches["wins"]["match_weight"].mean()
            )
            output.append(f"   • Average win influence: {avg_win_weight:.6f}")

    if not influential_matches["losses"].is_empty():
        if "match_flux" in influential_matches["losses"].columns:
            avg_loss_flux = influential_matches["losses"]["match_flux"].mean()
            output.append(f"   • Average loss flux: {avg_loss_flux:.2e}")
        else:
            avg_loss_weight = (
                influential_matches["losses"].get_column("w_m").mean()
                if "w_m" in influential_matches["losses"].columns
                else influential_matches["losses"]["match_weight"].mean()
            )
            output.append(f"   • Average loss influence: {avg_loss_weight:.6f}")

    return "\n".join(output)


def format_tournament_influence_summary(
    tournament_influence: dict[int, float],
    tournament_strength: Optional[pl.DataFrame] = None,
    tournament_names: Optional[dict[int, str]] = None,
    top_n: int = 10,
) -> str:
    """
    Format tournament influence and strength analysis.

    Parameters
    ----------
    tournament_influence : dict[int, float]
        Tournament influence values from RatingEngine
    tournament_strength : pl.DataFrame, optional
        Tournament strength DataFrame from RatingEngine
    tournament_names : dict[int, str], optional
        Tournament ID to name mapping
    top_n : int, optional
        Number of top tournaments to show

    Returns
    -------
    str
        Formatted tournament analysis
    """
    output = ["📊 Tournament Strength Analysis:", "=" * 50]

    # Show tournament influence
    influence_items = sorted(
        tournament_influence.items(), key=lambda x: x[1], reverse=True
    )

    output.append(
        f"\n🎖️  Top {min(top_n, len(influence_items))} Tournament Influences:"
    )
    for i, (tid, influence) in enumerate(influence_items[:top_n], 1):
        t_name = (
            tournament_names.get(tid, f"Tournament {tid}")
            if tournament_names
            else f"Tournament {tid}"
        )
        output.append(f"{i:2d}. {t_name:<30} Influence: {influence:.3f}")

    # Show tournament strength if available
    if tournament_strength is not None and not tournament_strength.is_empty():
        output.append(f"\n💪 Tournament Strength Analysis:")
        output.append("Top 5 by Retroactive Strength:")

        strength_df = tournament_strength.sort("strength", descending=True)
        top_strength = strength_df.head(5)

        for i, row in enumerate(top_strength.iter_rows(named=True), 1):
            tid = row["tournament_id"]
            t_name = (
                tournament_names.get(tid, f"Tournament {tid}")
                if tournament_names
                else f"T{tid}"
            )
            influence = row.get("influence", 0)
            strength = row.get("strength", 0)
            output.append(
                f"{i}. {t_name:<30} Influence: {influence:.3f}, Strength: {strength:.3f}"
            )

    # Summary
    output.append(f"\n📈 Algorithm Convergence:")
    output.append(
        f"✓ Final tournament influence range: {min(tournament_influence.values()):.3f} - {max(tournament_influence.values()):.3f}"
    )

    return "\n".join(output)
