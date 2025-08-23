"""Tournament filtering utilities for analyzing ranked tournaments."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl


def filter_ranked_tournaments(
    tables: dict[str, pl.DataFrame],
    min_teams: int = 4,
    min_players: int = 16,
    min_matches: int = 10,
    exclude_test_patterns: bool = True,
    exclusions: list[int] | None = None,
    tournament_before_id: int | None = None,
    buffer_weeks: float = 0,
) -> dict[str, pl.DataFrame]:
    """
    Filter tournament data to include only ranked tournaments and clean data issues.

    This function filters tournaments based on ranking settings, applies
    the filter to related tables (matches, players, teams), and also
    performs data cleaning by removing any duplicate entries that may
    exist in the source data.

    Args:
        tables: Dictionary containing tournament data tables with keys:
            - "tournaments": Tournament metadata
            - "matches": Match data
            - "players": Player participation data
            - "teams": Team data (optional)
        min_teams: Minimum number of teams required (default: 4)
        min_players: Minimum number of participating players required (default: 16)
        min_matches: Minimum number of matches required (default: 10)
        exclude_test_patterns: Whether to exclude test/debug/demo tournaments (default: True)
        exclusions: List of tournament IDs to always include regardless of other criteria (default: None)
        tournament_before_id: If provided, only include tournaments that occurred before this tournament ID
            chronologically (default: None)
        buffer_weeks: Number of weeks to buffer before the reference tournament. For example, if buffer_weeks=2,
            tournaments within 2 weeks before the reference tournament will be excluded (default: 0)

    Returns:
        Dictionary with filtered tables containing only ranked tournament data.
        Keys match the input tables dict.

    Example:
        >>> tables = parse_tournaments_data(tournaments)
        >>> ranked_tables = filter_ranked_tournaments(tables, exclusions=[1902, 1903])
        >>> print(f"Ranked matches: {ranked_tables['matches'].height:,}")

        >>> # Get all tournaments before tournament 2000 with a 2-week buffer
        >>> ranked_tables = filter_ranked_tournaments(
        ...     tables,
        ...     tournament_before_id=2000,
        ...     buffer_weeks=2
        ... )
    """
    if "tournaments" not in tables:
        raise ValueError("tables must contain 'tournaments' key")

    tournaments_df = tables["tournaments"]

    # Apply chronological filtering if tournament_before_id is provided
    if tournament_before_id is not None:
        # Get the reference tournament's start date
        ref_tournament = tournaments_df.filter(
            pl.col("tournament_id") == tournament_before_id
        )

        if ref_tournament.height == 0:
            raise ValueError(f"Tournament ID {tournament_before_id} not found")

        ref_start_timestamp = (
            ref_tournament.select("start_time").unique().item()
        )

        # Convert timestamp to datetime for buffer calculation
        ref_start_date = datetime.fromtimestamp(ref_start_timestamp)

        # Apply buffer if specified
        cutoff_date = ref_start_date
        if buffer_weeks > 0:
            cutoff_date = ref_start_date - timedelta(weeks=buffer_weeks)

        # Convert back to timestamp for filtering
        cutoff_timestamp = int(cutoff_date.timestamp())

        # Filter tournaments to only include those before the cutoff date
        tournaments_df = tournaments_df.filter(
            pl.col("start_time") < cutoff_timestamp
        )

    # Build filter conditions
    filter_conditions = [
        pl.col("settings_is_ranked"),
        pl.col("team_count") >= min_teams,
        pl.col("participated_users_count") >= min_players,
        pl.col("match_count") >= min_matches,
    ]

    if exclude_test_patterns:
        filter_conditions.append(
            ~pl.col("name").str.contains("(?i)test|debug|demo|practice")
        )

    # Apply filter to tournaments (use the potentially filtered tournaments_df)
    filtered_by_criteria = tournaments_df.filter(
        pl.all_horizontal(filter_conditions)
    ).select("tournament_id")

    # Add exclusions if provided (tournaments to always include)
    # Note: exclusions should also respect the chronological filter
    if exclusions:
        excluded_tournaments = tournaments_df.filter(
            pl.col("tournament_id").is_in(exclusions)
        ).select("tournament_id")
        # Combine filtered tournaments with exclusions
        ranked_tournaments = pl.concat(
            [filtered_by_criteria, excluded_tournaments]
        ).unique()
    else:
        ranked_tournaments = filtered_by_criteria

    # Create result dictionary with filtered tables
    result = {}

    # Always include filtered tournaments
    result["tournaments"] = tables["tournaments"].join(
        ranked_tournaments, on="tournament_id", how="inner"
    )

    # Filter other tables if they exist
    for table_name in ["matches", "players", "teams"]:
        if table_name in tables:
            filtered_table = tables[table_name].join(
                ranked_tournaments, on="tournament_id", how="inner"
            )

            # Additional data cleaning - remove any duplicates that may still exist
            # Note: The parser should handle this, but we add this as a safety measure
            if table_name == "matches":
                # Keep only unique match_ids (first occurrence)
                filtered_table = filtered_table.unique(subset=["match_id"])

            elif table_name == "players":
                # Keep first occurrence of each player-tournament-team combination
                # This handles any remaining duplicate roster entries
                filtered_table = filtered_table.unique(
                    subset=["user_id", "tournament_id", "team_id"]
                )

            result[table_name] = filtered_table

    return result


def get_ranked_tournament_ids(
    tournaments_df: pl.DataFrame,
    min_teams: int = 4,
    min_players: int = 16,
    min_matches: int = 10,
    exclude_test_patterns: bool = True,
    exclusions: list[int] | None = None,
) -> pl.DataFrame:
    """
    Get tournament IDs for ranked tournaments only.

    Args:
        tournaments_df: DataFrame containing tournament metadata
        min_teams: Minimum number of teams required (default: 4)
        min_players: Minimum number of participating players required (default: 16)
        min_matches: Minimum number of matches required (default: 10)
        exclude_test_patterns: Whether to exclude test/debug/demo tournaments (default: True)
        exclusions: List of tournament IDs to always include regardless of other criteria (default: None)

    Returns:
        DataFrame with single column "tournament_id" containing ranked tournament IDs.

    Example:
        >>> ranked_ids = get_ranked_tournament_ids(tables["tournaments"], exclusions=[1902])
        >>> matches_df = matches_df.join(ranked_ids, on="tournament_id", how="inner")
    """
    # Build filter conditions
    filter_conditions = [
        pl.col("settings_is_ranked"),
        pl.col("team_count") >= min_teams,
        pl.col("participated_users_count") >= min_players,
        pl.col("match_count") >= min_matches,
    ]

    if exclude_test_patterns:
        filter_conditions.append(
            ~pl.col("name").str.contains("(?i)test|debug|demo|practice")
        )

    filtered_by_criteria = tournaments_df.filter(
        pl.all_horizontal(filter_conditions)
    ).select("tournament_id")

    # Add exclusions if provided (tournaments to always include)
    if exclusions:
        excluded_tournaments = tournaments_df.filter(
            pl.col("tournament_id").is_in(exclusions)
        ).select("tournament_id")
        # Combine filtered tournaments with exclusions
        return pl.concat([filtered_by_criteria, excluded_tournaments]).unique()
    else:
        return filtered_by_criteria


def apply_ranked_filter(
    df: pl.DataFrame,
    ranked_tournament_ids: pl.DataFrame,
    join_column: str = "tournament_id",
) -> pl.DataFrame:
    """
    Apply ranked tournament filter to any dataframe with tournament IDs.

    Args:
        df: DataFrame to filter
        ranked_tournament_ids: DataFrame with ranked tournament IDs (single column)
        join_column: Column name to join on (default: "tournament_id")

    Returns:
        Filtered DataFrame containing only rows from ranked tournaments.

    Example:
        >>> ranked_ids = get_ranked_tournament_ids(tables["tournaments"])
        >>> ranked_matches = apply_ranked_filter(matches_df, ranked_ids)
    """
    return df.join(ranked_tournament_ids, on=join_column, how="inner")


def get_ranked_stats(
    tables: dict[str, pl.DataFrame],
    min_teams: int = 4,
    min_players: int = 16,
    min_matches: int = 10,
    exclude_test_patterns: bool = True,
    exclusions: list[int] | None = None,
) -> dict[str, int]:
    """
    Get statistics about ranked vs total tournaments.

    Args:
        tables: Dictionary containing tournament data tables
        min_teams: Minimum number of teams required (default: 4)
        min_players: Minimum number of participating players required (default: 16)
        min_matches: Minimum number of matches required (default: 10)
        exclude_test_patterns: Whether to exclude test/debug/demo tournaments (default: True)
        exclusions: List of tournament IDs to always include regardless of other criteria (default: None)

    Returns:
        Dictionary with statistics about the data before and after filtering.

    Example:
        >>> stats = get_ranked_stats(tables, exclusions=[1902])
        >>> print(f"Ranked tournaments: {stats['ranked_tournaments']:,} / {stats['total_tournaments']:,}")
    """
    # Get original counts
    stats = {
        "total_tournaments": tables["tournaments"].height,
        "total_matches": tables.get("matches", pl.DataFrame()).height,
        "total_players": tables.get("players", pl.DataFrame()).height,
        "total_teams": tables.get("teams", pl.DataFrame()).height,
    }

    # Get filtered data
    ranked_tables = filter_ranked_tournaments(
        tables,
        min_teams,
        min_players,
        min_matches,
        exclude_test_patterns,
        exclusions,
    )

    # Add ranked counts
    stats.update(
        {
            "ranked_tournaments": ranked_tables["tournaments"].height,
            "ranked_matches": ranked_tables.get(
                "matches", pl.DataFrame()
            ).height,
            "ranked_players": ranked_tables.get(
                "players", pl.DataFrame()
            ).height,
            "ranked_teams": ranked_tables.get("teams", pl.DataFrame()).height,
        }
    )

    # Calculate percentages
    if stats["total_tournaments"] > 0:
        stats["ranked_tournament_pct"] = (
            100.0 * stats["ranked_tournaments"] / stats["total_tournaments"]
        )
    if stats["total_matches"] > 0:
        stats["ranked_match_pct"] = (
            100.0 * stats["ranked_matches"] / stats["total_matches"]
        )

    return stats
