"""Tournament filtering utilities for analyzing ranked tournaments."""

from typing import Dict, List, Optional

import polars as pl


def filter_ranked_tournaments(
    tables: Dict[str, pl.DataFrame],
    min_teams: int = 4,
    min_players: int = 16,
    min_matches: int = 10,
    exclude_test_patterns: bool = True,
    exclusions: Optional[List[int]] = None,
) -> Dict[str, pl.DataFrame]:
    """
    Filter tournament data to include only ranked tournaments.

    This function filters tournaments based on ranking settings and applies
    the filter to related tables (matches, players, teams).

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

    Returns:
        Dictionary with filtered tables containing only ranked tournament data.
        Keys match the input tables dict.

    Example:
        >>> tables = parse_tournaments_data(tournaments)
        >>> ranked_tables = filter_ranked_tournaments(tables, exclusions=[1902, 1903])
        >>> print(f"Ranked matches: {ranked_tables['matches'].height:,}")
    """
    if "tournaments" not in tables:
        raise ValueError("tables must contain 'tournaments' key")

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

    # Apply filter to tournaments
    filtered_by_criteria = (
        tables["tournaments"]
        .filter(pl.all_horizontal(filter_conditions))
        .select("tournament_id")
    )

    # Add exclusions if provided (tournaments to always include)
    if exclusions:
        excluded_tournaments = (
            tables["tournaments"]
            .filter(pl.col("tournament_id").is_in(exclusions))
            .select("tournament_id")
        )
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
            result[table_name] = tables[table_name].join(
                ranked_tournaments, on="tournament_id", how="inner"
            )

    return result


def get_ranked_tournament_ids(
    tournaments_df: pl.DataFrame,
    min_teams: int = 4,
    min_players: int = 16,
    min_matches: int = 10,
    exclude_test_patterns: bool = True,
    exclusions: Optional[List[int]] = None,
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
    tables: Dict[str, pl.DataFrame],
    min_teams: int = 4,
    min_players: int = 16,
    min_matches: int = 10,
    exclude_test_patterns: bool = True,
    exclusions: Optional[List[int]] = None,
) -> Dict[str, int]:
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
