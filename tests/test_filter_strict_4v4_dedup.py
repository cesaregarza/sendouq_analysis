import polars as pl

from rankings.analysis.utils.matches import filter_strict_4v4


def test_filter_strict_4v4_deduplicates_appearances_before_counting():
    matches = pl.DataFrame(
        {
            "tournament_id": [42],
            "match_id": [1],
            "winner_team_id": [10],
            "loser_team_id": [11],
        }
    )
    players = pl.DataFrame(
        {
            "tournament_id": [42] * 8,
            "team_id": [10, 10, 10, 10, 11, 11, 11, 11],
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    # Appearances with deliberate duplicates for some users on both sides
    appearances = pl.DataFrame(
        {
            "tournament_id": [42] * 12,
            "match_id": [1] * 12,
            "user_id": [1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 8],
        }
    )

    filtered = filter_strict_4v4(matches, players, appearances)
    # Despite duplicates, counts should resolve to 4 vs 4 and keep the match
    assert filtered.height == 1
    assert filtered.select("match_id").to_series().to_list() == [1]
