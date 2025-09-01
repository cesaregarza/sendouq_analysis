import polars as pl

from rankings.analysis.utils.matches import filter_strict_4v4


def test_filter_strict_4v4_infers_teams_when_missing():
    # Two matches in same tournament
    matches = pl.DataFrame(
        {
            "tournament_id": [1, 1],
            "match_id": [1, 2],
            "winner_team_id": [10, 10],
            "loser_team_id": [11, 11],
        }
    )

    # Full rosters (4 per team)
    players = pl.DataFrame(
        {
            "tournament_id": [1] * 8,
            "team_id": [10, 10, 10, 10, 11, 11, 11, 11],
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    # Appearances: match 1 is 4v4; match 2 is 4v3
    appearances = pl.DataFrame(
        {
            "tournament_id": [1] * (8 + 7),
            "match_id": [1] * 8 + [2] * 7,
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7],
        }
    )

    filtered = filter_strict_4v4(matches, players, appearances)
    assert filtered.height == 1
    assert filtered.select("match_id").to_series().to_list() == [1]
