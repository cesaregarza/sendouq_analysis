import polars as pl

from rankings.analysis.utils.matches import filter_strict_4v4


def test_filter_strict_4v4_no_appearances_returns_all():
    matches = pl.DataFrame(
        {
            "tournament_id": [1],
            "match_id": [1],
            "winner_team_id": [10],
            "loser_team_id": [11],
        }
    )
    players = pl.DataFrame(
        {"tournament_id": [1], "team_id": [10], "user_id": [100]}
    )
    out = filter_strict_4v4(matches, players, pl.DataFrame([]))
    assert out.height == 1
