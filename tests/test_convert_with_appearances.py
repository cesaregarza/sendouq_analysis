import polars as pl

from rankings.core.convert import convert_matches_dataframe


def test_convert_matches_uses_appearances_and_infers_team_from_roster():
    # One match, two teams
    matches = pl.DataFrame(
        {
            "match_id": [1],
            "tournament_id": [999],
            "winner_team_id": [10],
            "loser_team_id": [11],
            "last_game_finished_at": [1_700_000_000.0],
        }
    )

    # Roster contains more context; inference happens from (tournament_id, user_id) -> team_id
    players = pl.DataFrame(
        {
            "tournament_id": [999] * 8,
            "team_id": [10, 10, 10, 10, 11, 11, 11, 11],
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    # Appearances list participants without team_id
    appearances = pl.DataFrame(
        {
            "tournament_id": [999] * 8,
            "match_id": [1] * 8,
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    df = convert_matches_dataframe(
        matches,
        players,
        tournament_influence={},
        now_timestamp=1_700_000_100.0,
        decay_rate=0.0,
        appearances=appearances,
    )

    assert df.height == 1
    winners = set(df.select("winners").to_series()[0])
    losers = set(df.select("losers").to_series()[0])
    assert winners == {1, 2, 3, 4}
    assert losers == {5, 6, 7, 8}
