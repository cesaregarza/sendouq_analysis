import polars as pl

from rankings.scraping.storage import (
    extract_appearances_from_players_payload as extract,
)
from rankings.scraping.storage import load_match_appearances


def test_extract_appearances_legacy_team_based():
    tournament_id = 77
    payload = {
        "matches": [
            {
                "matchId": 1,
                "teams": [
                    {
                        "teamId": 10,
                        "players": [{"userId": 1001}, {"userId": 1002}],
                    },
                    {"teamId": 11, "players": [{"userId": 1003}]},
                ],
            }
        ]
    }

    rows = extract(tournament_id, payload)
    df = pl.DataFrame(rows)
    # Basic shape and content
    assert set(df.columns) >= {"tournament_id", "match_id", "user_id"}
    assert df.height == 3
    assert (
        df.filter(
            (pl.col("match_id") == 1) & (pl.col("user_id") == 1001)
        ).height
        == 1
    )
    # team_id present in legacy format
    assert "team_id" in df.columns
    assert df.filter(pl.col("team_id") == 10).height == 2


def test_extract_appearances_player_based_list():
    tournament_id = 88
    payload = [
        {"userId": 2001, "matchIds": [1, 2]},
        {"userId": 2002, "matchIds": [2]},
    ]

    rows = extract(tournament_id, payload)
    df = pl.DataFrame(rows)
    # One row per (user, match)
    assert df.height == 3
    assert (
        df.filter(
            (pl.col("user_id") == 2001) & (pl.col("match_id") == 1)
        ).height
        == 1
    )
    # team_id may be absent/None in new format
    if "team_id" in df.columns:
        assert df.select(pl.col("team_id").is_null().any()).item() is True
