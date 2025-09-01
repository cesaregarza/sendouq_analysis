import polars as pl

from rankings.scraping.storage import (
    extract_appearances_from_players_payload as extract,
)


def test_extract_appearances_supports_userIds_and_users_alt_keys():
    tournament_id = 101
    payload = {
        "matches": [
            {
                "matchId": 5,
                "teams": [
                    {"teamId": 1, "userIds": [101, 102]},
                    {
                        "teamId": 2,
                        "users": [{"userId": 201}, {"userId": 202}],
                    },
                ],
            }
        ]
    }

    rows = extract(tournament_id, payload)
    df = pl.DataFrame(rows)
    assert df.height == 4
    assert set(df.columns) >= {
        "tournament_id",
        "match_id",
        "user_id",
        "team_id",
    }
    assert set(df.filter(pl.col("team_id") == 1)["user_id"].to_list()) == {
        101,
        102,
    }
    assert set(df.filter(pl.col("team_id") == 2)["user_id"].to_list()) == {
        201,
        202,
    }
