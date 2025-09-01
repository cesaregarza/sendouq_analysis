import polars as pl

from rankings.scraping.storage import (
    extract_appearances_from_players_payload as extract,
)


def test_extract_appearances_handles_wrapped_payload():
    tournament_id = 55
    payload = {
        "data": {
            "result": {
                "matches": [
                    {
                        "matchId": 9,
                        "teams": [
                            {"teamId": 1, "players": [{"userId": 10}]},
                            {"teamId": 2, "players": [{"userId": 20}]},
                        ],
                    }
                ]
            }
        }
    }
    rows = extract(tournament_id, payload)
    df = pl.DataFrame(rows)
    assert df.height == 2
    assert set(df.columns) >= {"tournament_id", "match_id", "user_id"}
    assert set(df["user_id"].to_list()) == {10, 20}
