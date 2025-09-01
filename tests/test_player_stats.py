import polars as pl

from rankings.cli import update as update_cli


def test_compute_player_stats_prefers_appearances():
    # Two tournaments; player A appears in both, player B in one
    matches = pl.DataFrame(
        {
            "tournament_id": [1, 2],
            # last_game_finished_at preferred when present
            "last_game_finished_at": [1_700_000_000.0, None],
            "match_created_at": [None, 1_700_100_000.0],
        }
    )

    players = pl.DataFrame(
        {
            "tournament_id": [1, 1, 2],
            "user_id": [10, 20, 10],
            "team_id": [101, 102, 201],
        }
    )

    # Appearances: A=10 plays in both tournaments, B=20 only in t1
    appearances = pl.DataFrame(
        {
            "tournament_id": [1, 2, 1],
            "match_id": [100, 200, 101],
            "user_id": [10, 10, 20],
        }
    )

    stats = update_cli._compute_player_stats(matches, players, appearances)
    assert set(stats.columns) == {
        "player_id",
        "tournament_count",
        "last_active_ms",
    }

    a = stats.filter(pl.col("player_id") == 10).to_dicts()[0]
    b = stats.filter(pl.col("player_id") == 20).to_dicts()[0]

    assert a["tournament_count"] == 2
    assert b["tournament_count"] == 1
    # last_active_ms uses the max per-tournament of last_game_finished_at or match_created_at, in ms
    assert a["last_active_ms"] == 1_700_100_000_000
    assert b["last_active_ms"] == 1_700_000_000_000
