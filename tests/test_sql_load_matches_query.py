import polars as pl

from rankings.sql.load import load_matches_df


def test_load_matches_df_builds_join_and_time_filters(monkeypatch):
    captured = {}

    def fake_read_sql(engine, sql, params=None):  # noqa: ARG001
        captured["sql"] = sql
        captured["params"] = params or {}
        # Return minimal columns that the downstream casting logic expects
        return pl.DataFrame(
            {
                "match_id": [1],
                "tournament_id": [1],
                "stage_id": [None],
                "group_id": [None],
                "round_id": [None],
                "team1_id": [10],
                "team1_position": [None],
                "team1_score": [None],
                "team2_id": [11],
                "team2_position": [None],
                "team2_score": [None],
                "winner_team_id": [10],
                "loser_team_id": [11],
                "is_bye": [False],
                "last_game_finished_at": [1_700_000_000],
                "match_created_at": [1_699_000_000],
            }
        )

    import rankings.sql.load as load_mod

    monkeypatch.setattr(load_mod, "_read_sql", fake_read_sql)

    # since_ms in ms; until_ms in seconds; only_ranked True
    df = load_matches_df(
        engine=None,
        since_ms=1_700_123_456_789,
        until_ms=1_700_222_222,
        only_ranked=True,
    )
    assert df.height == 1
    assert "sql" in captured and "params" in captured
    sql = captured["sql"]
    params = captured["params"]

    # Assert ranked filter and JOIN appear in SQL
    assert "JOIN" in sql and "is_ranked" in sql
    # since param was in ms and should be converted to seconds
    assert params.get("since_ts") == 1_700_123_456
    # until param left as seconds
    assert params.get("until_ts") == 1_700_222_222
