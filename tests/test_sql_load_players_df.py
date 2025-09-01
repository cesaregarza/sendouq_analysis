import polars as pl

from rankings.sql.load import load_players_df


def test_load_players_df_casts_and_booleans(monkeypatch):
    def fake_read_sql(engine, sql, params=None):  # noqa: ARG001
        return pl.DataFrame(
            {
                "tournament_id": ["1", "1"],
                "team_id": ["10", "11"],
                "user_id": ["100", "101"],
                "username": ["a", "b"],
                "discord_id": [None, None],
                "country": [None, None],
                "roster_created_at": [None, None],
                "is_owner": [True, False],
            }
        )

    import rankings.sql.load as load_mod

    monkeypatch.setattr(load_mod, "_read_sql", fake_read_sql)
    df = load_players_df(engine=None)
    assert df.height == 2
    # dtypes for ids are ints
    for col in ["tournament_id", "team_id", "user_id"]:
        assert df[col].dtype.__class__.__name__.lower().startswith("int")
    assert df["is_owner"].dtype.__class__.__name__.lower() == "boolean"
