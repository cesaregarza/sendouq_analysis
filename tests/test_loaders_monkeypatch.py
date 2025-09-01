import polars as pl

from rankings.sql.load import load_player_appearances_df


def test_load_player_appearances_df_casts_and_dedups(monkeypatch):
    # Provide duplicate rows and string types via monkeypatched _read_sql
    fake = pl.DataFrame(
        {
            "tournament_id": ["1", "1", "1"],
            "match_id": ["10", "10", "10"],
            "user_id": ["100", "100", "101"],
        }
    )

    def fake_read_sql(engine, sql, params=None):  # noqa: ARG001
        return fake

    import rankings.sql.load as load_mod

    monkeypatch.setattr(load_mod, "_read_sql", fake_read_sql)

    df = load_player_appearances_df(engine=None)  # engine not used by fake
    assert df.height == 2  # deduped
    # Int casts applied
    assert (
        df.dtypes[df.columns.index("tournament_id")]
        .__class__.__name__.lower()
        .startswith("int")
    )
    assert (
        df.dtypes[df.columns.index("match_id")]
        .__class__.__name__.lower()
        .startswith("int")
    )
    assert (
        df.dtypes[df.columns.index("user_id")]
        .__class__.__name__.lower()
        .startswith("int")
    )
