import polars as pl

from rankings.sql.load import load_player_match_loo_impacts_df


def test_load_player_match_loo_impacts_df_casts_ints_and_booleans(monkeypatch):
    def fake_read_sql(engine, sql, params=None):  # noqa: ARG001
        return pl.DataFrame(
            {
                "player_id": ["100"],
                "match_id": ["11"],
                "tournament_id": ["7"],
                "calculated_at_ms": ["1742255999999"],
                "build_version": ["weekly-test-build"],
                "player_rank": ["1"],
                "player_score": [0.9],
                "is_win": [True],
                "approx_variant": ["perturb_2"],
                "approx_positive_rank": ["1"],
                "approx_negative_rank": [None],
                "approx_old_score": [0.9],
                "approx_new_score": [1.02],
                "approx_score_delta": [0.12],
                "approx_abs_delta": [0.12],
                "exact_variant": ["exact_combined"],
                "exact_old_score": [0.9],
                "exact_new_score": [0.911],
                "exact_score_delta": [0.011],
                "exact_abs_delta": [0.011],
            }
        )

    import rankings.sql.load as load_mod

    monkeypatch.setattr(load_mod, "_read_sql", fake_read_sql)
    df = load_player_match_loo_impacts_df(
        engine=None,
        build_version="weekly-test-build",
        calculated_at_ms=1_742_255_999_999,
        player_id=100,
    )

    assert df.height == 1
    for col in [
        "player_id",
        "match_id",
        "tournament_id",
        "calculated_at_ms",
        "player_rank",
        "approx_positive_rank",
    ]:
        assert df[col].dtype.__class__.__name__.lower().startswith("int")
    assert df["is_win"].dtype.__class__.__name__.lower() == "boolean"

