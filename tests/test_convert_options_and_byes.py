import polars as pl

from rankings.core.convert import convert_matches_dataframe


def test_convert_matches_excludes_byes_and_respects_include_share_flag():
    # Two matches: one bye, one real
    matches = pl.DataFrame(
        {
            "match_id": [1, 2],
            "tournament_id": [999, 999],
            "winner_team_id": [10, 10],
            "loser_team_id": [11, 11],
            "last_game_finished_at": [1_700_000_000.0, 1_700_000_100.0],
            "is_bye": [True, False],
        }
    )
    players = pl.DataFrame(
        {
            "tournament_id": [999] * 8,
            "team_id": [10, 10, 10, 10, 11, 11, 11, 11],
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    df = convert_matches_dataframe(
        matches,
        players,
        tournament_influence={},
        now_timestamp=1_700_000_500.0,
        decay_rate=0.0,
        include_share=False,
    )

    # The bye match should be excluded; only match_id 2 remains
    assert df.height == 1
    assert df.select("match_id").to_series().to_list() == [2]
    # Winners/losers lists present; share columns absent when include_share=False
    cols = set(df.columns)
    assert {"winners", "losers", "weight", "ts"}.issubset(cols)
    assert (
        "share" not in cols
        and "winner_count" not in cols
        and "loser_count" not in cols
    )
