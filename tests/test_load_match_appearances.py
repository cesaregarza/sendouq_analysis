import json
from pathlib import Path

import polars as pl

from rankings.scraping.storage import load_match_appearances


def test_load_match_appearances_from_temp_dir(tmp_path: Path):
    # Build a minimal tournament entry with players payload
    entry = {
        "tournament": {"ctx": {"id": 123}},
        "player_matches": {
            "matches": [
                {
                    "matchId": 1,
                    "teams": [
                        {
                            "teamId": 10,
                            "players": [{"userId": 1}, {"userId": 2}],
                        },
                        {
                            "teamId": 11,
                            "players": [{"userId": 3}, {"userId": 4}],
                        },
                    ],
                }
            ]
        },
    }

    # Write as a batch file that the loader can discover
    p = tmp_path / "tournament_0.json"
    p.write_text(json.dumps([entry]))

    df = load_match_appearances(str(tmp_path))
    assert isinstance(df, pl.DataFrame)
    assert df.height == 4
    # Check dedup and dtypes
    assert set(df.columns) >= {"tournament_id", "match_id", "user_id"}
    assert (
        df.dtypes[df.columns.index("tournament_id")]
        .__class__.__name__.lower()
        .startswith("int")
    )
    assert (
        df.filter((pl.col("match_id") == 1) & (pl.col("user_id") == 1)).height
        == 1
    )
