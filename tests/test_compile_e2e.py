from pathlib import Path

import polars as pl

import rankings.cli.compile as compile_cli


def test_compile_main_e2e_no_db(tmp_path, monkeypatch):
    # Synthetic data
    matches = pl.DataFrame(
        {
            "match_id": [1],
            "tournament_id": [77],
            "winner_team_id": [10],
            "loser_team_id": [11],
            "last_game_finished_at": [1_700_000_000.0],
        }
    )
    players = pl.DataFrame(
        {
            "tournament_id": [77] * 8,
            "team_id": [10, 10, 10, 10, 11, 11, 11, 11],
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    appearances = pl.DataFrame(
        {
            "tournament_id": [77] * 8,
            "match_id": [1] * 8,
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    # Monkeypatch loaders and DB helpers
    monkeypatch.setattr(
        compile_cli,
        "load_core_tables",
        lambda *_a, **_k: {"matches": matches, "players": players},
    )
    monkeypatch.setattr(
        compile_cli, "load_player_appearances_df", lambda *_a, **_k: appearances
    )
    monkeypatch.setattr(
        compile_cli, "rankings_create_engine", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(
        compile_cli, "rankings_create_all", lambda *_a, **_k: None
    )

    out_dir = tmp_path / "compiled"
    args = [
        "--write-parquet",
        "--compiled-out",
        str(out_dir),
        "--run-engine",
    ]

    rc = compile_cli.main(args)
    assert rc == 0

    runs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "matches.parquet").exists()
    assert (run_dir / "players.parquet").exists()
    assert (run_dir / "rankings.parquet").exists()
