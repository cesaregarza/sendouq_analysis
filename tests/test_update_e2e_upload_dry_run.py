import json
from pathlib import Path

import polars as pl

import rankings.cli.update as update_cli


class _DummyEngine:
    def dispose(self) -> None:
        return None


def test_update_main_upload_s3_dry_run(tmp_path, monkeypatch, capsys):
    # Synthetic matches/players/appearances
    matches = pl.DataFrame(
        {
            "match_id": [1],
            "tournament_id": [42],
            "winner_team_id": [10],
            "loser_team_id": [11],
            "last_game_finished_at": [1_700_000_000.0],
        }
    )
    players = pl.DataFrame(
        {
            "tournament_id": [42] * 8,
            "team_id": [10, 10, 10, 10, 11, 11, 11, 11],
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    appearances = pl.DataFrame(
        {
            "tournament_id": [42] * 8,
            "match_id": [1] * 8,
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    # Monkeypatch DB and loaders
    monkeypatch.setattr(
        update_cli, "rankings_create_engine", lambda *_a, **_k: _DummyEngine()
    )
    monkeypatch.setattr(
        update_cli, "rankings_create_all", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        update_cli,
        "load_core_tables",
        lambda *_a, **_k: {"matches": matches, "players": players},
    )
    monkeypatch.setattr(
        update_cli, "load_player_appearances_df", lambda *_a, **_k: appearances
    )

    # Use dry run and S3 URI so no network/boto3 is needed
    monkeypatch.setenv("RANKINGS_UPLOAD_DRY_RUN", "true")
    monkeypatch.setenv("RANKINGS_S3_URI", "s3://mybucket/runs")
    # Intercept upload call to assert it's invoked by the CLI
    calls = {}

    def _fake_upload(out_run: Path, s3_prefix=None):  # noqa: ANN001
        calls["out_run"] = out_run
        calls["s3_prefix"] = s3_prefix
        print(f"UPLOAD_CALLED {out_run} PREFIX={s3_prefix}")

    monkeypatch.setattr(update_cli, "upload_outputs", _fake_upload)

    out_dir = tmp_path / "compiled"
    args = [
        "--skip-discovery",
        "--no-save-to-db",
        "--write-parquet",
        "--compiled-out",
        str(out_dir),
        "--upload-s3",
    ]
    rc = update_cli.main(args)
    assert rc == 0

    # Validate outputs under compiled_out/<ts>/
    runs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "engine_state.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "players.parquet").exists()
    assert (run_dir / "matches.parquet").exists()
    assert (run_dir / "rankings.parquet").exists()

    out = capsys.readouterr().out
    # Our fake uploader prints a sentinel; also verify params
    assert "UPLOAD_CALLED" in out
    assert calls.get("out_run") == run_dir
