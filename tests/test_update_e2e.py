import json
from pathlib import Path

import polars as pl

import rankings.cli.update as update_cli


class _DummyEngine:
    def dispose(self) -> None:
        return None


def test_update_main_e2e_no_db_no_network(tmp_path, monkeypatch):
    # Synthetic matches/players/appearances
    matches = pl.DataFrame(
        {
            "match_id": [1, 2],
            "tournament_id": [42, 42],
            "winner_team_id": [10, 11],
            "loser_team_id": [11, 10],
            "last_game_finished_at": [1_700_000_000.0, 1_700_010_000.0],
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
            "tournament_id": [42] * 8 * 2,
            "match_id": [1] * 8 + [2] * 8,
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8] * 2,
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
    # Avoid random network scraping path by setting skip-discovery; also avoid upload
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
retrieval_days: 30
upload_s3: false
only_ranked: false
        """
    )

    out_dir = tmp_path / "compiled"
    args = [
        "--config",
        str(cfg_path),
        "--skip-discovery",
        "--no-save-to-db",
        "--write-parquet",
        "--compiled-out",
        str(out_dir),
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

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest.get("compiled_matches") == 2
    assert manifest.get("compiled_players") == 8
    assert manifest.get("rankings_rows", 0) > 0
