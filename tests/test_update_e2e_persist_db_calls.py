import polars as pl

import rankings.cli.update as update_cli


class _DummyEngine:
    def dispose(self) -> None:
        return None


def test_update_main_persists_rankings_and_stats_when_enabled(
    tmp_path, monkeypatch, capsys
):
    # Minimal synthetic data
    matches = pl.DataFrame(
        {
            "match_id": [1],
            "tournament_id": [7],
            "winner_team_id": [10],
            "loser_team_id": [11],
            "last_game_finished_at": [1_700_000_000.0],
            "match_created_at": [None],
        }
    )
    players = pl.DataFrame(
        {
            "tournament_id": [7] * 8,
            "team_id": [10, 10, 10, 10, 11, 11, 11, 11],
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    appearances = pl.DataFrame(
        {
            "tournament_id": [7] * 8,
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

    # Capture persist calls
    calls = {"ranks": 0, "stats": 0}

    def _fake_persist_rankings(
        engine, ranks, build_version, calculated_at_ms
    ):  # noqa: ANN001
        assert hasattr(ranks, "height") and ranks.height > 0
        calls["ranks"] += int(ranks.height)
        return ranks.height

    def _fake_persist_stats(
        engine, stats, build_version, calculated_at_ms
    ):  # noqa: ANN001
        # Stats may be filtered to ranked IDs; still should be non-empty
        assert hasattr(stats, "height") and stats.height > 0
        calls["stats"] += int(stats.height)
        return stats.height

    monkeypatch.setattr(update_cli, "_persist_rankings", _fake_persist_rankings)
    monkeypatch.setattr(
        update_cli, "_persist_ranking_stats", _fake_persist_stats
    )

    out_dir = tmp_path / "compiled"
    args = [
        "--skip-discovery",
        "--save-to-db",
        "--write-parquet",
        "--compiled-out",
        str(out_dir),
    ]
    rc = update_cli.main(args)
    assert rc == 0

    # Validate persist calls occurred
    assert calls["ranks"] > 0
    assert calls["stats"] > 0
