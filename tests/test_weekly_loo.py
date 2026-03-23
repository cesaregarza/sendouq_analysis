import json

import polars as pl

import rankings.cli.weekly_loo as weekly_loo


class _DummyEngine:
    def dispose(self) -> None:
        return None


class _FakeLooAnalyzer:
    def __init__(self) -> None:
        self.matches_df = pl.DataFrame(
            {
                "match_id": [11, 12, 21, 22],
                "tournament_id": [7, 7, 8, 8],
            }
        )

    def analyze_entity_matches_variant(
        self,
        player_id: int,
        *,
        variant: str,
        limit,
        include_teleport: bool,
        parallel: bool,
        max_workers: int,
    ) -> pl.DataFrame:
        assert variant == weekly_loo.APPROX_VARIANT
        assert limit is None
        assert include_teleport is True
        assert parallel is True
        assert max_workers == 1

        if player_id == 100:
            return pl.DataFrame(
                {
                    "match_id": [11, 12],
                    "is_win": [True, False],
                    "old_score": [0.90, 0.90],
                    "new_score": [1.02, 0.81],
                    "score_delta": [0.12, -0.09],
                    "abs_delta": [0.12, 0.09],
                }
            )
        if player_id == 200:
            return pl.DataFrame(
                {
                    "match_id": [21, 22],
                    "is_win": [True, False],
                    "old_score": [0.70, 0.70],
                    "new_score": [0.79, 0.64],
                    "score_delta": [0.09, -0.06],
                    "abs_delta": [0.09, 0.06],
                }
            )
        return pl.DataFrame([])

    def impact_of_match_on_entity_variant(
        self,
        match_id: int,
        player_id: int,
        *,
        variant: str,
        include_teleport: bool,
    ) -> dict:
        assert variant == weekly_loo.EXACT_VARIANT
        assert include_teleport is True
        base = 0.90 if player_id == 100 else 0.70
        delta = match_id / 1000.0
        return {
            "ok": True,
            "old": {"score": base},
            "new": {"score": base + delta},
            "delta": {"score": delta},
        }


class _FakeRankEngine:
    def __init__(self, *_args, **_kwargs) -> None:
        self._loo = _FakeLooAnalyzer()

    def rank_players(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        appearances: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        assert matches.height == 2
        assert players.height == 8
        assert appearances is not None and appearances.height == 8
        return pl.DataFrame(
            {
                "id": [100, 200],
                "player_rank": [0.90, 0.70],
                "score": [0.90, 0.70],
            }
        )

    def prepare_loo_analyzer(self) -> None:
        return None

    def get_loo_analyzer(self) -> _FakeLooAnalyzer:
        return self._loo


def test_shortlist_matches_selects_top_positive_and_negative():
    approx_df = pl.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "is_win": [True, True, False, False],
            "old_score": [1.0, 1.0, 1.0, 1.0],
            "new_score": [1.6, 1.3, 0.8, 0.1],
            "score_delta": [0.6, 0.3, -0.2, -0.9],
            "abs_delta": [0.6, 0.3, 0.2, 0.9],
        }
    )

    shortlist = weekly_loo._shortlist_matches(approx_df, top_k=1).sort(
        "match_id"
    )

    assert shortlist["match_id"].to_list() == [1, 4]
    rows = {row["match_id"]: row for row in shortlist.iter_rows(named=True)}
    assert rows[1]["approx_positive_rank"] == 1
    assert rows[1]["approx_negative_rank"] is None
    assert rows[4]["approx_positive_rank"] is None
    assert rows[4]["approx_negative_rank"] == 1


def test_shortlist_matches_merges_duplicate_match_across_both_sides():
    approx_df = pl.DataFrame(
        {
            "match_id": [10, 10],
            "is_win": [True, True],
            "old_score": [1.0, 1.0],
            "new_score": [1.5, 0.5],
            "score_delta": [0.5, -0.5],
            "abs_delta": [0.5, 0.5],
        }
    )

    shortlist = weekly_loo._shortlist_matches(approx_df, top_k=1)

    assert shortlist.height == 1
    row = shortlist.to_dicts()[0]
    assert row["match_id"] == 10
    assert row["approx_positive_rank"] == 1
    assert row["approx_negative_rank"] == 1


def test_weekly_loo_main_persists_base_run_and_shortlisted_exact_rows(
    monkeypatch,
):
    matches = pl.DataFrame(
        {
            "match_id": [11, 21],
            "tournament_id": [7, 8],
            "winner_team_id": [10, 20],
            "loser_team_id": [11, 21],
            "last_game_finished_at": [1_700_000_000.0, 1_700_010_000.0],
            "match_created_at": [1_700_000_000.0, 1_700_010_000.0],
        }
    )
    players = pl.DataFrame(
        {
            "tournament_id": [7] * 4 + [8] * 4,
            "team_id": [10, 10, 11, 11, 20, 20, 21, 21],
            "user_id": [100, 101, 102, 103, 200, 201, 202, 203],
        }
    )
    appearances = pl.DataFrame(
        {
            "tournament_id": [7] * 4 + [8] * 4,
            "match_id": [11] * 4 + [21] * 4,
            "user_id": [100, 101, 102, 103, 200, 201, 202, 203],
            "team_id": [10, 10, 11, 11, 20, 20, 21, 21],
        }
    )
    stats = pl.DataFrame(
        {
            "player_id": [100, 200],
            "tournament_count": [1, 1],
            "last_active_ms": [1_700_000_000_000, 1_700_010_000_000],
        }
    )

    persisted: dict[str, object] = {}

    monkeypatch.setattr(
        weekly_loo, "rankings_create_engine", lambda *_a, **_k: _DummyEngine()
    )
    monkeypatch.setattr(weekly_loo, "rankings_create_all", lambda *_a, **_k: None)
    monkeypatch.setattr(
        weekly_loo,
        "load_core_tables",
        lambda *_a, **_k: {"matches": matches, "players": players},
    )
    monkeypatch.setattr(
        weekly_loo, "_load_appearances", lambda *_a, **_k: appearances
    )
    monkeypatch.setattr(weekly_loo, "setup_logging", lambda **_k: None)
    monkeypatch.setattr(weekly_loo, "init_sentry", lambda **_k: None)
    monkeypatch.setattr(weekly_loo, "ExposureLogOddsEngine", _FakeRankEngine)
    monkeypatch.setattr(weekly_loo, "_build_engine_config", lambda: object())
    monkeypatch.setattr(
        weekly_loo.update_cli,
        "_compute_player_stats",
        lambda *_a, **_k: stats,
    )
    monkeypatch.setattr(
        weekly_loo.update_cli,
        "_persist_rankings",
        lambda *_a, **_k: 2,
    )
    monkeypatch.setattr(
        weekly_loo.update_cli,
        "_persist_ranking_stats",
        lambda *_a, **_k: 2,
    )

    def _fake_persist_loo(
        engine,
        impacts: pl.DataFrame,
        *,
        build_version: str,
        calculated_at_ms: int,
    ) -> int:
        persisted["build_version"] = build_version
        persisted["calculated_at_ms"] = calculated_at_ms
        persisted["impacts"] = impacts.sort(["player_id", "match_id"])
        return impacts.height

    monkeypatch.setattr(
        weekly_loo, "_persist_weekly_loo_impacts", _fake_persist_loo
    )

    rc = weekly_loo.main(
        [
            "--build-version",
            "weekly-test-build",
            "--calculated-at",
            "2026-03-17",
            "--top-k",
            "1",
            "--max-workers",
            "1",
        ]
    )

    assert rc == 0
    assert persisted["build_version"] == "weekly-test-build"
    assert persisted["calculated_at_ms"] == weekly_loo.update_cli._parse_ts_to_ms(
        "2026-03-17", end_of_day_for_date=True
    )

    impacts = persisted["impacts"]
    assert isinstance(impacts, pl.DataFrame)
    assert impacts.height == 4
    assert impacts["player_id"].to_list() == [100, 100, 200, 200]
    assert impacts["match_id"].to_list() == [11, 12, 21, 22]
    assert impacts["player_rank"].to_list() == [1, 1, 2, 2]
    assert impacts["player_score"].to_list() == [0.9, 0.9, 0.7, 0.7]
    assert impacts["approx_positive_rank"].to_list() == [1, None, 1, None]
    assert impacts["approx_negative_rank"].to_list() == [None, 1, None, 1]
    assert impacts["approx_variant"].to_list() == [
        weekly_loo.APPROX_VARIANT
    ] * 4
    assert impacts["exact_variant"].to_list() == [weekly_loo.EXACT_VARIANT] * 4


def test_weekly_loo_main_writes_local_bundle_without_db_persist(
    monkeypatch,
    tmp_path,
):
    matches = pl.DataFrame(
        {
            "match_id": [11, 21],
            "tournament_id": [7, 8],
            "winner_team_id": [10, 20],
            "loser_team_id": [11, 21],
            "last_game_finished_at": [1_700_000_000.0, 1_700_010_000.0],
            "match_created_at": [1_700_000_000.0, 1_700_010_000.0],
        }
    )
    players = pl.DataFrame(
        {
            "tournament_id": [7] * 4 + [8] * 4,
            "team_id": [10, 10, 11, 11, 20, 20, 21, 21],
            "user_id": [100, 101, 102, 103, 200, 201, 202, 203],
        }
    )
    appearances = pl.DataFrame(
        {
            "tournament_id": [7] * 4 + [8] * 4,
            "match_id": [11] * 4 + [21] * 4,
            "user_id": [100, 101, 102, 103, 200, 201, 202, 203],
            "team_id": [10, 10, 11, 11, 20, 20, 21, 21],
        }
    )
    stats = pl.DataFrame(
        {
            "player_id": [100, 200],
            "tournament_count": [1, 1],
            "last_active_ms": [1_700_000_000_000, 1_700_010_000_000],
        }
    )

    monkeypatch.setattr(
        weekly_loo, "rankings_create_engine", lambda *_a, **_k: _DummyEngine()
    )
    monkeypatch.setattr(
        weekly_loo,
        "rankings_create_all",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("rankings_create_all should not be called")
        ),
    )
    monkeypatch.setattr(
        weekly_loo,
        "load_core_tables",
        lambda *_a, **_k: {"matches": matches, "players": players},
    )
    monkeypatch.setattr(
        weekly_loo, "_load_appearances", lambda *_a, **_k: appearances
    )
    monkeypatch.setattr(weekly_loo, "setup_logging", lambda **_k: None)
    monkeypatch.setattr(weekly_loo, "init_sentry", lambda **_k: None)
    monkeypatch.setattr(weekly_loo, "ExposureLogOddsEngine", _FakeRankEngine)
    monkeypatch.setattr(weekly_loo, "_build_engine_config", lambda: object())
    monkeypatch.setattr(
        weekly_loo.update_cli,
        "_compute_player_stats",
        lambda *_a, **_k: stats,
    )
    monkeypatch.setattr(
        weekly_loo.update_cli,
        "_persist_rankings",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("_persist_rankings should not be called")
        ),
    )
    monkeypatch.setattr(
        weekly_loo.update_cli,
        "_persist_ranking_stats",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("_persist_ranking_stats should not be called")
        ),
    )
    monkeypatch.setattr(
        weekly_loo,
        "_persist_weekly_loo_impacts",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("_persist_weekly_loo_impacts should not be called")
        ),
    )

    rc = weekly_loo.main(
        [
            "--build-version",
            "weekly-test-local",
            "--calculated-at",
            "2026-03-17",
            "--top-k",
            "1",
            "--max-workers",
            "1",
            "--no-save-to-db",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0

    run_dir = tmp_path / (
        f"{weekly_loo.update_cli._parse_ts_to_ms('2026-03-17', end_of_day_for_date=True)}"
        "_weekly-test-local"
    )
    assert (run_dir / "matches.parquet").exists()
    assert (run_dir / "players.parquet").exists()
    assert (run_dir / "appearances.parquet").exists()
    assert (run_dir / "rankings.parquet").exists()
    assert (run_dir / "ranking_stats.parquet").exists()
    assert (run_dir / "weekly_loo_impacts.parquet").exists()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["build_version"] == "weekly-test-local"
    assert manifest["save_to_db"] is False
    assert manifest["artifacts"]["weekly_loo_impacts"]["rows"] == 4
