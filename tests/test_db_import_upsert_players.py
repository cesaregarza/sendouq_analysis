"""Tests for _upsert_players behavior in db_import."""

import polars as pl

from rankings.cli.db_import import _upsert_players


class FakeConnection:
    """Capture executed statements."""

    def __init__(self):
        self.executed = []

    def execute(self, stmt):
        self.executed.append(stmt)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeEngine:
    """Fake engine that captures statements."""

    def __init__(self):
        self.conn = FakeConnection()

    def begin(self):
        return self.conn


def test_upsert_players_empty_df_does_nothing():
    """Empty DataFrame should not execute any statements."""
    engine = FakeEngine()
    _upsert_players(pl.DataFrame([]), engine)
    assert len(engine.conn.executed) == 0


def test_upsert_players_none_does_nothing():
    """None input should not execute any statements."""
    engine = FakeEngine()
    _upsert_players(None, engine)
    assert len(engine.conn.executed) == 0


def test_upsert_players_builds_on_conflict_do_update(monkeypatch):
    """Verify the upsert statement uses ON CONFLICT DO UPDATE."""
    captured = {}

    def fake_pg_insert(table):
        class FakeInsert:
            def __init__(self):
                self.values_data = None
                self.conflict_config = None

            def values(self, rows):
                self.values_data = rows
                return self

            def on_conflict_do_update(self, index_elements, set_):
                self.conflict_config = {
                    "index_elements": index_elements,
                    "set_": set_,
                }
                captured["stmt"] = self
                return self

            @property
            def excluded(self):
                return type(
                    "Excluded",
                    (),
                    {
                        "display_name": "excluded.display_name",
                        "discord_id": "excluded.discord_id",
                        "country": "excluded.country",
                    },
                )()

        return FakeInsert()

    import rankings.cli.db_import as db_import_mod

    monkeypatch.setattr(db_import_mod, "pg_insert", fake_pg_insert)

    df = pl.DataFrame(
        {
            "player_id": [123, 456],
            "display_name": ["Alice", "Bob"],
            "discord_id": ["alice#1234", "bob#5678"],
            "country": ["US", "JP"],
        }
    )
    engine = FakeEngine()
    _upsert_players(df, engine)

    assert "stmt" in captured
    stmt = captured["stmt"]
    assert stmt.values_data is not None
    assert len(stmt.values_data) == 2
    assert stmt.conflict_config is not None
    # Verify it updates display_name, discord_id, country on conflict
    assert "display_name" in stmt.conflict_config["set_"]
    assert "discord_id" in stmt.conflict_config["set_"]
    assert "country" in stmt.conflict_config["set_"]
