import os

from rankings.sql.engine import _build_url_from_env


def test_build_url_from_env_components(monkeypatch):
    monkeypatch.setenv("RANKINGS_DB_HOST", "localhost")
    monkeypatch.setenv("RANKINGS_DB_USER", "user")
    monkeypatch.setenv("RANKINGS_DB_PASSWORD", "pass")
    monkeypatch.setenv("RANKINGS_DB_NAME", "db")
    monkeypatch.setenv("RANKINGS_DB_PORT", "5433")
    monkeypatch.setenv("RANKINGS_DB_SSLMODE", "require")

    url = _build_url_from_env()
    assert url.startswith("postgresql://user:pass@localhost:5433/db")
    assert "sslmode=require" in url
