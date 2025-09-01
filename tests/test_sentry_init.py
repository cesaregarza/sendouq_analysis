import logging
import os

import pytest

from rankings.core.sentry import _parse_float_env, init_sentry


def test_parse_float_env_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "2.0")
    assert _parse_float_env("SENTRY_TRACES_SAMPLE_RATE", 0.0) == 1.0
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "-0.5")
    assert _parse_float_env("SENTRY_TRACES_SAMPLE_RATE", 0.0) == 0.0
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.25")
    assert _parse_float_env("SENTRY_TRACES_SAMPLE_RATE", 0.0) == 0.25


def test_init_sentry_no_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure DSN envs are unset
    for k in ("SENTRY_DSN", "RANKINGS_SENTRY_DSN", "DASH_SENTRY_DSN"):
        monkeypatch.delenv(k, raising=False)
    initialized = init_sentry(context="test_cli")
    assert initialized is False


def test_init_sentry_invalid_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "not-a-valid-dsn")
    initialized = init_sentry(context="test_cli")
    assert initialized is False
