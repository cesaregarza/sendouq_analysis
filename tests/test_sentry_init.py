import logging
import os
import sys
from types import ModuleType

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


def _install_fake_sentry(monkeypatch: pytest.MonkeyPatch, record: dict) -> None:
    """Install a minimal fake sentry_sdk module and logging integration submodule."""
    fake = ModuleType("sentry_sdk")

    def fake_init(**kwargs):
        record.update(kwargs)

    def fake_set_tag(k, v):
        record.setdefault("tags", {})[k] = v

    fake.init = fake_init  # type: ignore[attr-defined]
    fake.set_tag = fake_set_tag  # type: ignore[attr-defined]

    integ_mod = ModuleType("sentry_sdk.integrations.logging")

    class LoggingIntegration:  # type: ignore
        def __init__(self, level=None, event_level=None):
            self.level = level
            self.event_level = event_level

    integ_mod.LoggingIntegration = LoggingIntegration  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    monkeypatch.setitem(
        sys.modules, "sentry_sdk.integrations.logging", integ_mod
    )


def test_init_sentry_success_and_order(monkeypatch: pytest.MonkeyPatch, caplog):
    caplog.set_level(logging.INFO, logger="rankings.core.sentry")
    # Install fake sentry to avoid external deps
    record: dict = {}
    _install_fake_sentry(monkeypatch, record)

    # Prefer first env in provided list
    monkeypatch.setenv("PRIMARY_DSN", "https://abc@host/project")
    monkeypatch.setenv("SECONDARY_DSN", "https://def@host/project")
    initialized = init_sentry(
        context="test_cli",
        release="r1",
        dsn_envs=["PRIMARY_DSN", "SECONDARY_DSN"],
    )
    assert initialized is True
    assert record.get("dsn") == "https://abc@host/project"
    assert record.get("release") == "r1"
    # Info log should state initialized
    assert any("Sentry initialized" in m for m in caplog.messages)
