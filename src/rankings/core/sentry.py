from __future__ import annotations

"""
Sentry initialization helpers for the rankings project.

Environment variables (all optional; safe to omit):
- SENTRY_DSN / RANKINGS_SENTRY_DSN / DASH_SENTRY_DSN: DSN URL used to enable Sentry.
- SENTRY_ENV / ENV: Environment name (e.g., production, staging). Defaults to development.
- SENTRY_TRACES_SAMPLE_RATE: Float in [0,1] for performance tracing sample rate.
- SENTRY_PROFILES_SAMPLE_RATE: Float in [0,1] for profiling sample rate.
- SENTRY_DEBUG: If set to a truthy value (1/true/yes/on), enables SDK debug output.

Usage:
    from rankings.core.sentry import init_sentry
    init_sentry(context="rankings_update")

This module centralizes DSN discovery, float env parsing, and integrates
logging, while logging debug-only breadcrumbs when initialization is skipped.
"""

import logging
import os
from typing import Any, Iterable, Optional, Sequence

_LOG = logging.getLogger("rankings.core.sentry")


def _parse_float_env(name: str, default: float) -> float:
    """Parse a float environment variable with a default and clamping.

    Returns `default` if unset or invalid. Values below 0 or above 1 are
    clamped into [0.0, 1.0].
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        val = float(raw)
    except Exception:
        _LOG.debug(
            "Invalid float for %s: %r; using default=%s", name, raw, default
        )
        return default
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def _first_env(names: Iterable[str]) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def _is_valid_dsn(dsn: str) -> bool:
    """Lightweight DSN validation to catch obvious misconfigurations.

    Accepts http(s) DSNs with a host component. Returns False for strings
    missing a scheme or host.
    """
    try:
        from urllib.parse import urlparse

        u = urlparse(dsn)
        return (u.scheme in {"http", "https"}) and bool(u.netloc)
    except Exception:
        return False


def init_sentry(
    *,
    context: str,
    release: Optional[str] = None,
    dsn_envs: Optional[Iterable[str]] = None,
    extra_integrations: Optional[Sequence[Any]] = None,
) -> bool:
    """Initialize Sentry best-effort and return whether it initialized.

    - Reads the DSN from the first non-empty env in `dsn_envs` (default:
      ["SENTRY_DSN", "RANKINGS_SENTRY_DSN"]).
    - Reads environment from SENTRY_ENV or ENV (default: development).
    - Parses `SENTRY_TRACES_SAMPLE_RATE` and `SENTRY_PROFILES_SAMPLE_RATE` with
      safe float parsing and clamping to [0,1].
    - Enables LoggingIntegration so ERROR-level logs are captured as events.

    Returns True if Sentry SDK initialized, False otherwise.
    """
    dsn_envs = (
        list(dsn_envs)
        if dsn_envs is not None
        else [
            "SENTRY_DSN",
            "RANKINGS_SENTRY_DSN",
        ]
    )
    dsn = _first_env(dsn_envs)
    if not dsn:
        _LOG.info(
            "Sentry disabled: no DSN configured (checked envs=%s)", list(dsn_envs)
        )
        return False
    # Be resilient to accidental quotes/whitespace in env secret values
    dsn = dsn.strip().strip("\"").strip("'")
    if not _is_valid_dsn(dsn):
        _LOG.info("Sentry disabled: DSN appears invalid; check secrets/env")
        return False
    try:
        import sentry_sdk  # type: ignore
        from sentry_sdk.integrations.logging import (
            LoggingIntegration,  # type: ignore
        )
    except (
        Exception
    ) as e:  # pragma: no cover - import failures are environment-specific
        _LOG.info("Sentry disabled: sentry_sdk import failed: %s", e)
        return False

    # Support multiple common env var names for environment selection
    env = (
        os.getenv("SENTRY_ENV")
        or os.getenv("SENTRY_ENVIRONMENT")
        or os.getenv("ENV")
        or "development"
    )
    traces = _parse_float_env("SENTRY_TRACES_SAMPLE_RATE", 0.0)
    profiles = _parse_float_env("SENTRY_PROFILES_SAMPLE_RATE", 0.0)
    debug = _truthy_env("SENTRY_DEBUG")

    logging_integration = LoggingIntegration(
        level=logging.INFO,  # breadcrumb level
        event_level=logging.ERROR,  # event threshold
    )
    integrations = [logging_integration]
    if extra_integrations:
        integrations.extend(list(extra_integrations))

    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=env,
            release=release,
            integrations=integrations,
            traces_sample_rate=traces,
            profiles_sample_rate=profiles,
            debug=debug,
        )
        sentry_sdk.set_tag("service", context)
        _LOG.info(
            "Sentry initialized: context=%s env=%s traces=%s profiles=%s",
            context,
            env,
            traces,
            profiles,
        )
        return True
    except Exception as e:  # pragma: no cover - defensive
        _LOG.info("Sentry init failed: %s", e)
        return False


__all__ = [
    "init_sentry",
    "_parse_float_env",
]
