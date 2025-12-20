"""Tests for rankings.cli.backfill_daily module."""

from datetime import date

import pytest

from rankings.cli.backfill_daily import (
    _day_anchor_ms,
    _format_build_version,
    _iter_dates,
    _parse_date,
)


class TestParseDate:
    def test_valid_date(self):
        assert _parse_date("2025-11-08") == date(2025, 11, 8)

    def test_date_with_whitespace(self):
        assert _parse_date("  2025-12-01  ") == date(2025, 12, 1)

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            _parse_date("not-a-date")


class TestIterDates:
    def test_single_day(self):
        result = _iter_dates(date(2025, 1, 1), date(2025, 1, 1))
        assert result == [date(2025, 1, 1)]

    def test_multiple_days(self):
        result = _iter_dates(date(2025, 1, 1), date(2025, 1, 3))
        assert result == [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="end-date must be >= start-date"):
            _iter_dates(date(2025, 1, 5), date(2025, 1, 1))


class TestDayAnchorMs:
    def test_end_of_day(self):
        ms = _day_anchor_ms(date(2025, 1, 15), cutoff="end")
        assert ms == 1736985599999

    def test_start_of_day(self):
        ms = _day_anchor_ms(date(2025, 1, 15), cutoff="start")
        assert ms == 1736899200000


class TestFormatBuildVersion:
    def test_default_template(self):
        result = _format_build_version("daily-{date}", date(2025, 11, 8))
        assert result == "daily-2025-11-08"

    def test_compact_template(self):
        result = _format_build_version("v{date_compact}", date(2025, 11, 8))
        assert result == "v20251108"

    def test_combined_template(self):
        result = _format_build_version(
            "{date}_build_{date_compact}", date(2025, 12, 25)
        )
        assert result == "2025-12-25_build_20251225"
