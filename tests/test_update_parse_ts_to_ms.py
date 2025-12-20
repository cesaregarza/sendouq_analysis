import calendar
from datetime import datetime, timezone

import pytest

from rankings.cli.update import _parse_ts_to_ms


def _dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return calendar.timegm(dt.utctimetuple()) * 1000 + (dt.microsecond // 1000)


def test_parse_ts_to_ms_rejects_empty():
    with pytest.raises(ValueError, match="Empty timestamp value"):
        _parse_ts_to_ms("")


@pytest.mark.parametrize(
    ("raw", "expected_ms"),
    [
        ("1700000000", 1700000000 * 1000),  # seconds -> ms
        ("1700000000000", 1700000000000),  # ms passthrough
        ("-1700000000", -1700000000 * 1000),  # negative seconds -> ms
    ],
)
def test_parse_ts_to_ms_epoch_ints(raw: str, expected_ms: int):
    assert _parse_ts_to_ms(raw) == expected_ms


def test_parse_ts_to_ms_date_end_of_day_default():
    # Default for YYYY-MM-DD is end-of-day (last ms of that UTC day)
    start_next_day = datetime(2025, 1, 16, tzinfo=timezone.utc)
    expected = _dt_to_ms(start_next_day) - 1
    assert _parse_ts_to_ms("2025-01-15") == expected


def test_parse_ts_to_ms_date_start_of_day_when_disabled():
    expected = _dt_to_ms(datetime(2025, 1, 15, tzinfo=timezone.utc))
    assert _parse_ts_to_ms("2025-01-15", end_of_day_for_date=False) == expected


@pytest.mark.parametrize(
    ("raw", "dt"),
    [
        (
            "2025-01-15T12:34:56Z",
            datetime(2025, 1, 15, 12, 34, 56, tzinfo=timezone.utc),
        ),
        (
            "2025-01-15T12:34:56",
            datetime(2025, 1, 15, 12, 34, 56, tzinfo=timezone.utc),
        ),
        (
            "2025-01-15T12:34:56+02:00",
            datetime(2025, 1, 15, 10, 34, 56, tzinfo=timezone.utc),
        ),
        (
            "2025-01-15T12:34:56.123Z",
            datetime(2025, 1, 15, 12, 34, 56, 123000, tzinfo=timezone.utc),
        ),
        (
            "2025-01-15T12:34:56.123456Z",
            datetime(2025, 1, 15, 12, 34, 56, 123456, tzinfo=timezone.utc),
        ),
    ],
)
def test_parse_ts_to_ms_iso_datetimes(raw: str, dt: datetime):
    assert _parse_ts_to_ms(raw) == _dt_to_ms(dt)
