"""Time and decay calculation helpers for ranking algorithms."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from typing import Any


@dataclass
class Clock:
    """Clock abstraction for time-based calculations.

    Allows injection of custom time for testing.
    """

    now_timestamp: float | None = None

    def __post_init__(self) -> None:
        """Initialize with current time if not provided."""
        if self.now_timestamp is None:
            self.now_timestamp = time.time()

    @property
    def now(self) -> float:
        """Get current timestamp.

        Returns:
            Current timestamp in seconds since epoch.
        """
        return self.now_timestamp or time.time()

    def days_ago(self, days: float) -> float:
        """Get timestamp for N days ago.

        Args:
            days: Number of days in the past.

        Returns:
            Timestamp for N days ago.
        """
        return self.now - (days * 86400.0)

    def days_since(self, timestamp: float) -> float:
        """Calculate days since given timestamp.

        Args:
            timestamp: The reference timestamp.

        Returns:
            Number of days since the timestamp.
        """
        return (self.now - timestamp) / 86400.0


def event_ts_expr(
    dataframe_columns: list[str] | None = None,
    default_timestamp: float | None = None,
) -> pl.Expr:
    """Create expression for event timestamp extraction.

    Handles multiple possible timestamp column names with fallbacks.

    Args:
        dataframe_columns: Available columns in DataFrame. Defaults to None.
        default_timestamp: Default timestamp if none found. Defaults to None.

    Returns:
        Polars expression for timestamp.
    """
    if default_timestamp is None:
        default_timestamp = time.time()

    if dataframe_columns is None:
        return pl.col("match_created_at").fill_null(default_timestamp)

    if "last_game_finished_at" in dataframe_columns:
        return (
            pl.when(pl.col("last_game_finished_at").is_not_null())
            .then(pl.col("last_game_finished_at"))
            .otherwise(pl.col("match_created_at"))
            .fill_null(default_timestamp)
        )
    elif "match_created_at" in dataframe_columns:
        return pl.col("match_created_at").fill_null(default_timestamp)
    elif "timestamp" in dataframe_columns:
        return pl.col("timestamp").fill_null(default_timestamp)
    elif "ts" in dataframe_columns:
        return pl.col("ts").fill_null(default_timestamp)
    else:
        return pl.lit(default_timestamp)


def decay_expr(
    timestamp_column: str,
    now_timestamp: float,
    decay_rate: float,
) -> pl.Expr:
    """Create expression for time decay calculation.

    Args:
        timestamp_column: Column containing timestamps.
        now_timestamp: Current timestamp.
        decay_rate: Decay rate (typically ln(2) / half_life_days).

    Returns:
        Polars expression for decay factor.
    """
    return (
        ((now_timestamp - pl.col(timestamp_column).cast(pl.Float64)) / 86400.0)
        .mul(-decay_rate)
        .exp()
    )


def compute_decay_factor(
    timestamps: np.ndarray | list[float],
    now_timestamp: float,
    half_life_days: float,
) -> np.ndarray:
    """Compute decay factors for given timestamps.

    Args:
        timestamps: Array of timestamps.
        now_timestamp: Current timestamp.
        half_life_days: Half-life for decay in days.

    Returns:
        Array of decay factors (between 0 and 1).
    """
    timestamp_array = np.asarray(timestamps, dtype=float)

    if half_life_days <= 0:
        return np.ones_like(timestamp_array)

    decay_rate = np.log(2) / half_life_days
    days_elapsed = (now_timestamp - timestamp_array) / 86400.0

    return np.exp(-decay_rate * days_elapsed)


def apply_inactivity_decay(
    scores: np.ndarray,
    last_activity: np.ndarray,
    now_timestamp: float,
    delay_days: float = 30.0,
    decay_rate: float = 0.01,
) -> np.ndarray:
    """Apply decay to scores based on inactivity.

    Scores decay after a delay period of inactivity.

    Args:
        scores: Original scores.
        last_activity: Timestamps of last activity.
        now_timestamp: Current timestamp.
        delay_days: Days before decay starts. Defaults to 30.0.
        decay_rate: Rate of decay after delay. Defaults to 0.01.

    Returns:
        Decayed scores.
    """
    days_inactive = (now_timestamp - last_activity) / 86400.0

    decay_days = np.maximum(0, days_inactive - delay_days)

    decay_factor = np.exp(-decay_rate * decay_days)

    return scores * decay_factor


def filter_by_recency(
    dataframe: pl.DataFrame,
    timestamp_column: str,
    now_timestamp: float,
    max_days: float,
) -> pl.DataFrame:
    """Filter DataFrame to only include recent records.

    Args:
        dataframe: DataFrame with timestamp column.
        timestamp_column: Name of timestamp column.
        now_timestamp: Current timestamp.
        max_days: Maximum age in days.

    Returns:
        Filtered DataFrame.
    """
    cutoff_timestamp = now_timestamp - (max_days * 86400.0)
    return dataframe.filter(pl.col(timestamp_column) >= cutoff_timestamp)


def add_time_features(
    dataframe: pl.DataFrame,
    timestamp_column: str,
    now_timestamp: float,
) -> pl.DataFrame:
    """Add time-based features to DataFrame.

    Adds columns for:
    - days_ago: Days since event
    - week_of_year: Week number
    - day_of_week: Day of week (0=Monday)
    - is_weekend: Whether event was on weekend

    Args:
        dataframe: DataFrame with timestamp column.
        timestamp_column: Name of timestamp column.
        now_timestamp: Current timestamp.

    Returns:
        DataFrame with additional time features.
    """
    return dataframe.with_columns(
        [
            ((now_timestamp - pl.col(timestamp_column)) / 86400.0).alias(
                "days_ago"
            ),
            pl.from_epoch(pl.col(timestamp_column))
            .dt.week()
            .alias("week_of_year"),
            pl.from_epoch(pl.col(timestamp_column))
            .dt.weekday()
            .alias("day_of_week"),
            (pl.from_epoch(pl.col(timestamp_column)).dt.weekday() >= 5).alias(
                "is_weekend"
            ),
        ]
    )


def create_time_windows(
    start_timestamp: float,
    end_timestamp: float,
    window_days: float,
    stride_days: float | None = None,
) -> list[tuple[float, float]]:
    """Create sliding time windows for analysis.

    Args:
        start_timestamp: Start timestamp.
        end_timestamp: End timestamp.
        window_days: Size of each window in days.
        stride_days: Stride between windows. Defaults to window_days.

    Returns:
        List of (window_start, window_end) tuples.
    """
    if stride_days is None:
        stride_days = window_days

    window_size = window_days * 86400.0
    stride = stride_days * 86400.0

    windows = []
    current_start = start_timestamp

    while current_start < end_timestamp:
        current_end = min(current_start + window_size, end_timestamp)
        windows.append((current_start, current_end))
        current_start += stride

    return windows
