"""Time and decay calculation helpers for ranking algorithms."""

import time
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import polars as pl


@dataclass
class Clock:
    """
    Clock abstraction for time-based calculations.

    Allows injection of custom time for testing.
    """

    now_ts: Optional[float] = None

    def __post_init__(self):
        """Initialize with current time if not provided."""
        if self.now_ts is None:
            self.now_ts = time.time()

    @property
    def now(self) -> float:
        """Get current timestamp."""
        return self.now_ts or time.time()

    def days_ago(self, days: float) -> float:
        """Get timestamp for N days ago."""
        return self.now - (days * 86400.0)

    def days_since(self, timestamp: float) -> float:
        """Calculate days since given timestamp."""
        return (self.now - timestamp) / 86400.0


def event_ts_expr(
    df_columns: Optional[list] = None,
    default_ts: Optional[float] = None,
) -> pl.Expr:
    """
    Create expression for event timestamp extraction.

    Handles multiple possible timestamp column names with fallbacks.

    Args:
        df_columns: Available columns in DataFrame
        default_ts: Default timestamp if none found

    Returns:
        Polars expression for timestamp
    """
    if default_ts is None:
        default_ts = time.time()

    # If we don't know the columns, use default behavior
    if df_columns is None:
        return pl.col("match_created_at").fill_null(default_ts)

    # Prefer finished timestamp over created timestamp
    if "last_game_finished_at" in df_columns:
        return (
            pl.when(pl.col("last_game_finished_at").is_not_null())
            .then(pl.col("last_game_finished_at"))
            .otherwise(pl.col("match_created_at"))
            .fill_null(default_ts)
        )
    elif "match_created_at" in df_columns:
        return pl.col("match_created_at").fill_null(default_ts)
    elif "timestamp" in df_columns:
        return pl.col("timestamp").fill_null(default_ts)
    elif "ts" in df_columns:
        return pl.col("ts").fill_null(default_ts)
    else:
        # No timestamp columns found
        return pl.lit(default_ts)


def decay_expr(
    ts_col: str,
    now_ts: float,
    decay_rate: float,
) -> pl.Expr:
    """
    Create expression for time decay calculation.

    Args:
        ts_col: Column containing timestamps
        now_ts: Current timestamp
        decay_rate: Decay rate (typically ln(2) / half_life_days)

    Returns:
        Polars expression for decay factor
    """
    return (
        ((now_ts - pl.col(ts_col).cast(pl.Float64)) / 86400.0)
        .mul(-decay_rate)
        .exp()
    )


def compute_decay_factor(
    timestamps: Union[np.ndarray, list],
    now_ts: float,
    half_life_days: float,
) -> np.ndarray:
    """
    Compute decay factors for given timestamps.

    Args:
        timestamps: Array of timestamps
        now_ts: Current timestamp
        half_life_days: Half-life for decay in days

    Returns:
        Array of decay factors (between 0 and 1)
    """
    timestamps = np.asarray(timestamps, dtype=float)

    if half_life_days <= 0:
        # No decay
        return np.ones_like(timestamps)

    decay_rate = np.log(2) / half_life_days
    days_elapsed = (now_ts - timestamps) / 86400.0

    return np.exp(-decay_rate * days_elapsed)


def apply_inactivity_decay(
    scores: np.ndarray,
    last_activity: np.ndarray,
    now_ts: float,
    delay_days: float = 30.0,
    decay_rate: float = 0.01,
) -> np.ndarray:
    """
    Apply decay to scores based on inactivity.

    Scores decay after a delay period of inactivity.

    Args:
        scores: Original scores
        last_activity: Timestamps of last activity
        now_ts: Current timestamp
        delay_days: Days before decay starts
        decay_rate: Rate of decay after delay

    Returns:
        Decayed scores
    """
    days_inactive = (now_ts - last_activity) / 86400.0

    # Only decay after delay period
    decay_days = np.maximum(0, days_inactive - delay_days)

    # Exponential decay
    decay_factor = np.exp(-decay_rate * decay_days)

    return scores * decay_factor


def filter_by_recency(
    df: pl.DataFrame,
    ts_col: str,
    now_ts: float,
    max_days: float,
) -> pl.DataFrame:
    """
    Filter DataFrame to only include recent records.

    Args:
        df: DataFrame with timestamp column
        ts_col: Name of timestamp column
        now_ts: Current timestamp
        max_days: Maximum age in days

    Returns:
        Filtered DataFrame
    """
    cutoff_ts = now_ts - (max_days * 86400.0)
    return df.filter(pl.col(ts_col) >= cutoff_ts)


def add_time_features(
    df: pl.DataFrame,
    ts_col: str,
    now_ts: float,
) -> pl.DataFrame:
    """
    Add time-based features to DataFrame.

    Adds columns for:
    - days_ago: Days since event
    - week_of_year: Week number
    - day_of_week: Day of week (0=Monday)
    - is_weekend: Whether event was on weekend

    Args:
        df: DataFrame with timestamp column
        ts_col: Name of timestamp column
        now_ts: Current timestamp

    Returns:
        DataFrame with additional time features
    """
    return df.with_columns(
        [
            # Days since event
            ((now_ts - pl.col(ts_col)) / 86400.0).alias("days_ago"),
            # Extract datetime features
            pl.from_epoch(pl.col(ts_col)).dt.week().alias("week_of_year"),
            pl.from_epoch(pl.col(ts_col)).dt.weekday().alias("day_of_week"),
            # Weekend indicator
            (pl.from_epoch(pl.col(ts_col)).dt.weekday() >= 5).alias(
                "is_weekend"
            ),
        ]
    )


def create_time_windows(
    start_ts: float,
    end_ts: float,
    window_days: float,
    stride_days: Optional[float] = None,
) -> list[tuple[float, float]]:
    """
    Create sliding time windows for analysis.

    Args:
        start_ts: Start timestamp
        end_ts: End timestamp
        window_days: Size of each window in days
        stride_days: Stride between windows (defaults to window_days)

    Returns:
        List of (window_start, window_end) tuples
    """
    if stride_days is None:
        stride_days = window_days

    window_size = window_days * 86400.0
    stride = stride_days * 86400.0

    windows = []
    current_start = start_ts

    while current_start < end_ts:
        current_end = min(current_start + window_size, end_ts)
        windows.append((current_start, current_end))
        current_start += stride

    return windows
