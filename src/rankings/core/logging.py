"""
Centralized logging configuration for the rankings package.

This module provides consistent logging setup across all components of the
rankings system, including structured logging, performance metrics, and
debug information.
"""

import logging
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_style: str = "detailed",
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Set up centralized logging for the rankings package.

    Parameters
    ----------
    level : Union[str, int]
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Optional[Union[str, Path]]
        Optional file to write logs to
    format_style : str
        Format style: "simple", "detailed", or "json"
    include_timestamp : bool
        Whether to include timestamps in log messages

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Create logger
    logger = logging.getLogger("rankings")
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter based on style
    if format_style == "simple":
        fmt = "%(levelname)s: %(message)s"
    elif format_style == "json":
        fmt = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
    else:  # detailed
        if include_timestamp:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        else:
            fmt = "%(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

    formatter = logging.Formatter(fmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module/component.

    Parameters
    ----------
    name : str
        Name of the component (usually __name__)

    Returns
    -------
    logging.Logger
        Logger instance for the component
    """
    return logging.getLogger(f"rankings.{name}")


@contextmanager
def log_timing(
    logger: logging.Logger, operation: str, level: int = logging.INFO
):
    """
    Context manager to log the timing of operations.

    Parameters
    ----------
    logger : logging.Logger
        Logger to use for timing messages
    operation : str
        Description of the operation being timed
    level : int
        Logging level for timing messages

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> with log_timing(logger, "computing player rankings"):
    ...     rankings = engine.rank_players(matches_df, players_df)
    """
    start_time = time.time()
    logger.log(level, f"Starting {operation}")

    try:
        yield
        elapsed = time.time() - start_time
        logger.log(level, f"Completed {operation} in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed {operation} after {elapsed:.2f}s: {e}")
        raise


def log_performance(logger: logging.Logger):
    """
    Decorator to log function performance metrics.

    Parameters
    ----------
    logger : logging.Logger
        Logger to use for performance messages

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> @log_performance(logger)
    ... def expensive_function():
    ...     # do work
    ...     return result
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"

            logger.debug(
                f"Calling {func_name} with args={len(args)}, kwargs={list(kwargs.keys())}"
            )

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{func_name} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func_name} failed after {elapsed:.2f}s: {e}")
                raise

        return wrapper

    return decorator


def log_dataframe_stats(
    logger: logging.Logger, df, name: str, level: int = logging.DEBUG
):
    """
    Log statistics about a DataFrame.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    df : polars.DataFrame or pandas.DataFrame
        DataFrame to analyze
    name : str
        Name/description of the DataFrame
    level : int
        Logging level
    """
    if df is None:
        logger.log(level, f"{name}: None")
        return

    try:
        # Handle both polars and pandas DataFrames
        if hasattr(df, "height"):  # Polars
            rows, cols = df.height, df.width
            memory_usage = (
                df.estimated_size("mb")
                if hasattr(df, "estimated_size")
                else "unknown"
            )
        else:  # Pandas
            rows, cols = df.shape
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB

        logger.log(
            level, f"{name}: {rows:,} rows × {cols} cols, ~{memory_usage:.1f}MB"
        )

        # Log column info for debug level
        if level <= logging.DEBUG:
            if hasattr(df, "dtypes"):  # Polars
                col_info = [
                    f"{col}({dtype})"
                    for col, dtype in zip(df.columns, df.dtypes)
                ]
            else:  # Pandas
                col_info = [
                    f"{col}({dtype})" for col, dtype in df.dtypes.items()
                ]
            logger.debug(f"{name} columns: {', '.join(col_info)}")

    except Exception as e:
        logger.warning(f"Could not log stats for {name}: {e}")


def log_function_entry(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Decorator to log function entry with parameters.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    level : int
        Logging level for entry messages
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"

            # Log function entry
            param_info = []
            if args:
                param_info.append(f"args={len(args)}")
            if kwargs:
                param_info.append(f"kwargs={list(kwargs.keys())}")

            param_str = ", ".join(param_info) if param_info else "no parameters"
            logger.log(level, f"Entering {func_name} with {param_str}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_algorithm_convergence(
    logger: logging.Logger,
    iteration: int,
    delta: float,
    threshold: float,
    algorithm: str = "algorithm",
):
    """
    Log algorithm convergence information.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    iteration : int
        Current iteration number
    delta : float
        Current convergence delta
    threshold : float
        Convergence threshold
    algorithm : str
        Name of the algorithm
    """
    if delta <= threshold:
        logger.info(
            f"{algorithm} converged at iteration {iteration} with delta {delta:.2e}"
        )
    else:
        logger.debug(
            f"{algorithm} iteration {iteration}: delta={delta:.2e} (threshold={threshold:.2e})"
        )


class ProgressLogger:
    """
    Context manager for logging progress of long-running operations.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> with ProgressLogger(logger, "processing tournaments", total=100) as progress:
    ...     for i in range(100):
    ...         # do work
    ...         progress.update(i + 1)
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        total: Optional[int] = None,
        update_interval: int = 10,
    ):
        self.logger = logger
        self.operation = operation
        self.total = total
        self.update_interval = update_interval
        self.start_time = None
        self.last_update = 0

    def __enter__(self):
        self.start_time = time.time()
        if self.total:
            self.logger.info(f"Starting {self.operation} (0/{self.total})")
        else:
            self.logger.info(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {elapsed:.2f}s")
        else:
            self.logger.error(
                f"Failed {self.operation} after {elapsed:.2f}s: {exc_val}"
            )

    def update(self, current: int, message: Optional[str] = None):
        """Update progress."""
        if (
            current - self.last_update >= self.update_interval
            or current == self.total
        ):
            elapsed = time.time() - self.start_time

            if self.total:
                pct = (current / self.total) * 100
                rate = current / elapsed if elapsed > 0 else 0
                eta = (self.total - current) / rate if rate > 0 else 0

                msg = f"{self.operation}: {current}/{self.total} ({pct:.1f}%) - {rate:.1f}/s"
                if eta > 0:
                    msg += f" - ETA: {eta:.1f}s"
                if message:
                    msg += f" - {message}"
            else:
                rate = current / elapsed if elapsed > 0 else 0
                msg = f"{self.operation}: {current} items - {rate:.1f}/s"
                if message:
                    msg += f" - {message}"

            self.logger.info(msg)
            self.last_update = current


# Initialize the default logger
_default_logger = None


def get_default_logger() -> logging.Logger:
    """Get the default logger for the rankings package."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger


# Convenience functions using default logger
def log_info(message: str):
    """Log an info message using the default logger."""
    get_default_logger().info(message)


def log_debug(message: str):
    """Log a debug message using the default logger."""
    get_default_logger().debug(message)


def log_warning(message: str):
    """Log a warning message using the default logger."""
    get_default_logger().warning(message)


def log_error(message: str):
    """Log an error message using the default logger."""
    get_default_logger().error(message)
