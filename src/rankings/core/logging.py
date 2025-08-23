"""
Centralized logging configuration for the rankings package.

This module provides consistent logging setup across all components of the
rankings system, including structured logging, performance metrics, and
debug information.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    pass

F = TypeVar("F", bound=Callable[..., Any])


def setup_logging(
    level: str | int = logging.INFO,
    log_file: str | Path | None = None,
    format_style: str = "detailed",
    include_timestamp: bool = True,
) -> logging.Logger:
    """Set up centralized logging for the rankings package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to logging.INFO.
        log_file: Optional file to write logs to. Defaults to None.
        format_style: Format style: "simple", "detailed", or "json". Defaults to "detailed".
        include_timestamp: Whether to include timestamps in log messages. Defaults to True.

    Returns:
        Configured logger instance.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = logging.getLogger("rankings")
    logger.setLevel(level)

    logger.handlers.clear()

    if format_style == "simple":
        format_string = "%(levelname)s: %(message)s"
    elif format_style == "json":
        format_string = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
    else:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

    formatter = logging.Formatter(format_string)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module/component.

    Args:
        name: Name of the component (usually __name__).

    Returns:
        Logger instance for the component.
    """
    return logging.getLogger(f"rankings.{name}")


@contextmanager
def log_timing(
    logger: logging.Logger, operation: str, level: int = logging.INFO
):
    """Context manager to log the timing of operations.

    Args:
        logger: Logger to use for timing messages.
        operation: Description of the operation being timed.
        level: Logging level for timing messages. Defaults to logging.INFO.

    Examples:
        >>> logger = get_logger(__name__)
        >>> with log_timing(logger, "computing player rankings"):
        ...     rankings = engine.rank_players(matches_df, players_df)
    """
    start_time = time.time()
    logger.log(level, f"Starting {operation}")

    try:
        yield
        elapsed_time = time.time() - start_time
        logger.log(level, f"Completed {operation} in {elapsed_time:.2f}s")
    except Exception as exception:
        elapsed_time = time.time() - start_time
        logger.error(
            f"Failed {operation} after {elapsed_time:.2f}s: {exception}"
        )
        raise


def log_performance(logger: logging.Logger):
    """Decorator to log function performance metrics.

    Args:
        logger: Logger to use for performance messages.

    Examples:
        >>> logger = get_logger(__name__)
        >>> @log_performance(logger)
        ... def expensive_function():
        ...     # do work
        ...     return result
    """

    def decorator(function: F) -> F:
        @wraps(function)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{function.__module__}.{function.__name__}"

            logger.debug(
                f"Calling {function_name} with args={len(args)}, kwargs={list(kwargs.keys())}"
            )

            try:
                result = function(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.info(f"{function_name} completed in {elapsed_time:.2f}s")
                return result
            except Exception as exception:
                elapsed_time = time.time() - start_time
                logger.error(
                    f"{function_name} failed after {elapsed_time:.2f}s: {exception}"
                )
                raise

        return wrapper

    return decorator


def log_dataframe_stats(
    logger: logging.Logger,
    dataframe: Any,
    name: str,
    level: int = logging.DEBUG,
) -> None:
    """Log statistics about a DataFrame.

    Args:
        logger: Logger instance.
        dataframe: DataFrame to analyze (polars or pandas).
        name: Name/description of the DataFrame.
        level: Logging level. Defaults to logging.DEBUG.
    """
    if dataframe is None:
        logger.log(level, f"{name}: None")
        return

    try:
        if hasattr(dataframe, "height"):
            rows, cols = dataframe.height, dataframe.width
            memory_usage = (
                dataframe.estimated_size("mb")
                if hasattr(dataframe, "estimated_size")
                else "unknown"
            )
        else:
            rows, cols = dataframe.shape
            memory_usage = dataframe.memory_usage(deep=True).sum() / 1024 / 1024

        logger.log(
            level, f"{name}: {rows:,} rows × {cols} cols, ~{memory_usage:.1f}MB"
        )

        if level <= logging.DEBUG:
            if hasattr(dataframe, "dtypes"):
                column_info = [
                    f"{column_name}({data_type})"
                    for column_name, data_type in zip(
                        dataframe.columns, dataframe.dtypes
                    )
                ]
            else:
                column_info = [
                    f"{column_name}({data_type})"
                    for column_name, data_type in dataframe.dtypes.items()
                ]
            logger.debug(f"{name} columns: {', '.join(column_info)}")

    except Exception as exception:
        logger.warning(f"Could not log stats for {name}: {exception}")


def log_function_entry(logger: logging.Logger, level: int = logging.DEBUG):
    """Decorator to log function entry with parameters.

    Args:
        logger: Logger instance.
        level: Logging level for entry messages. Defaults to logging.DEBUG.
    """

    def decorator(function: F) -> F:
        @wraps(function)
        def wrapper(*args, **kwargs):
            function_name = f"{function.__module__}.{function.__name__}"

            parameter_info = []
            if args:
                parameter_info.append(f"args={len(args)}")
            if kwargs:
                parameter_info.append(f"kwargs={list(kwargs.keys())}")

            parameter_string = (
                ", ".join(parameter_info) if parameter_info else "no parameters"
            )
            logger.log(
                level, f"Entering {function_name} with {parameter_string}"
            )

            return function(*args, **kwargs)

        return wrapper

    return decorator


def log_algorithm_convergence(
    logger: logging.Logger,
    iteration: int,
    delta: float,
    threshold: float,
    algorithm: str = "algorithm",
) -> None:
    """Log algorithm convergence information.

    Args:
        logger: Logger instance.
        iteration: Current iteration number.
        delta: Current convergence delta.
        threshold: Convergence threshold.
        algorithm: Name of the algorithm. Defaults to "algorithm".
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
        total: int | None = None,
        update_interval: int = 10,
    ) -> None:
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
        elapsed_time = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation} in {elapsed_time:.2f}s"
            )
        else:
            self.logger.error(
                f"Failed {self.operation} after {elapsed_time:.2f}s: {exc_val}"
            )

    def update(self, current: int, message: str | None = None) -> None:
        """Update progress."""
        if (
            current - self.last_update >= self.update_interval
            or current == self.total
        ):
            elapsed_time = time.time() - self.start_time

            if self.total:
                percentage = (current / self.total) * 100
                rate = current / elapsed_time if elapsed_time > 0 else 0
                estimated_time_remaining = (
                    (self.total - current) / rate if rate > 0 else 0
                )

                log_message = f"{self.operation}: {current}/{self.total} ({percentage:.1f}%) - {rate:.1f}/s"
                if estimated_time_remaining > 0:
                    log_message += f" - ETA: {estimated_time_remaining:.1f}s"
                if message:
                    log_message += f" - {message}"
            else:
                rate = current / elapsed_time if elapsed_time > 0 else 0
                log_message = (
                    f"{self.operation}: {current} items - {rate:.1f}/s"
                )
                if message:
                    log_message += f" - {message}"

            self.logger.info(log_message)
            self.last_update = current


_default_logger: logging.Logger | None = None


def get_default_logger() -> logging.Logger:
    """Get the default logger for the rankings package."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger


def log_info(message: str) -> None:
    """Log an info message using the default logger."""
    get_default_logger().info(message)


def log_debug(message: str) -> None:
    """Log a debug message using the default logger."""
    get_default_logger().debug(message)


def log_warning(message: str) -> None:
    """Log a warning message using the default logger."""
    get_default_logger().warning(message)


def log_error(message: str) -> None:
    """Log an error message using the default logger."""
    get_default_logger().error(message)
