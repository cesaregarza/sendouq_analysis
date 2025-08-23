"""Result dataclasses for ranking algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from typing import Any


@dataclass
class RankResult:
    """Results from a ranking algorithm run."""

    scores: np.ndarray
    ids: list[Any]

    win_pagerank: np.ndarray | None = None
    loss_pagerank: np.ndarray | None = None

    teleport: np.ndarray | None = None
    exposure: np.ndarray | None = None

    lambda_used: float | None = None
    iterations: int | None = None
    converged: bool = True

    convergence_history: list[float] | None = None
    computation_time: float | None = None

    def to_dataframe(
        self,
        id_column: str = "player_id",
        score_column: str = "score",
    ) -> pl.DataFrame:
        """Convert results to a Polars DataFrame.

        Args:
            id_column: Name for ID column. Defaults to "player_id".
            score_column: Name for score column. Defaults to "score".

        Returns:
            DataFrame with results.
        """
        dataframe = pl.DataFrame(
            {id_column: self.ids, score_column: self.scores.tolist()}
        )

        if self.win_pagerank is not None:
            dataframe = dataframe.with_columns(
                pl.Series("win_pagerank", self.win_pagerank)
            )

        if self.loss_pagerank is not None:
            dataframe = dataframe.with_columns(
                pl.Series("loss_pagerank", self.loss_pagerank)
            )

        if self.exposure is not None:
            dataframe = dataframe.with_columns(
                pl.Series("exposure", self.exposure)
            )

        if self.teleport is not None:
            dataframe = dataframe.with_columns(
                pl.Series("teleport", self.teleport)
            )

        return dataframe

    def get_top_n(self, count: int = 10) -> pl.DataFrame:
        """Get top N ranked entities.

        Args:
            count: Number of top entities to return. Defaults to 10.

        Returns:
            DataFrame with top N entities sorted by score.
        """
        dataframe = self.to_dataframe()
        return dataframe.sort("score", descending=True).head(count)


@dataclass
class TickTockResult(RankResult):
    """Results specific to Tick-Tock algorithm."""

    tournament_influence: dict[Any, float] | None = None
    retrospective_strength: np.ndarray | None = None
    tick_history: list[np.ndarray] | None = None
    tock_history: list[np.ndarray] | None = None
    denominators: np.ndarray | None = None


@dataclass
class ExposureLogOddsResult(RankResult):
    """Results specific to Exposure Log-Odds algorithm."""

    surprisal_weights: np.ndarray | None = None
    active_mask: np.ndarray | None = None
    raw_scores: np.ndarray | None = None
    decay_factors: np.ndarray | None = None


@dataclass
class PipelineResult:
    """Complete pipeline execution results."""

    result: RankResult
    algorithm: str
    config: dict[str, Any] = field(default_factory=dict)
    intermediate: dict[str, Any] | None = None
    total_time: float | None = None
    memory_usage: float | None = None
    graded_dataframe: pl.DataFrame | None = None

    def to_dataframe(self) -> pl.DataFrame:
        """Get main result as DataFrame.

        Returns:
            DataFrame with algorithm results and optional grading.
        """
        dataframe = self.result.to_dataframe()

        dataframe = dataframe.with_columns(
            pl.lit(self.algorithm).alias("algorithm")
        )

        if self.graded_dataframe is not None:
            id_column = dataframe.columns[0]
            dataframe = dataframe.join(
                self.graded_dataframe.select(
                    [id_column, "grade", "display_score"]
                ),
                on=id_column,
                how="left",
            )

        return dataframe


@dataclass
class ValidationResult:
    """Results from validation checks."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    pagerank_normalized: bool = True
    teleport_normalized: bool = True
    scores_finite: bool = True
    no_duplicates: bool = True

    def __str__(self) -> str:
        """String representation of validation results.

        Returns:
            Human-readable validation status.
        """
        if self.is_valid:
            message = "Validation passed"
            if self.warnings:
                message += f" with {len(self.warnings)} warning(s)"
        else:
            message = f"Validation failed with {len(self.errors)} error(s)"
        return message


@dataclass
class BenchmarkResult:
    """Results from performance benchmarking."""

    algorithm: str
    dataset_size: int

    total_time: float
    pagerank_time: float
    preprocessing_time: float
    postprocessing_time: float

    peak_memory_mb: float

    iterations: int
    converged: bool
    final_error: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for easy serialization.

        Returns:
            Dictionary with all benchmark results.
        """
        return {
            "algorithm": self.algorithm,
            "dataset_size": self.dataset_size,
            "total_time": self.total_time,
            "pagerank_time": self.pagerank_time,
            "preprocessing_time": self.preprocessing_time,
            "postprocessing_time": self.postprocessing_time,
            "peak_memory_mb": self.peak_memory_mb,
            "iterations": self.iterations,
            "converged": self.converged,
            "final_error": self.final_error,
        }
