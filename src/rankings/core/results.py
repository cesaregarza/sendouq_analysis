"""Result dataclasses for ranking algorithms."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl


@dataclass
class RankResult:
    """Results from a ranking algorithm run."""

    # Core results
    scores: np.ndarray
    ids: list

    # PageRank vectors
    win_pr: Optional[np.ndarray] = None
    loss_pr: Optional[np.ndarray] = None

    # Teleport and exposure
    teleport: Optional[np.ndarray] = None
    exposure: Optional[np.ndarray] = None

    # Algorithm parameters used
    lambda_used: Optional[float] = None
    iterations: Optional[int] = None
    converged: bool = True

    # Diagnostics
    convergence_history: Optional[List[float]] = None
    computation_time: Optional[float] = None

    def to_dataframe(
        self,
        id_col: str = "player_id",
        score_col: str = "score",
    ) -> pl.DataFrame:
        """
        Convert results to a Polars DataFrame.

        Args:
            id_col: Name for ID column
            score_col: Name for score column

        Returns:
            DataFrame with results
        """
        df = pl.DataFrame({id_col: self.ids, score_col: self.scores.tolist()})

        if self.win_pr is not None:
            df = df.with_columns(pl.Series("win_pr", self.win_pr))

        if self.loss_pr is not None:
            df = df.with_columns(pl.Series("loss_pr", self.loss_pr))

        if self.exposure is not None:
            df = df.with_columns(pl.Series("exposure", self.exposure))

        if self.teleport is not None:
            df = df.with_columns(pl.Series("teleport", self.teleport))

        return df

    def get_top_n(self, n: int = 10) -> pl.DataFrame:
        """Get top N ranked entities."""
        df = self.to_dataframe()
        return df.sort("score", descending=True).head(n)


@dataclass
class TickTockResult(RankResult):
    """Results specific to Tick-Tock algorithm."""

    # Tournament influence
    tournament_influence: Optional[Dict[Any, float]] = None

    # Retrospective strength
    retrospective_strength: Optional[np.ndarray] = None

    # Tick history
    tick_history: Optional[List[np.ndarray]] = None
    tock_history: Optional[List[np.ndarray]] = None

    # Denominators used
    denominators: Optional[np.ndarray] = None


@dataclass
class ExposureLogOddsResult(RankResult):
    """Results specific to Exposure Log-Odds algorithm."""

    # Surprisal weights
    surprisal_weights: Optional[np.ndarray] = None

    # Active player mask
    active_mask: Optional[np.ndarray] = None

    # Raw scores before transformation
    raw_scores: Optional[np.ndarray] = None

    # Decay factors applied
    decay_factors: Optional[np.ndarray] = None


@dataclass
class PipelineResult:
    """Complete pipeline execution results."""

    # Main result
    result: RankResult

    # Algorithm used
    algorithm: str

    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)

    # Intermediate results (if requested)
    intermediate: Optional[Dict[str, Any]] = None

    # Diagnostics
    total_time: Optional[float] = None
    memory_usage: Optional[float] = None

    # Graded results (if post-processing applied)
    graded_df: Optional[pl.DataFrame] = None

    def to_dataframe(self) -> pl.DataFrame:
        """Get main result as DataFrame."""
        df = self.result.to_dataframe()

        # Add algorithm column
        df = df.with_columns(pl.lit(self.algorithm).alias("algorithm"))

        # Add grading if available
        if self.graded_df is not None:
            # Join grading columns
            id_col = df.columns[0]
            df = df.join(
                self.graded_df.select([id_col, "grade", "display_score"]),
                on=id_col,
                how="left",
            )

        return df


@dataclass
class ValidationResult:
    """Results from validation checks."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Detailed checks
    pagerank_normalized: bool = True
    teleport_normalized: bool = True
    scores_finite: bool = True
    no_duplicates: bool = True

    def __str__(self) -> str:
        """String representation of validation results."""
        if self.is_valid:
            msg = "Validation passed"
            if self.warnings:
                msg += f" with {len(self.warnings)} warning(s)"
        else:
            msg = f"Validation failed with {len(self.errors)} error(s)"
        return msg


@dataclass
class BenchmarkResult:
    """Results from performance benchmarking."""

    algorithm: str
    dataset_size: int

    # Timing
    total_time: float
    pagerank_time: float
    preprocessing_time: float
    postprocessing_time: float

    # Memory
    peak_memory_mb: float

    # Convergence
    iterations: int
    converged: bool
    final_error: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
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
