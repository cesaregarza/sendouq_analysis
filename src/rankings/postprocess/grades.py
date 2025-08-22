"""
Grade systems for ranking classifications.

This module provides various grade systems to categorize players based on
their scores, percentiles, or other metrics.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import polars as pl


class GradeSystem(ABC):
    """Abstract base class for grade systems."""

    @abstractmethod
    def assign_grades(
        self, df: pl.DataFrame, column_name: str = "grade"
    ) -> pl.DataFrame:
        """
        Assign grades to a DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with ranking data
        column_name : str
            Name for the grade column

        Returns
        -------
        pl.DataFrame
            DataFrame with added grade column
        """
        pass


class ScoreGradeSystem(GradeSystem):
    """Grade system based on absolute score thresholds."""

    def __init__(
        self,
        score_column: str = "score",
        thresholds: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize score-based grade system.

        Parameters
        ----------
        score_column : str
            Column containing scores
        thresholds : List[float], optional
            Score thresholds for grades
        labels : List[str], optional
            Grade labels
        """
        self.score_column = score_column
        self.thresholds = thresholds or [-3, -1, 0, 0.8, 1.5, 2.4, 4, 5]
        # Based on +3, +2, and +1 thresholds of 20, 37.5, and 60 points when
        # score_multiplier=25
        self.labels = labels or [
            "A-",
            "A",
            "A+",
            "S-",
            "S",
            "S+",
            "X",
            "X+",
            "X★",
        ]

    def assign_grades(
        self, df: pl.DataFrame, column_name: str = "grade"
    ) -> pl.DataFrame:
        """Assign grades based on score thresholds."""
        if self.score_column not in df.columns:
            raise ValueError(
                f"Column '{self.score_column}' not found in DataFrame"
            )

        return df.with_columns(
            pl.col(self.score_column)
            .cut(breaks=self.thresholds, labels=self.labels)
            .alias(column_name)
        )


class PercentileGradeSystem(GradeSystem):
    """Grade system based on percentile rankings."""

    def __init__(
        self,
        rank_column: str = "rank",
        percentiles: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize percentile-based grade system.

        Parameters
        ----------
        rank_column : str
            Column containing ranks
        percentiles : List[float], optional
            Percentile cutoffs (0.0 to 1.0)
        labels : List[str], optional
            Grade labels
        """
        self.rank_column = rank_column
        self.percentiles = percentiles or [
            0.003,
            0.03,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.97,
        ]
        self.labels = labels or [
            "X★",
            "X+",
            "X",
            "S+",
            "S",
            "S-",
            "A+",
            "A",
            "A-",
        ]

    def assign_grades(
        self, df: pl.DataFrame, column_name: str = "grade"
    ) -> pl.DataFrame:
        """Assign grades based on percentile rankings."""
        if self.rank_column not in df.columns:
            raise ValueError(
                f"Column '{self.rank_column}' not found in DataFrame"
            )

        total_players = len(df)

        # Calculate percentile for each player
        df = df.with_columns(
            (pl.col(self.rank_column) / total_players).alias("_percentile")
        )

        # Assign grades based on percentiles
        df = df.with_columns(
            pl.col("_percentile")
            .cut(breaks=self.percentiles, labels=self.labels)
            .alias(column_name)
        )

        # Clean up temporary column
        return df.drop("_percentile")


class EloGradeSystem(GradeSystem):
    """Grade system mimicking traditional Elo rating categories."""

    def __init__(
        self,
        score_column: str = "display_score",
        scale_factor: float = 400.0,
        base_rating: float = 1500.0,
    ):
        """
        Initialize Elo-style grade system.

        Parameters
        ----------
        score_column : str
            Column containing scores
        scale_factor : float
            Scale factor for Elo conversion
        base_rating : float
            Base Elo rating
        """
        self.score_column = score_column
        self.scale_factor = scale_factor
        self.base_rating = base_rating

        # Traditional Elo thresholds
        self.thresholds = [1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
        self.labels = [
            "Beginner",
            "Novice",
            "Intermediate",
            "Advanced",
            "Expert",
            "Master",
            "Grandmaster",
            "Super GM",
            "Elite",
        ]

    def assign_grades(
        self, df: pl.DataFrame, column_name: str = "grade"
    ) -> pl.DataFrame:
        """Assign grades based on Elo-style ratings."""
        if self.score_column not in df.columns:
            raise ValueError(
                f"Column '{self.score_column}' not found in DataFrame"
            )

        # Convert scores to Elo-style ratings
        df = df.with_columns(
            (
                self.base_rating + pl.col(self.score_column) * self.scale_factor
            ).alias("_elo_rating")
        )

        # Assign grades
        df = df.with_columns(
            pl.col("_elo_rating")
            .cut(breaks=self.thresholds, labels=self.labels)
            .alias(column_name)
        )

        return df.drop("_elo_rating")


class SendouqGradeSystem(ScoreGradeSystem):
    """Sendouq-specific grade system based on score thresholds."""

    def __init__(self):
        """Initialize Sendouq grade system with default thresholds."""
        super().__init__(
            score_column="score",
            thresholds=[-3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
            labels=[
                "C",
                "C+",
                "B-",
                "B",
                "B+",
                "A-",
                "A",
                "A+",
                "S-",
                "S",
                "S+",
                "X-",
                "X",
                "X+",
                "X★",
            ],
        )


class SendouqPercentileGradeSystem(PercentileGradeSystem):
    """Sendouq-specific grade system based on percentiles."""

    def __init__(self):
        """Initialize Sendouq percentile grade system."""
        super().__init__(
            rank_column="rank",
            # Top 0.3%, 3%, 10%, 20%, 35%, 50%, 65%, 80%, 90%
            percentiles=[0.003, 0.03, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90],
            labels=["X★", "X+", "X", "S+", "S", "S-", "A+", "A", "A-", "B"],
        )


def create_grade_system(system_type: str = "score", **kwargs) -> GradeSystem:
    """
    Factory function to create grade systems.

    Parameters
    ----------
    system_type : str
        Type of grade system: "score", "percentile", "elo", "sendouq", "sendouq_percentile"
    **kwargs
        Additional arguments for the specific grade system

    Returns
    -------
    GradeSystem
        Configured grade system instance
    """
    systems = {
        "score": ScoreGradeSystem,
        "percentile": PercentileGradeSystem,
        "elo": EloGradeSystem,
        "sendouq": SendouqGradeSystem,
        "sendouq_percentile": SendouqPercentileGradeSystem,
    }

    if system_type not in systems:
        raise ValueError(f"Unknown grade system: {system_type}")

    system_class = systems[system_type]

    # Special handling for Sendouq systems (no kwargs)
    if system_type in ["sendouq", "sendouq_percentile"]:
        return system_class()

    return system_class(**kwargs)


def add_multiple_grade_systems(
    df: pl.DataFrame,
    systems: List[Union[str, tuple[str, dict], GradeSystem]],
    prefixes: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Add multiple grade systems to a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with ranking data
    systems : List[Union[str, tuple, GradeSystem]]
        List of grade systems to apply. Each can be:
        - str: Grade system type name
        - tuple: (system_type, kwargs_dict)
        - GradeSystem: Instance of a grade system
    prefixes : List[str], optional
        Prefixes for each grade column

    Returns
    -------
    pl.DataFrame
        DataFrame with multiple grade columns
    """
    if prefixes and len(prefixes) != len(systems):
        raise ValueError("Number of prefixes must match number of systems")

    result = df

    for i, system_spec in enumerate(systems):
        # Create grade system instance
        if isinstance(system_spec, str):
            system = create_grade_system(system_spec)
        elif isinstance(system_spec, tuple):
            system_type, kwargs = system_spec
            system = create_grade_system(system_type, **kwargs)
        elif isinstance(system_spec, GradeSystem):
            system = system_spec
        else:
            raise ValueError(f"Invalid system specification: {system_spec}")

        # Determine column name
        if prefixes:
            column_name = f"{prefixes[i]}_grade"
        else:
            column_name = f"grade_{i+1}"

        # Apply grade system
        result = system.assign_grades(result, column_name)

    return result


def compare_grade_distributions(
    df: pl.DataFrame,
    grade_columns: List[str],
) -> pl.DataFrame:
    """
    Compare grade distributions across multiple grade systems.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with multiple grade columns
    grade_columns : List[str]
        Names of grade columns to compare

    Returns
    -------
    pl.DataFrame
        Summary of grade distributions
    """
    distributions = []

    for col in grade_columns:
        if col not in df.columns:
            continue

        # Count grades
        grade_counts = (
            df.group_by(col)
            .agg(pl.count().alias("count"))
            .with_columns(
                (pl.col("count") / len(df) * 100).alias("percentage"),
                pl.lit(col).alias("system"),
            )
            .rename({col: "grade"})
        )

        distributions.append(grade_counts)

    if not distributions:
        return pl.DataFrame()

    # Combine all distributions
    result = pl.concat(distributions)

    # Pivot for easier comparison
    return result.pivot(
        values="percentage",
        index="grade",
        columns="system",
    ).fill_null(0)
