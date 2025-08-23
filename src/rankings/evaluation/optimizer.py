"""
Hyperparameter optimization for tournament rating engines.

This module provides optimization routines for finding the best hyperparameters
using cross-validation loss as the objective function.
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Any, Type

import numpy as np
import polars as pl

from rankings.analysis.engine import RatingEngine
from rankings.evaluation.cross_validation import cross_validate_ratings

logger = logging.getLogger(__name__)


class GridSearchOptimizer:
    """
    Grid search optimizer for rating engine hyperparameters.

    Exhaustively searches through a grid of parameter values.
    """

    def __init__(
        self,
        param_grid: dict[str, list[Any]],
        engine_class: type[RatingEngine] = RatingEngine,
        n_splits: int = 5,
        fit_alpha: bool = True,
        regularization_lambda: float = 0.0,
        verbose: bool = True,
    ):
        """
        Initialize grid search optimizer.

        Parameters
        ----------
        param_grid : dict[str, list[Any]
            Dictionary mapping parameter names to lists of values to try
        engine_class : type[RatingEngine]
            Rating engine class to optimize
        n_splits : int
            Number of CV folds
        fit_alpha : bool
            Whether to fit alpha parameter
        regularization_lambda : float
            L2 regularization strength
        verbose : bool
            Whether to print progress
        """
        self.param_grid = param_grid
        self.engine_class = engine_class
        self.n_splits = n_splits
        self.fit_alpha = fit_alpha
        self.regularization_lambda = regularization_lambda
        self.verbose = verbose

        # Results storage
        self.results_ = []
        self.best_params_ = None
        self.best_score_ = np.inf

    def fit(
        self,
        matches_df: pl.DataFrame,
        players_df: pl.DataFrame | None = None,
        teams_df: pl.DataFrame | None = None,
        ranking_entity: str = "player",
        prediction_entity: str = "team",
        agg_func: str = "mean",
    ) -> GridSearchOptimizer:
        """
        Run grid search optimization.

        Parameters
        ----------
        matches_df : pl.DataFrame
            Matches data
        players_df : pl.DataFrame | None
            Player metadata
        teams_df : pl.DataFrame | None
            Team metadata
        ranking_entity : str
            Entity type to rank ("player" or "team")
        prediction_entity : str
            Entity type to predict on ("player" or "team")
        agg_func : str
            Aggregation function for converting player ratings to team ratings

        Returns
        -------
        self
            Fitted optimizer
        """
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        all_combinations = list(product(*param_values))
        n_combinations = len(all_combinations)

        if self.verbose:
            print(f"Grid search: {n_combinations} parameter combinations")

        # Evaluate each combination
        for i, values in enumerate(all_combinations):
            params = dict(zip(param_names, values))

            if self.verbose:
                print(f"\nEvaluating {i+1}/{n_combinations}: {params}")

            try:
                cv_results = cross_validate_ratings(
                    engine_class=self.engine_class,
                    engine_params=params,
                    matches_df=matches_df,
                    players_df=players_df,
                    teams_df=teams_df,
                    ranking_entity=ranking_entity,
                    prediction_entity=prediction_entity,
                    agg_func=agg_func,
                    n_splits=self.n_splits,
                    fit_alpha=self.fit_alpha,
                    regularization_lambda=self.regularization_lambda,
                )

                score = cv_results["regularized_loss"]

                self.results_.append(
                    {
                        "params": params,
                        "mean_loss": cv_results["avg_loss"],
                        "std_loss": cv_results["std_loss"],
                        "regularized_loss": score,
                        "cv_results": cv_results,
                    }
                )

                if score < self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params.copy()

                if self.verbose:
                    print(
                        f"Loss: {cv_results['avg_loss']:.4f} ± {cv_results['std_loss']:.4f}"
                    )

            except Exception as e:
                logger.error(f"Error with params {params}: {e}")
                self.results_.append({"params": params, "error": str(e)})

        if self.verbose:
            print(f"\nBest parameters: {self.best_params_}")
            print(f"Best score: {self.best_score_:.4f}")

        return self

    def get_results_summary(self) -> pl.DataFrame:
        """Get summary of all results as a DataFrame."""
        valid_results = [r for r in self.results_ if "error" not in r]

        if not valid_results:
            return pl.DataFrame()

        # Flatten parameters and create summary
        rows = []
        for result in valid_results:
            row = result["params"].copy()
            row["mean_loss"] = result["mean_loss"]
            row["std_loss"] = result["std_loss"]
            row["regularized_loss"] = result["regularized_loss"]
            rows.append(row)

        return pl.DataFrame(rows).sort("regularized_loss")


class BayesianOptimizer:
    """
    Bayesian optimization for rating engine hyperparameters.

    Uses a Gaussian Process to model the objective function and
    intelligently selects points to evaluate.
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]],
        param_types: dict[str, str] | None = None,
        engine_class: type[RatingEngine] = RatingEngine,
        n_splits: int = 5,
        n_initial: int = 5,
        n_iterations: int = 20,
        fit_alpha: bool = True,
        regularization_lambda: float = 0.0,
        verbose: bool = True,
    ):
        """
        Initialize Bayesian optimizer.

        Parameters
        ----------
        param_bounds : dict[str, tuple[float, float]
            Bounds for each parameter
        param_types : dict[str, str| None
            Type hints for parameters ("float", "int", "choice")
        engine_class : type[RatingEngine]
            Rating engine class
        n_splits : int
            Number of CV folds
        n_initial : int
            Number of random initial points
        n_iterations : int
            Total optimization iterations
        fit_alpha : bool
            Whether to fit alpha
        regularization_lambda : float
            L2 regularization
        verbose : bool
            Print progress
        """
        self.param_bounds = param_bounds
        self.param_types = param_types or {}
        self.engine_class = engine_class
        self.n_splits = n_splits
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.fit_alpha = fit_alpha
        self.regularization_lambda = regularization_lambda
        self.verbose = verbose

        # Results
        self.results_ = []
        self.best_params_ = None
        self.best_score_ = np.inf

    def fit(
        self,
        matches_df: pl.DataFrame,
        players_df: pl.DataFrame | None = None,
        teams_df: pl.DataFrame | None = None,
        ranking_entity: str = "player",
        prediction_entity: str = "team",
        agg_func: str = "mean",
    ) -> BayesianOptimizer:
        """
        Run Bayesian optimization.

        Note: This is a simplified implementation. For production use,
        consider using libraries like scikit-optimize or Optuna.
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Categorical, Integer, Real
            from skopt.utils import use_named_args
        except ImportError:
            raise ImportError(
                "scikit-optimize required for Bayesian optimization. "
                "Install with: pip install scikit-optimize"
            )

        # Create search space
        dimensions = []
        param_names = []

        for param_name, bounds in self.param_bounds.items():
            param_names.append(param_name)
            param_type = self.param_types.get(param_name, "float")

            if param_type == "int":
                dimensions.append(
                    Integer(bounds[0], bounds[1], name=param_name)
                )
            elif param_type == "choice":
                dimensions.append(Categorical(bounds, name=param_name))
            else:
                dimensions.append(Real(bounds[0], bounds[1], name=param_name))

        # Define objective function
        @use_named_args(dimensions)
        def objective(**params):
            if self.verbose:
                print(f"\nEvaluating: {params}")

            try:
                cv_results = cross_validate_ratings(
                    engine_class=self.engine_class,
                    engine_params=params,
                    matches_df=matches_df,
                    players_df=players_df,
                    teams_df=teams_df,
                    ranking_entity=ranking_entity,
                    prediction_entity=prediction_entity,
                    agg_func=agg_func,
                    n_splits=self.n_splits,
                    fit_alpha=self.fit_alpha,
                    regularization_lambda=self.regularization_lambda,
                )

                score = cv_results["regularized_loss"]

                self.results_.append(
                    {
                        "params": params.copy(),
                        "score": score,
                        "cv_results": cv_results,
                    }
                )

                if score < self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params.copy()

                if self.verbose:
                    print(f"Loss: {score:.4f}")

                return score

            except Exception as e:
                logger.error(f"Error during optimization: {e}")
                return np.inf

        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.n_iterations,
            n_initial_points=self.n_initial,
            acq_func="EI",  # Expected Improvement
            random_state=42,
        )

        if self.verbose:
            print(f"\nOptimization complete!")
            print(f"Best parameters: {self.best_params_}")
            print(f"Best score: {self.best_score_:.4f}")

        return self


def optimize_rating_engine(
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame | None = None,
    teams_df: pl.DataFrame | None = None,
    ranking_entity: str = "player",
    prediction_entity: str = "team",
    agg_func: str = "mean",
    method: str = "grid",
    param_space: dict[str, Any] | None = None,
    n_splits: int = 5,
    **kwargs,
) -> dict[str, Any]:
    """
    Convenience function for optimizing rating engine parameters.

    Parameters
    ----------
    matches_df : pl.DataFrame
        Matches data
    players_df : pl.DataFrame | None
        Player metadata
    teams_df : pl.DataFrame | None
        Team metadata
    ranking_entity : str
        Entity type to rank ("player" or "team")
    prediction_entity : str
        Entity type to predict on ("player" or "team")
    agg_func : str
        Aggregation function for converting player ratings to team ratings
    method : str
        "grid" or "bayesian"
    param_space : dict[str, Any] | None
        Parameter search space
    n_splits : int
        Number of CV folds
    **kwargs
        Additional arguments for optimizer

    Returns
    -------
    dict[str, Any]
        Optimization results
    """
    if param_space is None:
        # Default parameter space
        if method == "grid":
            param_space = {
                "decay_half_life_days": [7.0, 14.0, 30.0, 60.0],
                "damping_factor": [0.8, 0.85, 0.9],
                "beta": [0.0, 0.5, 1.0],
                "influence_agg_method": ["top_10_sum", "top_20_sum", "max"],
            }
        else:
            param_space = {
                "decay_half_life_days": (7.0, 90.0),
                "damping_factor": (0.7, 0.95),
                "beta": (0.0, 1.0),
            }

    if method == "grid":
        optimizer = GridSearchOptimizer(
            param_grid=param_space, n_splits=n_splits, **kwargs
        )
    else:
        optimizer = BayesianOptimizer(
            param_bounds=param_space, n_splits=n_splits, **kwargs
        )

    optimizer.fit(
        matches_df=matches_df,
        players_df=players_df,
        teams_df=teams_df,
        ranking_entity=ranking_entity,
        prediction_entity=prediction_entity,
        agg_func=agg_func,
    )

    return {
        "best_params": optimizer.best_params_,
        "best_score": optimizer.best_score_,
        "results": optimizer.results_,
        "optimizer": optimizer,
    }
