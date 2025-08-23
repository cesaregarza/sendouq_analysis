"""Logistic regression calibration for match predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Results from calibration fitting."""

    beta: float  # Scaling parameter for rating difference
    bias: float  # Bias term
    log_loss: float  # Cross-validated log loss
    brier_score: float  # Cross-validated Brier score
    accuracy: float  # Cross-validated accuracy
    n_matches: int  # Number of matches used
    format_betas: dict[str, float] | None = None  # Format-specific betas

    def predict_probability(
        self, rating_diff: float, format: str | None = None
    ) -> float:
        """Calculate win probability given rating difference."""
        beta = self.beta
        if format and self.format_betas and format in self.format_betas:
            beta = self.format_betas[format]

        x = beta * rating_diff + self.bias
        return 1.0 / (1.0 + np.exp(-x))


class MatchCalibrator:
    """Calibrate match predictions using logistic regression."""

    def __init__(self, use_format_specific: bool = False):
        """
        Initialize calibrator.

        Parameters
        ----------
        use_format_specific : bool
            Whether to fit separate parameters for different match formats
        """
        self.use_format_specific = use_format_specific
        self.result: CalibrationResult | None = None

    def prepare_match_data(
        self,
        matches_df: pl.DataFrame,
        team_ratings: dict[int, float],
        match_format_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Prepare match data for calibration.

        Parameters
        ----------
        matches_df : pl.DataFrame
            DataFrame with match results including winner_team_id and loser_team_id
        team_ratings : dict[int, float]
            Pre-calculated team log-ratings
        match_format_col : str | None
            Column name for match format (e.g., 'best_of')

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray | None]
            (rating_differences, outcomes, formats)
        """
        rating_diffs = []
        outcomes = []
        formats = [] if match_format_col else None

        for match in matches_df.to_dicts():
            winner_id = match.get("winner_team_id")
            loser_id = match.get("loser_team_id")

            if winner_id in team_ratings and loser_id in team_ratings:
                winner_rating = team_ratings[winner_id]
                loser_rating = team_ratings[loser_id]

                # Add from winner's perspective (outcome = 1)
                rating_diffs.append(winner_rating - loser_rating)
                outcomes.append(1)
                if formats is not None and match_format_col:
                    formats.append(match.get(match_format_col, "default"))

                # Add from loser's perspective (outcome = 0)
                rating_diffs.append(loser_rating - winner_rating)
                outcomes.append(0)
                if formats is not None and match_format_col:
                    formats.append(match.get(match_format_col, "default"))

        return (
            np.array(rating_diffs),
            np.array(outcomes),
            np.array(formats) if formats else None,
        )

    def fit(
        self,
        rating_diffs: np.ndarray,
        outcomes: np.ndarray,
        formats: np.ndarray | None = None,
        cv_splits: int = 5,
    ) -> CalibrationResult:
        """
        Fit logistic regression parameters.

        Parameters
        ----------
        rating_diffs : np.ndarray
            Rating differences (team_a - team_b)
        outcomes : np.ndarray
            Match outcomes (1 if team_a won, 0 otherwise)
        formats : np.ndarray | None
            Match formats for format-specific calibration
        cv_splits : int
            Number of cross-validation splits

        Returns
        -------
        CalibrationResult
            Fitted calibration parameters and metrics
        """
        logger.info(
            f"Fitting calibration on {len(outcomes)} match observations"
        )

        # Basic logistic regression
        X = rating_diffs.reshape(-1, 1)
        y = outcomes

        # Use time series split for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        # Cross-validation metrics
        cv_log_losses = []
        cv_brier_scores = []
        cv_accuracies = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit model
            model = LogisticRegression(
                penalty=None, solver="lbfgs", max_iter=1000
            )
            model.fit(X_train, y_train)

            # Validate
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            cv_log_losses.append(log_loss(y_val, y_pred_proba))
            cv_brier_scores.append(brier_score_loss(y_val, y_pred_proba))
            cv_accuracies.append(np.mean((y_pred_proba > 0.5) == y_val))

        # Final fit on all data
        final_model = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=1000
        )
        final_model.fit(X, y)

        beta = float(final_model.coef_[0, 0])
        bias = float(final_model.intercept_[0])

        logger.info(f"Fitted β={beta:.4f}, b={bias:.4f}")

        # Format-specific calibration if requested
        format_betas = None
        if self.use_format_specific and formats is not None:
            format_betas = self._fit_format_specific(
                rating_diffs, outcomes, formats, beta, bias
            )

        result = CalibrationResult(
            beta=beta,
            bias=bias,
            log_loss=np.mean(cv_log_losses),
            brier_score=np.mean(cv_brier_scores),
            accuracy=np.mean(cv_accuracies),
            n_matches=len(outcomes)
            // 2,  # Divide by 2 since we doubled the data
            format_betas=format_betas,
        )

        self.result = result
        return result

    def _fit_format_specific(
        self,
        rating_diffs: np.ndarray,
        outcomes: np.ndarray,
        formats: np.ndarray,
        base_beta: float,
        base_bias: float,
    ) -> dict[str, float]:
        """
        Fit format-specific beta multipliers.

        Parameters
        ----------
        rating_diffs : np.ndarray
            Rating differences
        outcomes : np.ndarray
            Match outcomes
        formats : np.ndarray
            Match formats
        base_beta : float
            Base beta parameter
        base_bias : float
            Base bias parameter

        Returns
        -------
        dict[str, float]
            Format-specific beta values
        """
        unique_formats = np.unique(formats)
        format_betas = {}

        for fmt in unique_formats:
            mask = formats == fmt
            X_fmt = rating_diffs[mask].reshape(-1, 1)
            y_fmt = outcomes[mask]

            if len(y_fmt) < 20:  # Skip if too few matches
                format_betas[fmt] = base_beta
                continue

            # Fit format-specific model
            model = LogisticRegression(
                penalty=None, solver="lbfgs", max_iter=1000
            )

            try:
                model.fit(X_fmt, y_fmt)
                format_betas[fmt] = float(model.coef_[0, 0])
                logger.info(
                    f"Format '{fmt}': β={format_betas[fmt]:.4f} (n={len(y_fmt)})"
                )
            except Exception as e:
                format_betas[fmt] = base_beta
                logger.warning(
                    f"Failed to fit format '{fmt}', using base β: {e}"
                )

        return format_betas

    def calibrate_from_matches(
        self,
        matches_df: pl.DataFrame,
        team_ratings: dict[int, float],
        match_format_col: str | None = None,
    ) -> CalibrationResult:
        """
        Complete calibration pipeline from match data.

        Parameters
        ----------
        matches_df : pl.DataFrame
            Match results DataFrame
        team_ratings : dict[int, float]
            Pre-calculated team ratings
        match_format_col : str | None
            Column for match format

        Returns
        -------
        CalibrationResult
            Calibration results
        """
        # Prepare data
        rating_diffs, outcomes, formats = self.prepare_match_data(
            matches_df, team_ratings, match_format_col
        )

        if len(rating_diffs) == 0:
            raise ValueError("No valid matches for calibration")

        # Fit calibration
        return self.fit(rating_diffs, outcomes, formats)

    def evaluate_calibration(
        self, rating_diffs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
    ) -> dict[str, float | np.ndarray]:
        """
        Evaluate calibration quality.

        Parameters
        ----------
        rating_diffs : np.ndarray
            Rating differences for test set
        outcomes : np.ndarray
            Actual outcomes for test set
        n_bins : int
            Number of bins for calibration plot

        Returns
        -------
        dict[str, float | np.ndarray]
            Calibration metrics and bin statistics
        """
        if self.result is None:
            raise ValueError("Must fit calibration first")

        # Get predictions
        predictions = [
            self.result.predict_probability(diff) for diff in rating_diffs
        ]
        predictions = np.array(predictions)

        # Calculate metrics
        metrics = {
            "log_loss": log_loss(outcomes, predictions),
            "brier_score": brier_score_loss(outcomes, predictions),
            "accuracy": np.mean((predictions > 0.5) == outcomes),
        }

        # Calibration bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (
                predictions < bin_edges[i + 1]
            )
            if i == n_bins - 1:  # Include 1.0 in last bin
                mask = (predictions >= bin_edges[i]) & (
                    predictions <= bin_edges[i + 1]
                )

            if mask.sum() > 0:
                bin_centers.append(predictions[mask].mean())
                bin_accuracies.append(outcomes[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_accuracies.append(0)
                bin_counts.append(0)

        metrics["bin_centers"] = np.array(bin_centers)
        metrics["bin_accuracies"] = np.array(bin_accuracies)
        metrics["bin_counts"] = np.array(bin_counts)

        # Expected calibration error
        total_count = sum(bin_counts)
        if total_count > 0:
            ece = (
                sum(
                    count * abs(center - accuracy)
                    for center, accuracy, count in zip(
                        bin_centers, bin_accuracies, bin_counts
                    )
                )
                / total_count
            )
            metrics["expected_calibration_error"] = ece
        else:
            metrics["expected_calibration_error"] = 0.0

        return metrics


class AdaptiveCalibrator:
    """
    Adaptive calibration that updates with new match results.
    """

    def __init__(
        self,
        initial_beta: float = 1.0,
        initial_bias: float = 0.0,
        learning_rate: float = 0.01,
        window_size: int = 1000,
    ):
        """
        Initialize adaptive calibrator.

        Parameters
        ----------
        initial_beta : float
            Starting beta value
        initial_bias : float
            Starting bias value
        learning_rate : float
            Learning rate for online updates
        window_size : int
            Size of sliding window for recent matches
        """
        self.beta = initial_beta
        self.bias = initial_bias
        self.learning_rate = learning_rate
        self.window_size = window_size

        # Keep recent history
        self.recent_diffs = []
        self.recent_outcomes = []

    def update(self, rating_diff: float, outcome: bool) -> None:
        """
        Update calibration with a new match result.

        Parameters
        ----------
        rating_diff : float
            Rating difference for the match
        outcome : bool
            Whether the higher-rated team won
        """
        # Add to history
        self.recent_diffs.append(rating_diff)
        self.recent_outcomes.append(outcome)

        # Maintain window size
        if len(self.recent_diffs) > self.window_size:
            self.recent_diffs.pop(0)
            self.recent_outcomes.pop(0)

        # Gradient update if we have enough data
        if len(self.recent_diffs) >= 100:
            # Current prediction
            pred = self.predict_probability(rating_diff)
            error = outcome - pred

            # Gradient for logistic regression
            grad_beta = error * rating_diff
            grad_bias = error

            # Update parameters
            self.beta += self.learning_rate * grad_beta
            self.bias += self.learning_rate * grad_bias

    def predict_probability(self, rating_diff: float) -> float:
        """
        Predict win probability.

        Parameters
        ----------
        rating_diff : float
            Rating difference

        Returns
        -------
        float
            Win probability for higher-rated team
        """
        x = self.beta * rating_diff + self.bias
        return 1.0 / (1.0 + np.exp(-x))

    def refit(self) -> None:
        """Refit parameters using recent history."""
        if len(self.recent_diffs) < 100:
            return

        X = np.array(self.recent_diffs).reshape(-1, 1)
        y = np.array(self.recent_outcomes)

        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(X, y)

        self.beta = float(model.coef_[0, 0])
        self.bias = float(model.intercept_[0])


def calibrate_tournament_predictions(
    historical_matches: pl.DataFrame,
    team_ratings: dict[int, float],
    test_matches: pl.DataFrame | None = None,
    use_format_specific: bool = False,
    match_format_col: str | None = None,
) -> tuple[CalibrationResult, dict | None]:
    """
    Complete calibration pipeline for tournament predictions.

    Parameters
    ----------
    historical_matches : pl.DataFrame
        Historical matches for training
    team_ratings : Dict[int, float]
        Pre-calculated team ratings
    test_matches : pl.DataFrame | None
        Test matches for evaluation
    use_format_specific : bool
        Whether to use format-specific calibration
    match_format_col : Optional[str]
        Column name for match format

    Returns
    -------
    tuple[CalibrationResult, dict | None]
        Calibration result and optional evaluation metrics
    """
    # Initialize calibrator
    calibrator = MatchCalibrator(use_format_specific=use_format_specific)

    # Fit calibration
    result = calibrator.calibrate_from_matches(
        historical_matches, team_ratings, match_format_col
    )

    logger.info(f"Calibration complete:")
    logger.info(f"  β = {result.beta:.4f}")
    logger.info(f"  bias = {result.bias:.4f}")
    logger.info(f"  CV log loss = {result.log_loss:.4f}")
    logger.info(f"  CV accuracy = {result.accuracy:.1%}")

    # Evaluate on test set if provided
    evaluation = None
    if test_matches is not None:
        test_diffs, test_outcomes, _ = calibrator.prepare_match_data(
            test_matches, team_ratings
        )

        if len(test_diffs) > 0:
            evaluation = calibrator.evaluate_calibration(
                test_diffs, test_outcomes
            )

            logger.info(f"Test set evaluation:")
            logger.info(f"  Log loss = {evaluation['log_loss']:.4f}")
            logger.info(f"  Accuracy = {evaluation['accuracy']:.1%}")
            logger.info(
                f"  ECE = {evaluation['expected_calibration_error']:.4f}"
            )

    return result, evaluation
