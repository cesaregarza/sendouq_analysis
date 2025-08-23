"""Match outcome prediction using calibrated logistic regression."""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression


class MatchPredictor:
    """Predict match outcomes using calibrated logistic regression."""

    def __init__(self, beta: float = 1.0, bias: float = 0.0):
        """Initialize predictor with logistic parameters.

        Args:
            beta: Scaling parameter for rating difference
            bias: Bias term for side/format advantages
        """
        self.beta = beta
        self.bias = bias
        self.is_calibrated = False
        self.calibration_history = []

    def win_probability(
        self,
        team_a_rating: float,
        team_b_rating: float,
        format_multiplier: float = 1.0,
    ) -> float:
        """Calculate probability of team A beating team B.

        Implements: P(A > B) = sigmoid(beta * (R_A - R_B) + bias)

        Args:
            team_a_rating: Team A's log-rating
            team_b_rating: Team B's log-rating
            format_multiplier: Optional multiplier for different formats (Bo1/Bo3/Bo5)

        Returns:
            Probability that team A wins (0 to 1)
        """
        delta = team_a_rating - team_b_rating
        x = self.beta * format_multiplier * delta + self.bias
        return 1.0 / (1.0 + math.exp(-x))

    def calibrate(
        self,
        historical_matches: list[tuple[float, float, bool]],
        format_info: list[str] | None = None,
    ) -> dict[str, float]:
        """Calibrate logistic parameters on historical data.

        Args:
            historical_matches: List of (team_a_rating, team_b_rating, a_won)
            format_info: Optional list of format strings for each match

        Returns:
            Dictionary with calibration results and metrics
        """
        if not historical_matches:
            raise ValueError("Need historical matches for calibration")

        # Prepare data for logistic regression
        X = []
        y = []

        for match in historical_matches:
            r_a, r_b, a_won = match
            delta = r_a - r_b
            X.append([delta])
            y.append(1 if a_won else 0)

        X = np.array(X)
        y = np.array(y)

        # Fit logistic regression
        model = LogisticRegression(penalty=None, solver="lbfgs")
        model.fit(X, y)

        # Extract parameters
        self.beta = float(model.coef_[0, 0])
        self.bias = float(model.intercept_[0])
        self.is_calibrated = True

        # Calculate calibration metrics
        predictions = model.predict_proba(X)[:, 1]
        log_loss = -np.mean(
            y * np.log(predictions + 1e-10)
            + (1 - y) * np.log(1 - predictions + 1e-10)
        )
        accuracy = np.mean((predictions > 0.5) == y)

        result = {
            "beta": self.beta,
            "bias": self.bias,
            "log_loss": log_loss,
            "accuracy": accuracy,
            "n_matches": len(historical_matches),
        }

        # Store calibration history
        self.calibration_history.append(result)

        return result

    def calibration_curve(
        self,
        historical_matches: list[tuple[float, float, bool]],
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate calibration curve for model evaluation.

        Args:
            historical_matches: List of (team_a_rating, team_b_rating, a_won)
            n_bins: Number of probability bins

        Returns:
            Tuple of (mean_predicted_prob, fraction_positive) for each bin
        """
        predictions = []
        outcomes = []

        for r_a, r_b, a_won in historical_matches:
            prob = self.win_probability(r_a, r_b)
            predictions.append(prob)
            outcomes.append(1 if a_won else 0)

        predictions = np.array(predictions)
        outcomes = np.array(outcomes)

        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        mean_predicted = []
        fraction_positive = []

        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (
                predictions < bin_edges[i + 1]
            )
            if mask.sum() > 0:
                mean_predicted.append(predictions[mask].mean())
                fraction_positive.append(outcomes[mask].mean())

        return np.array(mean_predicted), np.array(fraction_positive)
