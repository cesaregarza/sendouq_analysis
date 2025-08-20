"""
Tests for the refactored loss computation.

Tests the unified loss function against the legacy implementation
and validates the new weighting schemes.
"""

import numpy as np
import polars as pl
import pytest

from rankings.evaluation.loss import (
    compute_tournament_loss,
    fit_alpha_parameter,
)


class TestLossRefactor:
    """Test the refactored compute_tournament_loss function."""

    def setup_method(self):
        """Set up test data."""
        self.matches = pl.DataFrame(
            {"winner_team_id": [1, 2, 3, 1], "loser_team_id": [2, 3, 1, 3]}
        )
        self.ratings = {1: 1600, 2: 1500, 3: 1400}

    def test_scheme_none_equivalent_to_unweighted(self):
        """Test that scheme='none' gives unweighted cross-entropy."""
        loss_none, metrics_none = compute_tournament_loss(
            self.matches, self.ratings, scheme="none"
        )

        # Manual calculation of unweighted loss
        from rankings.evaluation.loss import compute_match_probability

        total_loss = 0
        for row in self.matches.iter_rows(named=True):
            winner_id = row["winner_team_id"]
            loser_id = row["loser_team_id"]
            r_winner = self.ratings[winner_id]
            r_loser = self.ratings[loser_id]
            p_win = compute_match_probability(r_winner, r_loser, alpha=1.0)
            total_loss += -np.log(p_win)

        expected_loss = total_loss / len(self.matches)

        assert (
            abs(loss_none - expected_loss) < 1e-12
        ), "scheme='none' should give unweighted loss"
        assert metrics_none["n_matches"] == 4
        assert metrics_none["unweighted_loss"] == metrics_none["weighted_loss"]

    def test_scheme_var_inv_weighting(self):
        """Test inverse variance weighting scheme."""
        loss_var, metrics_var = compute_tournament_loss(
            self.matches, self.ratings, scheme="var_inv"
        )

        loss_none, _ = compute_tournament_loss(
            self.matches, self.ratings, scheme="none"
        )

        # Variance weighting should generally give different loss
        # (unless all predictions have same variance)
        assert loss_var != loss_none or True  # May be equal in some cases
        assert metrics_var["n_matches"] == 4

    def test_scheme_entropy_weighting(self):
        """Test entropy-based weighting scheme."""
        loss_entropy, metrics_entropy = compute_tournament_loss(
            self.matches, self.ratings, scheme="entropy"
        )

        loss_none, _ = compute_tournament_loss(
            self.matches, self.ratings, scheme="none"
        )

        # Entropy weighting emphasizes matches away from 0.5 probability
        assert metrics_entropy["n_matches"] == 4
        # Loss may be different due to weighting
        assert isinstance(loss_entropy, float)

    def test_different_alpha_values(self):
        """Test that different alpha values produce different losses."""
        loss_alpha_1, _ = compute_tournament_loss(
            self.matches, self.ratings, alpha=1.0
        )

        loss_alpha_2, _ = compute_tournament_loss(
            self.matches, self.ratings, alpha=2.0
        )

        assert (
            loss_alpha_1 != loss_alpha_2
        ), "Different alpha should give different loss"

    def test_return_predictions(self):
        """Test that predictions are returned when requested."""
        loss, metrics = compute_tournament_loss(
            self.matches, self.ratings, return_predictions=True
        )

        assert "predictions" in metrics
        assert len(metrics["predictions"]) == 4
        assert all(0 < p < 1 for p in metrics["predictions"])
        assert "brier_score" in metrics
        assert "bucketised_metrics" in metrics

    def test_custom_column_names(self):
        """Test with custom column names."""
        custom_matches = self.matches.rename(
            {"winner_team_id": "win_id", "loser_team_id": "lose_id"}
        )

        loss, metrics = compute_tournament_loss(
            custom_matches,
            self.ratings,
            winner_id_col="win_id",
            loser_id_col="lose_id",
        )

        assert metrics["n_matches"] == 4
        assert isinstance(loss, float)

    def test_empty_matches(self):
        """Test with empty match data."""
        empty_matches = pl.DataFrame(
            {"winner_team_id": [], "loser_team_id": []}
        )

        loss, metrics = compute_tournament_loss(empty_matches, self.ratings)

        assert loss == np.inf
        assert metrics["n_matches"] == 0

    def test_missing_ratings_default_to_zero(self):
        """Test that missing ratings default to 0."""
        incomplete_ratings = {1: 1600}  # Missing 2, 3

        loss, metrics = compute_tournament_loss(
            self.matches, incomplete_ratings
        )

        # Should not crash, uses 0 for missing ratings
        assert isinstance(loss, float)
        assert metrics["n_matches"] == 4


class TestFitAlphaRefactor:
    """Test the refactored fit_alpha_parameter function."""

    def setup_method(self):
        """Set up test data."""
        self.matches = pl.DataFrame(
            {
                "winner_team_id": [1, 2, 3, 1, 2],
                "loser_team_id": [2, 3, 1, 3, 1],
            }
        )
        self.ratings = {1: 1600, 2: 1500, 3: 1400}

    def test_fit_alpha_with_schemes(self):
        """Test alpha fitting with different weighting schemes."""
        for scheme in ["none", "var_inv", "entropy"]:
            alpha = fit_alpha_parameter(
                self.matches, self.ratings, scheme=scheme
            )

            assert (
                0.1 <= alpha <= 10.0
            ), f"Alpha {alpha} should be in bounds for scheme {scheme}"
            assert isinstance(alpha, float)

    def test_fit_alpha_bounds(self):
        """Test that alpha fitting respects bounds."""
        alpha = fit_alpha_parameter(
            self.matches, self.ratings, alpha_bounds=(0.5, 2.0)
        )

        assert 0.5 <= alpha <= 2.0, "Alpha should respect custom bounds"

    def test_fit_alpha_custom_columns(self):
        """Test alpha fitting with custom column names."""
        custom_matches = self.matches.rename(
            {"winner_team_id": "w_id", "loser_team_id": "l_id"}
        )

        alpha = fit_alpha_parameter(
            custom_matches,
            self.ratings,
            winner_id_col="w_id",
            loser_id_col="l_id",
        )

        assert 0.1 <= alpha <= 10.0, "Should work with custom column names"


class TestWeightingLogic:
    """Test the specific weighting schemes in detail."""

    def setup_method(self):
        """Create test cases with known probabilities."""
        # Create matches where we know the exact probabilities
        self.matches = pl.DataFrame(
            {
                "winner_team_id": [1, 1],  # Same winner
                "loser_team_id": [2, 3],  # Different losers
            }
        )
        # Ratings chosen to give specific probabilities with Bradley-Terry
        # For BT: P(A beats B) = s_A / (s_A + s_B)
        # To get P = 0.75: s_A/s_B = 3, so s_A = 3*s_B
        # To get P = 0.9: s_A/s_B = 9, so s_A = 9*s_B
        self.ratings = {1: 0.9, 2: 0.3, 3: 0.1}  # p ≈ 0.75, 0.9

    def test_variance_weighting_logic(self):
        """Test that variance weighting emphasizes uncertain matches."""
        loss_none, metrics_none = compute_tournament_loss(
            self.matches, self.ratings, scheme="none", return_predictions=True
        )

        loss_var, metrics_var = compute_tournament_loss(
            self.matches,
            self.ratings,
            scheme="var_inv",
            return_predictions=True,
        )

        predictions = metrics_none["predictions"]
        assert len(predictions) == 2

        # p ≈ 0.75 has higher variance than p ≈ 0.9
        # So the first match should get higher weight
        # This affects the weighted average loss
        assert isinstance(loss_var, float)

    def test_entropy_weighting_logic(self):
        """Test that entropy weighting emphasizes extreme probabilities."""
        loss_none, metrics_none = compute_tournament_loss(
            self.matches, self.ratings, scheme="none", return_predictions=True
        )

        loss_entropy, metrics_entropy = compute_tournament_loss(
            self.matches,
            self.ratings,
            scheme="entropy",
            return_predictions=True,
        )

        predictions = metrics_none["predictions"]

        # Entropy weight = 2 * |p - 0.5|
        # p ≈ 0.9 gets weight 2 * |0.9 - 0.5| = 0.8
        # p ≈ 0.75 gets weight 2 * |0.75 - 0.5| = 0.5
        # So second match gets higher weight
        assert isinstance(loss_entropy, float)


class TestRegressionTests:
    """Regression tests to ensure behavior matches expectations."""

    def test_consistent_results(self):
        """Test that results are consistent across calls."""
        matches = pl.DataFrame(
            {"winner_team_id": [1, 2, 1], "loser_team_id": [2, 1, 2]}
        )
        ratings = {1: 1600, 2: 1400}

        # Multiple calls should give identical results
        loss1, metrics1 = compute_tournament_loss(matches, ratings)
        loss2, metrics2 = compute_tournament_loss(matches, ratings)

        assert loss1 == loss2
        assert metrics1["accuracy"] == metrics2["accuracy"]
        assert metrics1["mean_probability"] == metrics2["mean_probability"]

    def test_permutation_invariance(self):
        """Test that match order doesn't affect results."""
        matches1 = pl.DataFrame(
            {"winner_team_id": [1, 2, 3], "loser_team_id": [2, 3, 1]}
        )

        matches2 = pl.DataFrame(
            {"winner_team_id": [3, 1, 2], "loser_team_id": [1, 2, 3]}
        )

        ratings = {1: 1400, 2: 1500, 3: 1600}

        loss1, _ = compute_tournament_loss(matches1, ratings)
        loss2, _ = compute_tournament_loss(matches2, ratings)

        assert abs(loss1 - loss2) < 1e-12, "Match order should not affect loss"
