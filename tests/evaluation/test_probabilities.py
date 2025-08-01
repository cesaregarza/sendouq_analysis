"""
Unit tests for Bradley-Terry probability model implementation.

This module tests the bt_prob function and the score_transform functionality
according to the test matrix in section 6 of plan.md.
"""

import math

import numpy as np
import polars as pl
import pytest

from rankings.analysis.transforms import bt_prob
from rankings.evaluation.loss import (
    compute_match_probability,
    compute_tournament_loss,
)


class TestBTProb:
    """Test cases for the bt_prob function."""

    def test_bt_prob_symmetry(self):
        """Test that bt_prob(sa, sb) + bt_prob(sb, sa) == 1."""
        sa, sb = 2.0, 1.0
        p_a_beats_b = bt_prob(sa, sb)
        p_b_beats_a = bt_prob(sb, sa)

        assert abs(p_a_beats_b + p_b_beats_a - 1.0) < 1e-10

    def test_bt_prob_extreme_ratio(self):
        """Test bt_prob with extreme score ratios."""
        sa, sb = 1e-3, 1.0
        p = bt_prob(sa, sb)

        # With very low sa, probability should be close to sa/(sa+sb) ≈ 0.001
        expected = sa / (sa + sb)
        assert abs(p - expected) < 1e-6
        assert p < 0.01  # Should be very small

    def test_bt_prob_alpha_scaling(self):
        """Test that alpha scaling increases discrimination."""
        sa, sb = 0.01, 0.002
        p_alpha_1 = bt_prob(sa, sb, alpha=1.0)
        p_alpha_2 = bt_prob(sa, sb, alpha=2.0)

        # Higher alpha should increase probability for the stronger competitor
        assert p_alpha_2 > p_alpha_1

    def test_bt_prob_classical_formula(self):
        """Test that alpha=1 gives classical Bradley-Terry formula."""
        sa, sb = 0.6, 0.4
        p = bt_prob(sa, sb, alpha=1.0)
        expected = sa / (sa + sb)

        assert abs(p - expected) < 1e-10

    def test_bt_prob_edge_cases(self):
        """Test bt_prob with edge cases like zero scores."""
        # Very small scores
        p = bt_prob(1e-15, 1e-15)
        assert abs(p - 0.5) < 1e-10

        # One zero score (gets clamped to eps)
        p = bt_prob(0.0, 1.0)
        assert p < 1e-10  # Should be very small

    def test_bt_prob_alpha_equivalence(self):
        """Test that BT with alpha != 1 equals logistic on log-ratio."""
        sa, sb = 0.1, 0.05
        alpha = 2.5

        bt_result = bt_prob(sa, sb, alpha=alpha)

        # Manual logistic calculation
        z = alpha * (math.log(sa) - math.log(sb))
        logistic_result = 1.0 / (1.0 + math.exp(-z))

        assert abs(bt_result - logistic_result) < 1e-10


class TestScoreTransforms:
    """Test cases for score_transform functionality."""

    def test_compute_match_probability_bradley_terry(self):
        """Test that score_transform='bradley_terry' calls bt_prob."""
        rating_a, rating_b = 0.6, 0.4
        alpha = 1.5

        p_bt = compute_match_probability(
            rating_a, rating_b, alpha=alpha, score_transform="bradley_terry"
        )
        p_direct = bt_prob(rating_a, rating_b, alpha=alpha)

        assert abs(p_bt - p_direct) < 1e-10

    def test_compute_match_probability_logistic(self):
        """Test that score_transform='logistic' gives legacy behavior."""
        rating_a, rating_b = 1.2, 0.8
        alpha = 2.0

        p_logistic = compute_match_probability(
            rating_a, rating_b, alpha=alpha, score_transform="logistic"
        )

        # Manual logistic calculation
        z = alpha * (rating_a - rating_b)
        expected = 1.0 / (1.0 + np.exp(-z))

        assert abs(p_logistic - expected) < 1e-10

    def test_compute_match_probability_identity(self):
        """Test that score_transform='identity' returns rating_a directly."""
        rating_a, rating_b = 0.75, 0.25

        p_identity = compute_match_probability(
            rating_a, rating_b, alpha=1.0, score_transform="identity"
        )

        assert abs(p_identity - rating_a) < 1e-10

    def test_compute_match_probability_invalid_transform(self):
        """Test that invalid score_transform raises ValueError."""
        with pytest.raises(ValueError, match="Unknown score_transform"):
            compute_match_probability(
                0.6, 0.4, alpha=1.0, score_transform="invalid"
            )

    def test_score_transform_none_handling(self):
        """Test that None ratings are handled properly."""
        p = compute_match_probability(
            None, 0.5, alpha=1.0, score_transform="bradley_terry"
        )

        # None should become 0.0, so bt_prob(0.0, 0.5)
        expected = bt_prob(0.0, 0.5)
        assert abs(p - expected) < 1e-10


class TestTournamentLoss:
    """Test tournament loss computation with Bradley-Terry model."""

    def test_tournament_loss_sanity(self):
        """Test that BT model gives reasonable loss on synthetic data."""
        # Create synthetic matches where high-rated teams usually win
        matches_df = pl.DataFrame(
            {
                "winner_team_id": [1, 1, 2, 2],
                "loser_team_id": [3, 4, 3, 4],
            }
        )

        # High ratings for winners, low for losers
        rating_map = {1: 0.8, 2: 0.7, 3: 0.2, 4: 0.1}

        loss, metrics = compute_tournament_loss(
            matches_df,
            rating_map,
            alpha=1.0,
            score_transform="bradley_terry",
        )

        # Loss should be better than random (0.693)
        assert loss < 0.693
        assert metrics["accuracy"] > 0.5

    def test_tournament_loss_vs_logistic(self):
        """Test that BT and logistic give different results."""
        matches_df = pl.DataFrame(
            {
                "winner_team_id": [1, 2],
                "loser_team_id": [2, 3],
            }
        )

        rating_map = {1: 0.1, 2: 0.05, 3: 0.01}  # Small positive values

        loss_bt, _ = compute_tournament_loss(
            matches_df, rating_map, alpha=1.0, score_transform="bradley_terry"
        )

        loss_logistic, _ = compute_tournament_loss(
            matches_df, rating_map, alpha=1.0, score_transform="logistic"
        )

        # They should give different results
        assert abs(loss_bt - loss_logistic) > 1e-6


class TestCVIntegration:
    """Integration tests for cross-validation with score_transform."""

    def test_cv_smoke_test(self):
        """Basic smoke test for CV with Bradley-Terry."""
        # Create minimal viable tournament data
        matches_df = pl.DataFrame(
            {
                "tournament_id": [1, 1, 2, 2, 3, 3],
                "winner_team_id": [1, 2, 1, 3, 2, 3],
                "loser_team_id": [2, 3, 3, 2, 1, 1],
                "last_game_finished_at": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-02-01",
                    "2023-02-02",
                    "2023-03-01",
                    "2023-03-02",
                ],
            }
        ).with_columns(
            pl.col("last_game_finished_at").str.strptime(
                pl.Datetime, "%Y-%m-%d"
            )
        )

        rating_map = {1: 0.6, 2: 0.5, 3: 0.4}

        # Just test that tournament loss computation works
        loss, metrics = compute_tournament_loss(
            matches_df.filter(pl.col("tournament_id") == 1),
            rating_map,
            alpha=1.0,
            score_transform="bradley_terry",
        )

        assert np.isfinite(loss)
        assert metrics["n_matches"] == 2
        assert "accuracy" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
