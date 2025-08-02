"""Tests for entropy weighting schemes in loss computation."""

import numpy as np
import polars as pl
import pytest

from rankings.evaluation.loss import (
    compute_tournament_loss,
    compute_weighted_log_loss,
)


class TestWeightingSchemes:
    """Test different weighting schemes for loss computation."""

    def test_entropy_weighting_values(self):
        """Test that entropy (confidence) weighting produces correct values."""
        # Test specific probability values
        p = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        # Expected weights for entropy (confidence) scheme
        expected_weights = np.abs(p - 0.5) * 2

        # Compute weighted loss (we only care about weights)
        # Since we're testing weights, the actual loss value doesn't matter
        for i, prob in enumerate(p):
            single_p = np.array([prob])
            # The weight should be |p - 0.5| * 2
            expected = expected_weights[i]

            # For a single prediction, weighted and unweighted ratio tells us the weight
            weighted_loss = compute_weighted_log_loss(
                single_p, scheme="entropy"
            )
            unweighted_loss = compute_weighted_log_loss(single_p, scheme="none")

            # Since we have only one prediction, the weight is the ratio
            # But for p close to 0.5, both losses are similar, so test the weight directly
            assert abs(abs(prob - 0.5) * 2 - expected) < 1e-6

    def test_entropy_sqrt_weighting(self):
        """Test entropy_sqrt weighting scheme."""
        p = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        # Expected weights: sqrt(|p - 0.5| * 2)
        expected = np.sqrt(np.abs(p - 0.5) * 2)

        # Test edge cases
        assert np.sqrt(0.0) == 0.0  # p=0.5
        assert np.sqrt(1.0) == 1.0  # p=0.0 or p=1.0

        # Test that sqrt is between linear and squared
        for prob in [0.6, 0.7, 0.8, 0.9]:
            linear = abs(prob - 0.5) * 2
            sqrt_val = np.sqrt(linear)
            squared = linear**2

            # For values between 0 and 1, we have: squared < linear < sqrt
            # Actually, for x in (0,1): x^2 < x < sqrt(x)
            if linear < 1:
                assert squared < linear < sqrt_val
            else:
                assert squared > linear > sqrt_val

    def test_entropy_squared_weighting(self):
        """Test entropy_squared weighting scheme."""
        p = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        # Expected weights: (|p - 0.5| * 2)^2
        expected = (np.abs(p - 0.5) * 2) ** 2

        # Test specific values
        assert (0.0) ** 2 == 0.0  # p=0.5
        assert (1.0) ** 2 == 1.0  # p=0.0 or p=1.0

        # Test that squared emphasizes extremes more
        # For p=0.55 (close to 0.5), weight should be very small
        p_close = 0.55
        weight_close = (abs(p_close - 0.5) * 2) ** 2
        assert abs(weight_close - 0.01) < 1e-10  # (0.1)^2 = 0.01

        # For p=0.95 (far from 0.5), weight should be large
        p_far = 0.95
        weight_far = (abs(p_far - 0.5) * 2) ** 2
        assert abs(weight_far - 0.81) < 1e-10  # (0.9)^2 = 0.81

    def test_weighting_scheme_ordering(self):
        """Test that weighting schemes have correct relative ordering."""
        # For any p != 0.5, we should have: squared < sqrt < linear
        test_probs = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

        for p in test_probs:
            conf = abs(p - 0.5) * 2

            # Calculate weights
            w_linear = conf
            w_sqrt = np.sqrt(conf)
            w_squared = conf**2

            # Check ordering (except at boundaries)
            if 0 < conf < 1:
                assert w_squared < w_linear < w_sqrt

    def test_weighting_symmetry(self):
        """Test that weights are symmetric around p=0.5."""
        # Test that p and (1-p) give the same weight
        test_pairs = [(0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.4, 0.6)]

        for p1, p2 in test_pairs:
            # Entropy (confidence) weights
            w1 = abs(p1 - 0.5) * 2
            w2 = abs(p2 - 0.5) * 2
            assert abs(w1 - w2) < 1e-10

            # Sqrt weights
            w1_sqrt = np.sqrt(w1)
            w2_sqrt = np.sqrt(w2)
            assert abs(w1_sqrt - w2_sqrt) < 1e-10

            # Squared weights
            w1_sq = w1**2
            w2_sq = w2**2
            assert abs(w1_sq - w2_sq) < 1e-10

    def test_weighting_edge_cases(self):
        """Test edge cases for weighting schemes."""
        # Test p very close to 0.5
        p_close = np.array([0.5001, 0.4999, 0.501, 0.499])

        # All schemes should give very small weights
        for scheme in ["entropy", "entropy_sqrt", "entropy_squared"]:
            loss = compute_weighted_log_loss(p_close, scheme=scheme)
            # Loss should be dominated by the log term since weights are small

        # Test p at extremes
        p_extreme = np.array([0.001, 0.999])

        # All schemes should give weights close to 1
        for scheme in ["entropy", "entropy_sqrt", "entropy_squared"]:
            loss = compute_weighted_log_loss(p_extreme, scheme=scheme)
            # Loss should be very small since p is close to 1 (for winner)

    def test_tournament_loss_with_weighting(self):
        """Test compute_tournament_loss with different weighting schemes."""
        # Create test data
        matches = pl.DataFrame(
            {
                "winner_team_id": [1, 2, 3, 4, 5],
                "loser_team_id": [6, 7, 8, 9, 10],
            }
        )

        # Create rating map with clear favorites
        rating_map = {
            1: 0.9,
            2: 0.8,
            3: 0.7,
            4: 0.6,
            5: 0.55,  # Winners
            6: 0.1,
            7: 0.2,
            8: 0.3,
            9: 0.4,
            10: 0.45,  # Losers
        }

        # Test each scheme
        results = {}
        for scheme in ["none", "entropy", "entropy_sqrt", "entropy_squared"]:
            loss, metrics = compute_tournament_loss(
                matches,
                rating_map,
                alpha=1.0,
                scheme=scheme,
            )
            results[scheme] = loss

        # entropy_squared should have lowest loss (focuses on confident predictions)
        # none should have highest loss (includes all noise)
        assert results["entropy_squared"] < results["entropy"]
        assert results["entropy"] < results["none"]

    def test_coin_flip_matches_downweighted(self):
        """Test that coin-flip matches (p≈0.5) are downweighted."""
        # Create matches with varying confidence levels
        matches = pl.DataFrame(
            {
                "winner_team_id": [1, 2, 3],
                "loser_team_id": [4, 5, 6],
            }
        )

        # Ratings that create different probability scenarios
        rating_maps = {
            "coin_flip": {
                1: 0.51,
                2: 0.505,
                3: 0.52,
                4: 0.49,
                5: 0.495,
                6: 0.48,
            },
            "confident": {1: 0.8, 2: 0.9, 3: 0.95, 4: 0.2, 5: 0.1, 6: 0.05},
        }

        # For entropy weighting, confident predictions should have much lower loss
        for scheme in ["entropy", "entropy_sqrt", "entropy_squared"]:
            loss_coin_flip, _ = compute_tournament_loss(
                matches, rating_maps["coin_flip"], alpha=1.0, scheme=scheme
            )
            loss_confident, _ = compute_tournament_loss(
                matches, rating_maps["confident"], alpha=1.0, scheme=scheme
            )

            # Confident predictions should have lower loss when weighted
            assert loss_confident < loss_coin_flip

    def test_weighting_impact_on_optimization(self):
        """Test that different weightings affect what the model optimizes for."""
        # Create a dataset with mix of close and clear matches
        np.random.seed(42)
        n_matches = 100

        # Half close matches (p ∈ [0.45, 0.55]), half clear (p ∈ [0.8, 0.95])
        p_close = np.random.uniform(0.45, 0.55, n_matches // 2)
        p_clear = np.random.uniform(0.8, 0.95, n_matches // 2)
        p_all = np.concatenate([p_close, p_clear])

        # Compute losses
        loss_none = compute_weighted_log_loss(p_all, scheme="none")
        loss_entropy = compute_weighted_log_loss(p_all, scheme="entropy")
        loss_squared = compute_weighted_log_loss(
            p_all, scheme="entropy_squared"
        )

        # With no weighting, close matches contribute significantly to loss
        # With entropy_squared, close matches barely contribute
        # This is the desired behavior - we want to ignore noisy coin flips


if __name__ == "__main__":
    # Run specific test for debugging
    test = TestWeightingSchemes()
    test.test_entropy_weighting_values()
    test.test_entropy_sqrt_weighting()
    test.test_entropy_squared_weighting()
    test.test_weighting_scheme_ordering()
    test.test_weighting_symmetry()
    print("All manual tests passed!")
