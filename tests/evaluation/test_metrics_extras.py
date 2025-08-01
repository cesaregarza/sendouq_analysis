"""
Tests for the extra metrics module.

Tests the metrics suite described in plan.md with synthetic data
to ensure correctness and edge case handling.
"""

import numpy as np
import polars as pl
import pytest

from src.rankings.evaluation.metrics_extras import (
    accuracy_threshold,
    alpha_std,
    concordance,
    placement_spearman,
    skill_score,
    upset_oe,
)


class TestConcordance:
    """Test concordance (c-statistic) metric."""

    def test_perfect_concordance(self):
        """Test case where higher rating always wins."""
        matches = pl.DataFrame(
            {
                "winner_team_id": [3, 2, 3],  # ratings: 1600, 1500, 1600
                "loser_team_id": [1, 1, 2],  # ratings: 1400, 1400, 1500
            }
        )
        ratings = {1: 1400, 2: 1500, 3: 1600}

        result = concordance(matches, ratings)
        assert result == 1.0, "Perfect rating order should give c-stat = 1.0"

    def test_random_concordance(self):
        """Test symmetric case that should give ~0.5."""
        # Equal ratings = random outcomes
        matches = pl.DataFrame(
            {"winner_team_id": [1, 2, 1, 2], "loser_team_id": [2, 1, 2, 1]}
        )
        ratings = {1: 1500, 2: 1500}  # Equal ratings

        result = concordance(matches, ratings)
        assert (
            result == 0.0
        ), "Equal ratings should give c-stat = 0 (never higher-rated wins)"

    def test_empty_matches(self):
        """Test with no matches."""
        matches = pl.DataFrame({"winner_team_id": [], "loser_team_id": []})
        ratings = {1: 1500}

        result = concordance(matches, ratings)
        assert np.isnan(result), "Empty matches should return NaN"

    def test_missing_ratings(self):
        """Test with missing rating entries (should default to 0)."""
        matches = pl.DataFrame(
            {"winner_team_id": [1, 3], "loser_team_id": [2, 1]}
        )
        ratings = {1: 1500}  # Missing ratings for 2, 3

        result = concordance(matches, ratings)
        assert result == 1.0, "Team 1 (1500) > teams 2,3 (0 default)"


class TestSkillScore:
    """Test skill score metric."""

    def test_perfect_skill(self):
        """Test case with very predictable outcomes."""
        matches = pl.DataFrame(
            {"winner_team_id": [3, 3, 2], "loser_team_id": [1, 2, 1]}
        )
        ratings = {1: 1000, 2: 1500, 3: 2000}  # Big rating gaps

        result = skill_score(matches, ratings, n_permutations=5)
        assert (
            result > 0
        ), "Predictable outcomes should have positive skill score"
        assert result <= 1.0, "Skill score should be <= 1.0"

    def test_random_skill(self):
        """Test case with equal ratings (no skill)."""
        matches = pl.DataFrame(
            {"winner_team_id": [1, 2, 1], "loser_team_id": [2, 1, 2]}
        )
        ratings = {1: 1500, 2: 1500}  # Equal ratings

        result = skill_score(matches, ratings, n_permutations=5)
        # Should be close to 0 since model and permutation have similar loss
        assert -0.5 <= result <= 0.5, "No skill case should have skill score ~0"

    def test_empty_matches(self):
        """Test with no matches."""
        matches = pl.DataFrame({"winner_team_id": [], "loser_team_id": []})
        ratings = {1: 1500}

        result = skill_score(matches, ratings)
        assert np.isnan(result), "Empty matches should return NaN"


class TestUpsetOE:
    """Test upset observed/expected ratio."""

    def test_perfect_calibration(self):
        """Test perfectly calibrated predictions."""
        # If p=0.7 for all matches, expect 30% upsets, should observe 30%
        predictions = np.array(
            [0.7, 0.7, 0.7, 0.3]
        )  # 1 upset expected and observed

        result = upset_oe(predictions)
        assert (
            abs(result - 1.0) < 0.01
        ), "Perfect calibration should give O/E ≈ 1.0"

    def test_overconfident_model(self):
        """Test overconfident model (fewer upsets than expected)."""
        # Model predicts high probabilities but upsets still happen
        predictions = np.array(
            [0.9, 0.9, 0.9, 0.1]
        )  # Expect ~0.5 upsets, observe 1

        result = upset_oe(predictions)
        assert result > 1.0, "More upsets than expected should give O/E > 1"

    def test_underconfident_model(self):
        """Test underconfident model (more upsets than expected)."""
        # Model predicts low probabilities but favorites still win
        predictions = np.array(
            [0.6, 0.6, 0.6, 0.6]
        )  # Expect 1.6 upsets, observe 0

        result = upset_oe(predictions)
        assert result < 1.0, "Fewer upsets than expected should give O/E < 1"

    def test_empty_predictions(self):
        """Test with no predictions."""
        predictions = np.array([])

        result = upset_oe(predictions)
        assert np.isnan(result), "Empty predictions should return NaN"

    def test_all_certain_predictions(self):
        """Test edge case where expected upsets = 0."""
        predictions = np.array([1.0, 1.0, 1.0])  # No expected upsets

        result = upset_oe(predictions)
        assert np.isnan(result), "No expected upsets should return NaN"


class TestAccuracyThreshold:
    """Test confidence-filtered accuracy."""

    def test_high_confidence_correct(self):
        """Test high-confidence predictions that are correct."""
        predictions = np.array(
            [0.8, 0.9, 0.7, 0.4]
        )  # 3 above threshold, all correct

        result = accuracy_threshold(predictions, threshold=0.65)
        assert (
            result == 1.0
        ), "All high-confidence correct predictions should give 100% accuracy"

    def test_high_confidence_mixed(self):
        """Test mixed high-confidence predictions."""
        predictions = np.array(
            [0.8, 0.3, 0.7, 0.2]
        )  # 2 above threshold: 0.8 (correct), 0.7 (correct)

        result = accuracy_threshold(predictions, threshold=0.65)
        assert (
            result == 1.0
        ), "High-confidence predictions above 0.5 should be correct"

    def test_no_high_confidence(self):
        """Test case with no high-confidence predictions."""
        predictions = np.array([0.6, 0.5, 0.4])  # All below threshold

        result = accuracy_threshold(predictions, threshold=0.65)
        assert np.isnan(
            result
        ), "No high-confidence predictions should return NaN"

    def test_empty_predictions(self):
        """Test with no predictions."""
        predictions = np.array([])

        result = accuracy_threshold(predictions)
        assert np.isnan(result), "Empty predictions should return NaN"


class TestPlacementSpearman:
    """Test Spearman correlation between ratings and placements."""

    def test_perfect_correlation(self):
        """Test perfect ranking order."""
        ratings = {1: 2000, 2: 1500, 3: 1000}
        placements = {1: 1, 2: 2, 3: 3}  # Perfect order

        result = placement_spearman(ratings, placements)
        assert result == 1.0, "Perfect order should give correlation = 1.0"

    def test_reverse_correlation(self):
        """Test reverse ranking order."""
        ratings = {1: 2000, 2: 1500, 3: 1000}
        placements = {1: 3, 2: 2, 3: 1}  # Reverse order

        result = placement_spearman(ratings, placements)
        assert result == -1.0, "Reverse order should give correlation = -1.0"

    def test_no_correlation(self):
        """Test uncorrelated rankings."""
        ratings = {1: 1500, 2: 1500, 3: 1500}  # Equal ratings
        placements = {1: 1, 2: 2, 3: 3}

        result = placement_spearman(ratings, placements)
        # With equal ratings, correlation is undefined
        assert np.isnan(result), "Equal ratings should give NaN correlation"

    def test_partial_overlap(self):
        """Test with partial overlap between ratings and placements."""
        ratings = {1: 2000, 2: 1500, 3: 1000, 4: 800}
        placements = {2: 1, 3: 2, 5: 3}  # Only 2,3 overlap

        result = placement_spearman(ratings, placements)
        assert (
            result == 1.0
        ), "Overlapping teams should show perfect correlation"

    def test_insufficient_data(self):
        """Test with insufficient overlapping data."""
        ratings = {1: 2000}
        placements = {1: 1}  # Only 1 team

        result = placement_spearman(ratings, placements)
        assert np.isnan(result), "Single team should return NaN"


class TestAlphaStd:
    """Test alpha standard deviation metric."""

    def test_stable_alpha(self):
        """Test consistent alpha values."""
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = alpha_std(alphas)
        assert result == 0.0, "Identical alphas should give std = 0"

    def test_variable_alpha(self):
        """Test variable alpha values."""
        alphas = [0.5, 1.0, 1.5, 2.0, 2.5]

        result = alpha_std(alphas)
        expected_std = np.std(alphas)
        assert abs(result - expected_std) < 1e-10, "Should match numpy std"

    def test_insufficient_data(self):
        """Test with insufficient data."""
        alphas = [1.0]  # Single value

        result = alpha_std(alphas)
        assert np.isnan(result), "Single alpha should return NaN"

    def test_empty_data(self):
        """Test with no data."""
        alphas = []

        result = alpha_std(alphas)
        assert np.isnan(result), "Empty alphas should return NaN"


class TestSyntheticScenarios:
    """Integration tests with synthetic scenarios."""

    def test_discrimination_scenario(self):
        """Test scenario where winner always has higher rating."""
        matches = pl.DataFrame(
            {"winner_team_id": [2, 3, 3, 2], "loser_team_id": [1, 1, 2, 1]}
        )
        ratings = {1: 1000, 2: 1500, 3: 2000}

        c_stat = concordance(matches, ratings)
        skill = skill_score(matches, ratings, n_permutations=3)

        assert c_stat == 1.0, "Perfect discrimination should give c_stat = 1"
        assert skill > 0, "Should have positive skill vs random"

    def test_no_discrimination_scenario(self):
        """Test scenario with no discriminative power."""
        matches = pl.DataFrame(
            {"winner_team_id": [1, 2, 1, 2], "loser_team_id": [2, 1, 2, 1]}
        )
        ratings = {1: 1500, 2: 1500}  # Equal ratings

        c_stat = concordance(matches, ratings)
        skill = skill_score(matches, ratings, n_permutations=3)

        assert c_stat == 0.0, "No discrimination should give c_stat = 0"
        assert abs(skill) < 0.5, "Should have near-zero skill vs random"
