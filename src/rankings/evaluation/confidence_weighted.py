"""
Confidence-weighted evaluation metrics for ranking systems.

This module provides evaluation metrics that weight players by their confidence
scores, giving more importance to players with sufficient data for accurate ranking.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from rankings.analysis.confidence import ConfidenceTier, RankingConfidence


class ConfidenceWeightedEvaluator:
    """
    Evaluate ranking quality with confidence-based weighting.

    Key principle: Players with low connectivity cannot be accurately ranked,
    so they should contribute less to overall evaluation metrics.
    """

    def __init__(
        self,
        confidence_calculator: RankingConfidence | None = None,
        weight_power: float = 2.0,
    ):
        """
        Initialize evaluator.

        Parameters
        ----------
        confidence_calculator : RankingConfidence | None
            Confidence calculator instance
        weight_power : float
            Power to raise confidence scores to for weighting (higher = more aggressive weighting)
        """
        self.confidence_calculator = confidence_calculator
        self.weight_power = weight_power

    def weighted_spearman_correlation(
        self,
        true_ranks: dict[str, int],
        predicted_ranks: dict[str, int],
        confidence_scores: dict[str, float],
    ) -> float:
        """
        Calculate weighted Spearman correlation.

        Players with higher confidence scores contribute more to the correlation.

        Parameters
        ----------
        true_ranks : dict[str, int]
            True rankings (player_id -> rank)
        predicted_ranks : dict[str, int]
            Predicted rankings (player_id -> rank)
        confidence_scores : dict[str, float]
            Confidence scores (player_id -> score 0-1)

        Returns
        -------
        float
            Weighted Spearman correlation coefficient
        """
        # Get common players
        common_players = (
            set(true_ranks.keys())
            & set(predicted_ranks.keys())
            & set(confidence_scores.keys())
        )

        if len(common_players) < 2:
            return 0.0

        # Build arrays
        true_r = []
        pred_r = []
        weights = []

        for player in common_players:
            true_r.append(true_ranks[player])
            pred_r.append(predicted_ranks[player])
            # Weight by confidence^power
            weights.append(confidence_scores[player] ** self.weight_power)

        true_r = np.array(true_r)
        pred_r = np.array(pred_r)
        weights = np.array(weights)

        # Normalize weights
        weights = weights / weights.sum()

        # Calculate weighted rank differences
        rank_diffs = pred_r - true_r
        weighted_sq_diffs = weights * (rank_diffs**2)

        # Weighted Spearman correlation
        # Using effective sample size for normalization
        effective_n = 1.0 / np.sum(weights**2)

        # Standard formula with weighted differences
        weighted_rho = 1.0 - (6.0 * np.sum(weighted_sq_diffs)) / (
            effective_n * (effective_n**2 - 1)
        )

        return np.clip(weighted_rho, -1.0, 1.0)

    def tier_specific_correlation(
        self,
        true_ranks: dict[str, int],
        predicted_ranks: dict[str, int],
        player_tiers: dict[str, ConfidenceTier],
    ) -> dict[str, float]:
        """
        Calculate correlation separately for each confidence tier.

        This shows how well the ranking system performs for different
        levels of player connectivity.

        Parameters
        ----------
        true_ranks : dict[str, int]
            True rankings
        predicted_ranks : dict[str, int]
            Predicted rankings
        player_tiers : dict[str, ConfidenceTier]
            Player confidence tiers

        Returns
        -------
        dict[str, float]
            Correlation for each tier
        """
        results = {}

        for tier in ConfidenceTier:
            # Get players in this tier
            tier_players = [p for p, t in player_tiers.items() if t == tier]

            if len(tier_players) < 2:
                results[tier.value] = None
                continue

            # Get ranks for tier players
            tier_true = [
                true_ranks[p]
                for p in tier_players
                if p in true_ranks and p in predicted_ranks
            ]
            tier_pred = [
                predicted_ranks[p]
                for p in tier_players
                if p in true_ranks and p in predicted_ranks
            ]

            if len(tier_true) < 2:
                results[tier.value] = None
                continue

            # Calculate Spearman correlation for this tier
            correlation, _ = stats.spearmanr(tier_true, tier_pred)
            results[tier.value] = correlation

        return results

    def activity_weighted_correlation(
        self,
        true_ranks: dict[str, int],
        predicted_ranks: dict[str, int],
        player_activity: dict[str, int],
    ) -> float:
        """
        Weight correlation by player activity (e.g., tournament count).

        Simple alternative to confidence scoring - weight by raw activity.

        Parameters
        ----------
        true_ranks : dict[str, int]
            True rankings
        predicted_ranks : dict[str, int]
            Predicted rankings
        player_activity : dict[str, int]
            Activity measure (e.g., tournament count)

        Returns
        -------
        float
            Activity-weighted correlation
        """
        common_players = (
            set(true_ranks.keys())
            & set(predicted_ranks.keys())
            & set(player_activity.keys())
        )

        if len(common_players) < 2:
            return 0.0

        # Build arrays
        true_r = []
        pred_r = []
        weights = []

        for player in common_players:
            true_r.append(true_ranks[player])
            pred_r.append(predicted_ranks[player])
            # Weight by sqrt of activity (less aggressive than linear)
            weights.append(np.sqrt(player_activity[player]))

        # Calculate weighted correlation
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted covariance
        true_r = np.array(true_r)
        pred_r = np.array(pred_r)

        weighted_mean_true = np.sum(weights * true_r)
        weighted_mean_pred = np.sum(weights * pred_r)

        weighted_cov = np.sum(
            weights
            * (true_r - weighted_mean_true)
            * (pred_r - weighted_mean_pred)
        )
        weighted_var_true = np.sum(weights * (true_r - weighted_mean_true) ** 2)
        weighted_var_pred = np.sum(weights * (pred_r - weighted_mean_pred) ** 2)

        if weighted_var_true == 0 or weighted_var_pred == 0:
            return 0.0

        return weighted_cov / np.sqrt(weighted_var_true * weighted_var_pred)

    def connected_component_correlation(
        self,
        true_ranks: dict[str, int],
        predicted_ranks: dict[str, int],
        player_components: dict[str, int],
    ) -> dict[str, float]:
        """
        Calculate correlation within connected components.

        Players in different components cannot be meaningfully compared,
        so evaluate each component separately.

        Parameters
        ----------
        true_ranks : dict[str, int]
            True rankings
        predicted_ranks : dict[str, int]
            Predicted rankings
        player_components : dict[str, int]
            Component ID for each player

        Returns
        -------
        dict[str, float]
            Correlation for each component
        """
        # Group players by component
        components = {}
        for player, comp_id in player_components.items():
            if comp_id not in components:
                components[comp_id] = []
            components[comp_id].append(player)

        results = {}
        for comp_id, players in components.items():
            if len(players) < 2:
                continue

            # Get ranks for this component
            comp_true = [
                true_ranks[p]
                for p in players
                if p in true_ranks and p in predicted_ranks
            ]
            comp_pred = [
                predicted_ranks[p]
                for p in players
                if p in true_ranks and p in predicted_ranks
            ]

            if len(comp_true) < 2:
                continue

            correlation, _ = stats.spearmanr(comp_true, comp_pred)
            results[f"component_{comp_id}"] = {
                "correlation": correlation,
                "size": len(players),
            }

        return results

    def calculate_comprehensive_metrics(
        self,
        true_ranks: dict[str, int],
        predicted_ranks: dict[str, int],
        players_df: pl.DataFrame,
        matches_df: pl.DataFrame,
    ) -> dict:
        """
        Calculate all confidence-weighted metrics.

        Parameters
        ----------
        true_ranks : dict[str, int]
            True rankings
        predicted_ranks : dict[str, int]
            Predicted rankings
        players_df : pl.DataFrame
            Player data
        matches_df : pl.DataFrame
            Match data

        Returns
        -------
        dict
            Comprehensive evaluation metrics
        """
        # Calculate confidence scores
        if self.confidence_calculator is None:
            n_players = len(set(players_df["user_id"].to_list()))
            self.confidence_calculator = RankingConfidence(
                total_players=n_players
            )

        confidence_df = self.confidence_calculator.calculate_all_confidences(
            players_df, matches_df
        )

        # Convert to dictionaries
        confidence_scores = {}
        player_tiers = {}
        player_activity = {}

        for row in confidence_df.iter_rows(named=True):
            player_id = row["player_id"]
            confidence_scores[player_id] = row["confidence_score"]
            player_tiers[player_id] = ConfidenceTier(row["confidence_tier"])
            player_activity[player_id] = row["tournament_count"]

        # Calculate metrics
        results = {
            # Overall metrics
            "standard_spearman": stats.spearmanr(
                [true_ranks[p] for p in true_ranks if p in predicted_ranks],
                [
                    predicted_ranks[p]
                    for p in true_ranks
                    if p in predicted_ranks
                ],
            )[0]
            if len(set(true_ranks.keys()) & set(predicted_ranks.keys())) > 1
            else 0,
            # Weighted metrics
            "confidence_weighted_spearman": self.weighted_spearman_correlation(
                true_ranks, predicted_ranks, confidence_scores
            ),
            "activity_weighted_spearman": self.activity_weighted_correlation(
                true_ranks, predicted_ranks, player_activity
            ),
            # Tier-specific metrics
            "tier_correlations": self.tier_specific_correlation(
                true_ranks, predicted_ranks, player_tiers
            ),
            # Tier distribution
            "tier_distribution": {
                tier.value: sum(1 for t in player_tiers.values() if t == tier)
                for tier in ConfidenceTier
            },
        }

        # Add summary statistics
        tier_corrs = results["tier_correlations"]
        valid_tier_corrs = [c for c in tier_corrs.values() if c is not None]

        if valid_tier_corrs:
            results["mean_tier_correlation"] = np.mean(valid_tier_corrs)
            results["tier_correlation_spread"] = np.std(valid_tier_corrs)

        # Improvement metrics
        results["weighted_improvement"] = (
            results["confidence_weighted_spearman"]
            - results["standard_spearman"]
        )
        results["weighted_improvement_pct"] = (
            100
            * results["weighted_improvement"]
            / abs(results["standard_spearman"])
            if results["standard_spearman"] != 0
            else 0
        )

        return results

    def print_evaluation_summary(self, metrics: dict) -> None:
        """Print formatted evaluation summary."""
        print("\n" + "=" * 60)
        print("CONFIDENCE-WEIGHTED EVALUATION SUMMARY")
        print("=" * 60)

        print("\n📊 Overall Correlations:")
        print(
            f"  Standard (all players equal):     {metrics['standard_spearman']:.3f}"
        )
        print(
            f"  Confidence-weighted:               {metrics['confidence_weighted_spearman']:.3f}"
        )
        print(
            f"  Activity-weighted:                 {metrics['activity_weighted_spearman']:.3f}"
        )
        print(
            f"  Improvement with weighting:        {metrics['weighted_improvement_pct']:+.1f}%"
        )

        print("\n🎯 Correlation by Confidence Tier:")
        for tier, corr in metrics["tier_correlations"].items():
            count = metrics["tier_distribution"][tier]
            if corr is not None:
                print(
                    f"  {tier.capitalize():12} ({count:4} players): {corr:.3f}"
                )
            else:
                print(f"  {tier.capitalize():12} ({count:4} players): N/A")

        if "mean_tier_correlation" in metrics:
            print(
                f"\n  Mean tier correlation:    {metrics['mean_tier_correlation']:.3f}"
            )
            print(
                f"  Tier correlation spread:  ±{metrics['tier_correlation_spread']:.3f}"
            )

        print("\n💡 Key Insight:")
        if metrics["weighted_improvement_pct"] > 10:
            print(
                "  ✅ Confidence weighting significantly improves correlation!"
            )
            print(
                "     This confirms that low-confidence players were hurting metrics."
            )
        elif metrics["weighted_improvement_pct"] > 0:
            print("  ✅ Modest improvement with confidence weighting.")
            print(
                "     The ranking system handles sparse data reasonably well."
            )
        else:
            print("  ⚠️  No improvement with weighting - investigate further.")

        print("\n" + "=" * 60)
