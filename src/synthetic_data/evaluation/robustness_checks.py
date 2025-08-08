"""
Robustness checks for PageRank evaluation.

This module provides various robustness checks to ensure the PageRank
implementation is stable and not overfitting to specific data patterns.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.stats import kendalltau, spearmanr

from rankings.analysis.engine import RatingEngine
from synthetic_data.evaluation.metrics import (
    ranking_correlation,
    top_k_accuracy,
)


class RobustnessChecker:
    """
    Performs robustness checks on PageRank rankings.

    These checks help identify if the ranking algorithm is:
    - Stable to hyperparameter choices
    - Robust to data perturbations
    - Not overfitting to specific patterns
    """

    def __init__(self, base_engine_params: Optional[dict] = None):
        """
        Initialize robustness checker.

        Parameters
        ----------
        base_engine_params : dict, optional
            Base parameters for RatingEngine
        """
        self.base_params = base_engine_params or {}

    def teleport_sweep_stability(
        self,
        matches: pl.DataFrame,
        players: Optional[pl.DataFrame] = None,
        teleport_options: Optional[List[str]] = None,
        top_k_values: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Test ranking stability across different teleport vector options.

        Parameters
        ----------
        matches : pl.DataFrame
            Match data
        players : pl.DataFrame, optional
            Player data (for player rankings)
        teleport_options : list, optional
            Teleport options to test (default: uniform, volume_inverse, volume_mix)
        top_k_values : list, optional
            Top-k values for evaluation (default: [10, 25, 50])

        Returns
        -------
        dict
            Correlation metrics between different teleport options
        """
        if teleport_options is None:
            teleport_options = ["uniform", "volume_inverse", "volume_mix"]

        if top_k_values is None:
            top_k_values = [10, 25, 50]

        # Compute rankings for each teleport option
        rankings = {}
        for teleport in teleport_options:
            params = {**self.base_params, "teleport": teleport}
            engine = RatingEngine(**params)

            if players is not None:
                result = engine.rank_players(matches, players)
                # Rename player_rank back to score for consistency
                if "player_rank" in result.columns:
                    result = result.rename({"player_rank": "score"})
            else:
                result = engine.rank_teams(matches)
                # Rename team_rank back to score for consistency
                if "team_rank" in result.columns:
                    result = result.rename({"team_rank": "score"})

            rankings[teleport] = result.sort("score", descending=True)

        # Compare rankings between options
        results = {}
        for i, opt1 in enumerate(teleport_options):
            for opt2 in teleport_options[i + 1 :]:
                comparison_key = f"{opt1}_vs_{opt2}"

                # Get scores for common entities
                df1 = rankings[opt1]
                df2 = rankings[opt2]

                # Join on id column (which is just "id" in the result)
                id_col = (
                    "id"
                    if "id" in df1.columns
                    else "user_id"
                    if "user_id" in df1.columns
                    else "team_id"
                )
                joined = df1.join(df2, on=id_col, suffix="_2")

                scores1 = joined["score"].to_numpy()
                scores2 = joined["score_2"].to_numpy()

                # Calculate correlations
                kendall_tau, _ = kendalltau(scores1, scores2)
                spearman_rho, _ = spearmanr(scores1, scores2)

                # Calculate top-k Jaccard similarities
                top_k_metrics = {}
                for k in top_k_values:
                    if k <= len(scores1):
                        # Get top-k indices
                        top_k_1 = set(np.argsort(-scores1)[:k])
                        top_k_2 = set(np.argsort(-scores2)[:k])

                        # Jaccard similarity
                        intersection = len(top_k_1 & top_k_2)
                        union = len(top_k_1 | top_k_2)
                        jaccard = intersection / union if union > 0 else 0
                        top_k_metrics[f"top_{k}_jaccard"] = jaccard

                results[comparison_key] = {
                    "kendall_tau": kendall_tau,
                    "spearman_rho": spearman_rho,
                    **top_k_metrics,
                }

        return results

    def edge_drop_perturbation(
        self,
        matches: pl.DataFrame,
        players: Optional[pl.DataFrame] = None,
        drop_fraction: float = 0.05,
        n_trials: int = 5,
        stratified: bool = True,
        top_k: int = 25,
    ) -> Dict[str, float]:
        """
        Test robustness to randomly dropping edges (matches).

        Parameters
        ----------
        matches : pl.DataFrame
            Match data
        players : pl.DataFrame, optional
            Player data (for player rankings)
        drop_fraction : float
            Fraction of matches to drop
        n_trials : int
            Number of random trials
        stratified : bool
            Whether to stratify drops by tournament
        top_k : int
            Top-k for Jaccard similarity

        Returns
        -------
        dict
            Stability metrics across perturbations
        """
        # Get baseline rankings
        engine = RatingEngine(**self.base_params)
        if players is not None:
            baseline = engine.rank_players(matches, players)
            id_col = "id"  # Result uses "id" column
        else:
            baseline = engine.rank_teams(matches)
            id_col = "id"  # Result uses "id" column

        # Determine score column based on what's available
        if "team_rank" in baseline.columns:
            score_col = "team_rank"
        elif "player_rank" in baseline.columns:
            score_col = "player_rank"
        else:
            score_col = "score"
        baseline_scores = dict(zip(baseline[id_col], baseline[score_col]))

        # Run perturbation trials
        kendall_values = []
        spearman_values = []
        jaccard_values = []

        for trial in range(n_trials):
            # Drop random matches
            if stratified and "tournament_id" in matches.columns:
                # Stratified sampling by tournament
                perturbed = matches.group_by("tournament_id").map_groups(
                    lambda df: df.sample(fraction=1 - drop_fraction, seed=trial)
                )
            else:
                # Simple random sampling
                perturbed = matches.sample(
                    fraction=1 - drop_fraction, seed=trial
                )

            # Recompute rankings
            engine_perturbed = RatingEngine(**self.base_params)
            if players is not None:
                result = engine_perturbed.rank_players(perturbed, players)
            else:
                result = engine_perturbed.rank_teams(perturbed)

            # Compare with baseline
            # Determine score column based on what's available
            if "team_rank" in result.columns:
                score_col = "team_rank"
            elif "player_rank" in result.columns:
                score_col = "player_rank"
            else:
                score_col = "score"
            perturbed_scores = dict(zip(result[id_col], result[score_col]))

            # Get common entities
            common_ids = set(baseline_scores.keys()) & set(
                perturbed_scores.keys()
            )
            if len(common_ids) == 0:
                continue

            base_vals = np.array([baseline_scores[id] for id in common_ids])
            pert_vals = np.array([perturbed_scores[id] for id in common_ids])

            # Calculate correlations
            kendall, _ = kendalltau(base_vals, pert_vals)
            spearman, _ = spearmanr(base_vals, pert_vals)

            kendall_values.append(kendall)
            spearman_values.append(spearman)

            # Top-k Jaccard
            if top_k <= len(base_vals):
                top_k_base = set(np.argsort(-base_vals)[:top_k])
                top_k_pert = set(np.argsort(-pert_vals)[:top_k])

                intersection = len(top_k_base & top_k_pert)
                union = len(top_k_base | top_k_pert)
                jaccard = intersection / union if union > 0 else 0
                jaccard_values.append(jaccard)

        return {
            "kendall_tau_mean": np.mean(kendall_values),
            "kendall_tau_std": np.std(kendall_values),
            "spearman_rho_mean": np.mean(spearman_values),
            "spearman_rho_std": np.std(spearman_values),
            f"top_{top_k}_jaccard_mean": np.mean(jaccard_values)
            if jaccard_values
            else 0,
            f"top_{top_k}_jaccard_std": np.std(jaccard_values)
            if jaccard_values
            else 0,
        }

    def null_shuffle_test(
        self,
        matches: pl.DataFrame,
        players: Optional[pl.DataFrame] = None,
        n_trials: int = 5,
    ) -> Dict[str, float]:
        """
        Test behavior with shuffled (random) match outcomes.

        Rankings should collapse near uniform when outcomes are random.

        Parameters
        ----------
        matches : pl.DataFrame
            Match data
        players : pl.DataFrame, optional
            Player data (for player rankings)
        n_trials : int
            Number of shuffle trials

        Returns
        -------
        dict
            Metrics showing how much rankings deviate from uniform
        """
        results = []

        for trial in range(n_trials):
            # Shuffle winners and losers
            shuffled = matches.clone()

            # Randomly swap winners and losers with 50% probability
            np.random.seed(trial)
            swap_mask = np.random.random(len(shuffled)) < 0.5

            # Create swap expressions
            shuffled = shuffled.with_columns(
                [
                    pl.when(pl.lit(swap_mask))
                    .then(pl.col("loser_team_id"))
                    .otherwise(pl.col("winner_team_id"))
                    .alias("winner_team_id_new"),
                    pl.when(pl.lit(swap_mask))
                    .then(pl.col("winner_team_id"))
                    .otherwise(pl.col("loser_team_id"))
                    .alias("loser_team_id_new"),
                ]
            )

            # Replace original columns
            shuffled = shuffled.drop(["winner_team_id", "loser_team_id"])
            shuffled = shuffled.rename(
                {
                    "winner_team_id_new": "winner_team_id",
                    "loser_team_id_new": "loser_team_id",
                }
            )

            # Compute rankings on shuffled data
            engine = RatingEngine(**self.base_params)
            if players is not None:
                result = engine.rank_players(shuffled, players)
                score_col = (
                    "player_rank"
                    if "player_rank" in result.columns
                    else "score"
                )
            else:
                result = engine.rank_teams(shuffled)
                score_col = (
                    "team_rank" if "team_rank" in result.columns else "score"
                )

            # Measure deviation from uniform
            scores = result[score_col].to_numpy()

            # Normalized entropy (should be close to 1 for uniform)
            # Using coefficient of variation as simpler metric
            cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0

            # Gini coefficient (0 = perfect equality, 1 = perfect inequality)
            gini = self._gini_coefficient(scores)

            results.append(
                {
                    "cv": cv,
                    "gini": gini,
                    "score_std": np.std(scores),
                    "score_range": np.max(scores) - np.min(scores),
                }
            )

        # Average across trials
        return {
            "cv_mean": np.mean([r["cv"] for r in results]),
            "cv_std": np.std([r["cv"] for r in results]),
            "gini_mean": np.mean([r["gini"] for r in results]),
            "gini_std": np.std([r["gini"] for r in results]),
            "score_std_mean": np.mean([r["score_std"] for r in results]),
            "score_range_mean": np.mean([r["score_range"] for r in results]),
        }

    def monotonicity_test(
        self,
        matches: pl.DataFrame,
        players: Optional[pl.DataFrame] = None,
        test_fraction: float = 0.1,
    ) -> Dict[str, bool]:
        """
        Test monotonicity: adding a loss should not increase score.

        Parameters
        ----------
        matches : pl.DataFrame
            Match data
        players : pl.DataFrame, optional
            Player data (for player rankings)
        test_fraction : float
            Fraction of entities to test

        Returns
        -------
        dict
            Test results and violation statistics
        """
        # Get baseline rankings
        engine = RatingEngine(**self.base_params)
        if players is not None:
            baseline = engine.rank_players(matches, players)
            id_col = "id"  # Result uses "id" column
        else:
            baseline = engine.rank_teams(matches)
            id_col = "id"  # Result uses "id" column

        # Determine score column based on what's available
        if "team_rank" in baseline.columns:
            score_col = "team_rank"
        elif "player_rank" in baseline.columns:
            score_col = "player_rank"
        else:
            score_col = "score"
        baseline_scores = dict(zip(baseline[id_col], baseline[score_col]))

        # Sample entities to test
        n_test = max(1, int(len(baseline) * test_fraction))
        test_ids = baseline.sample(n=n_test)[id_col].to_list()

        violations = 0
        tests_performed = 0

        for test_id in test_ids:
            # Add a synthetic loss for this entity
            if players is not None:
                # Find a team this player is on
                player_teams = players.filter(pl.col("user_id") == test_id)
                if len(player_teams) == 0:
                    continue

                team_id = player_teams["team_id"][0]
                tournament_id = player_teams["tournament_id"][0]

                # Create a synthetic loss
                # Find another team in the same tournament
                other_teams = (
                    matches.filter(pl.col("tournament_id") == tournament_id)
                    .select("winner_team_id")
                    .unique()
                )

                if len(other_teams) == 0:
                    continue

                winner_id = other_teams["winner_team_id"][0]
            else:
                # For team rankings, just pick any other team
                team_id = test_id
                other_teams = baseline.filter(pl.col(id_col) != test_id)
                if len(other_teams) == 0:
                    continue
                winner_id = other_teams[id_col][0]
                tournament_id = matches["tournament_id"][
                    0
                ]  # Use first tournament

            # Create synthetic match (cast to match schema types)
            synthetic_match = pl.DataFrame(
                {
                    "match_id": [int(matches["match_id"].max() + 1)],
                    "tournament_id": [int(tournament_id)],
                    "winner_team_id": [int(winner_id)],
                    "loser_team_id": [int(team_id)],
                    "winner_score": [2],
                    "loser_score": [0],
                    "match_created_at": [
                        int(matches["match_created_at"].max())
                    ],
                }
            )

            # Add synthetic match
            augmented = pl.concat([matches, synthetic_match])

            # Recompute rankings
            engine_aug = RatingEngine(**self.base_params)
            if players is not None:
                result = engine_aug.rank_players(augmented, players)
                score_col = (
                    "player_rank"
                    if "player_rank" in result.columns
                    else "score"
                )
            else:
                result = engine_aug.rank_teams(augmented)
                score_col = (
                    "team_rank" if "team_rank" in result.columns else "score"
                )

            # Check if score increased
            new_scores = dict(zip(result[id_col], result[score_col]))

            if test_id in new_scores and test_id in baseline_scores:
                if new_scores[test_id] > baseline_scores[test_id]:
                    violations += 1
                tests_performed += 1

        return {
            "tests_performed": tests_performed,
            "violations": violations,
            "violation_rate": violations / tests_performed
            if tests_performed > 0
            else 0,
            "passed": violations == 0,
        }

    def _gini_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient for inequality measurement.

        Parameters
        ----------
        values : np.ndarray
            Values to measure inequality of

        Returns
        -------
        float
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
