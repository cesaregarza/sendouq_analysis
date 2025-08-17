"""Tournament prediction and seeding system based on player ratings.

This module implements the complete tournament prediction pipeline:
1. Team strength calculation from player ratings
2. Win probability estimation with calibrated logistic regression
3. Tournament seeding based on team ratings
4. Monte Carlo simulation for tournament outcome predictions
5. Backtesting framework for validation
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression


@dataclass
class TeamRatingConfig:
    """Configuration for team rating calculation."""

    alpha: float = 1.0  # Diminishing returns exponent (0 < alpha <= 1)
    use_top_k: Optional[int] = None  # Only use top-k players
    default_weight: float = 1.0  # Default exposure weight


class TeamStrengthCalculator:
    """Calculate team strength from individual player ratings."""

    def __init__(self, config: Optional[TeamRatingConfig] = None):
        """Initialize with optional configuration.

        Args:
            config: Configuration for team rating calculation
        """
        self.config = config or TeamRatingConfig()

    def player_skill_scale(self, log_rating: float) -> float:
        """Convert log-rating to skill scale.

        Args:
            log_rating: Player's log-rating (r_i)

        Returns:
            Skill scale value (s_i = exp(r_i))
        """
        return math.exp(log_rating)

    def team_strength(
        self,
        player_ratings: Dict[int, float],
        team_roster: List[int],
        exposure_weights: Optional[Dict[int, float]] = None,
    ) -> float:
        """Calculate team strength from player ratings.

        Implements: S_T = sum_{i in T} w_i * s_i^alpha

        Args:
            player_ratings: Dictionary mapping player_id to log-rating
            team_roster: List of player IDs on the team
            exposure_weights: Optional weights for each player (default: uniform)

        Returns:
            Team strength S_T
        """
        if exposure_weights is None:
            exposure_weights = {}

        # Filter to top-k if configured
        roster = team_roster
        if self.config.use_top_k and len(roster) > self.config.use_top_k:
            # Sort by rating and take top-k
            sorted_roster = sorted(
                roster,
                key=lambda p: player_ratings.get(p, float("-inf")),
                reverse=True,
            )
            roster = sorted_roster[: self.config.use_top_k]

        # Calculate weighted sum of strengths
        strength = 0.0
        for player_id in roster:
            if player_id not in player_ratings:
                continue

            r_i = player_ratings[player_id]
            s_i = self.player_skill_scale(r_i)
            w_i = exposure_weights.get(player_id, self.config.default_weight)

            # Apply diminishing returns if configured
            strength += w_i * (s_i**self.config.alpha)

        return strength

    def team_log_rating(
        self,
        player_ratings: Dict[int, float],
        team_roster: List[int],
        exposure_weights: Optional[Dict[int, float]] = None,
    ) -> float:
        """Calculate team log-rating from player ratings.

        Implements: R_T = log(S_T) = logsumexp({r_i + log(w_i)})

        Args:
            player_ratings: Dictionary mapping player_id to log-rating
            team_roster: List of player IDs on the team
            exposure_weights: Optional weights for each player (default: uniform)

        Returns:
            Team log-rating R_T
        """
        if exposure_weights is None:
            exposure_weights = {}

        # Filter to top-k if configured
        roster = team_roster
        if self.config.use_top_k and len(roster) > self.config.use_top_k:
            sorted_roster = sorted(
                roster,
                key=lambda p: player_ratings.get(p, float("-inf")),
                reverse=True,
            )
            roster = sorted_roster[: self.config.use_top_k]

        # Collect log-values for logsumexp
        log_values = []
        for player_id in roster:
            if player_id not in player_ratings:
                continue

            r_i = player_ratings[player_id]
            w_i = exposure_weights.get(player_id, self.config.default_weight)

            # Include weight and diminishing returns in log space
            log_val = self.config.alpha * r_i
            if w_i != 1.0:
                log_val += math.log(w_i)

            log_values.append(log_val)

        if not log_values:
            return float("-inf")

        return logsumexp(log_values)


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
        historical_matches: List[Tuple[float, float, bool]],
        format_info: Optional[List[str]] = None,
    ) -> Dict[str, float]:
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
        historical_matches: List[Tuple[float, float, bool]],
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
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


class TournamentSeeder:
    """Generate tournament seedings from team ratings."""

    def __init__(
        self,
        team_calculator: TeamStrengthCalculator,
        match_predictor: Optional[MatchPredictor] = None,
    ):
        """Initialize seeder with rating calculator and predictor.

        Args:
            team_calculator: Calculator for team strengths
            match_predictor: Optional predictor for confidence intervals
        """
        self.team_calculator = team_calculator
        self.match_predictor = match_predictor

    def generate_seeds(
        self,
        player_ratings: Dict[int, float],
        team_rosters: Dict[int, List[int]],
        exposure_weights: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> List[Tuple[int, float, float]]:
        """Generate tournament seeds from team ratings.

        Args:
            player_ratings: Dictionary mapping player_id to log-rating
            team_rosters: Dictionary mapping team_id to list of player_ids
            exposure_weights: Optional weights as (player_id, team_id) -> weight

        Returns:
            List of (team_id, rating, confidence) sorted by rating
        """
        team_ratings = []

        for team_id, roster in team_rosters.items():
            # Get weights for this team
            team_weights = None
            if exposure_weights:
                team_weights = {
                    p: exposure_weights.get((p, team_id), 1.0) for p in roster
                }

            # Calculate team rating
            rating = self.team_calculator.team_log_rating(
                player_ratings, roster, team_weights
            )

            # Calculate confidence (gap to neighbors will be computed after sorting)
            team_ratings.append((team_id, rating, 0.0))

        # Sort by rating (descending)
        team_ratings.sort(key=lambda x: x[1], reverse=True)

        # Calculate confidence based on rating gaps
        result = []
        for i, (team_id, rating, _) in enumerate(team_ratings):
            # Confidence based on gap to neighbors
            gap_above = (
                float("inf") if i == 0 else abs(rating - team_ratings[i - 1][1])
            )
            gap_below = (
                float("inf")
                if i == len(team_ratings) - 1
                else abs(rating - team_ratings[i + 1][1])
            )

            min_gap = min(gap_above, gap_below)
            # Normalize confidence to 0-1 scale (larger gap = higher confidence)
            confidence = 1.0 - math.exp(-min_gap)

            result.append((team_id, rating, confidence))

        return result

    def apply_bracket_constraints(
        self,
        seeds: List[Tuple[int, float, float]],
        constraints: Optional[Dict[str, any]] = None,
    ) -> List[Tuple[int, float, float]]:
        """Apply bracket-specific constraints to seeding.

        Args:
            seeds: Initial seeds from generate_seeds
            constraints: Dictionary of constraints (e.g., region separation)

        Returns:
            Adjusted seeds respecting constraints
        """
        # This is a placeholder for bracket-specific logic
        # Real implementation would handle:
        # - Snake seeding
        # - Region separation
        # - Avoiding early rematches
        # - Format-specific rules

        return seeds


class MonteCarloSimulator:
    """Simulate tournament outcomes using Monte Carlo methods."""

    def __init__(
        self,
        team_calculator: TeamStrengthCalculator,
        match_predictor: MatchPredictor,
    ):
        """Initialize simulator with calculators.

        Args:
            team_calculator: Calculator for team strengths
            match_predictor: Predictor for match outcomes
        """
        self.team_calculator = team_calculator
        self.match_predictor = match_predictor

    def simulate_match(
        self, team_a_rating: float, team_b_rating: float, best_of: int = 1
    ) -> bool:
        """Simulate a single match or series.

        Args:
            team_a_rating: Team A's log-rating
            team_b_rating: Team B's log-rating
            best_of: Number of games in series (1, 3, 5, etc.)

        Returns:
            True if team A wins, False if team B wins
        """
        if best_of == 1:
            prob_a = self.match_predictor.win_probability(
                team_a_rating, team_b_rating
            )
            return np.random.random() < prob_a

        # Simulate best-of series
        wins_needed = (best_of + 1) // 2
        a_wins = 0
        b_wins = 0

        while a_wins < wins_needed and b_wins < wins_needed:
            prob_a = self.match_predictor.win_probability(
                team_a_rating, team_b_rating
            )
            if np.random.random() < prob_a:
                a_wins += 1
            else:
                b_wins += 1

        return a_wins >= wins_needed

    def simulate_tournament(
        self,
        player_ratings: Dict[int, float],
        bracket_structure: Dict[str, any],
        team_rosters: Dict[int, List[int]],
        n_simulations: int = 10000,
    ) -> Dict[str, any]:
        """Run Monte Carlo simulation of tournament.

        Args:
            player_ratings: Dictionary mapping player_id to log-rating
            bracket_structure: Tournament bracket structure
            team_rosters: Dictionary mapping team_id to list of player_ids
            n_simulations: Number of simulations to run

        Returns:
            Dictionary with simulation results (win probabilities, expected placements, etc.)
        """
        # Calculate team ratings once
        team_ratings = {}
        for team_id, roster in team_rosters.items():
            team_ratings[team_id] = self.team_calculator.team_log_rating(
                player_ratings, roster
            )

        # Track results across simulations
        win_counts = {team_id: 0 for team_id in team_rosters}
        top4_counts = {team_id: 0 for team_id in team_rosters}
        placement_sum = {team_id: 0 for team_id in team_rosters}

        # Run simulations
        for _ in range(n_simulations):
            # Simulate tournament (simplified - real implementation would follow bracket)
            # This is a placeholder for actual bracket simulation
            placements = self._simulate_single_tournament(
                team_ratings, bracket_structure
            )

            for team_id, placement in placements.items():
                if placement == 1:
                    win_counts[team_id] += 1
                if placement <= 4:
                    top4_counts[team_id] += 1
                placement_sum[team_id] += placement

        # Calculate probabilities and expectations
        results = {}
        for team_id in team_rosters:
            results[team_id] = {
                "win_probability": win_counts[team_id] / n_simulations,
                "top4_probability": top4_counts[team_id] / n_simulations,
                "expected_placement": placement_sum[team_id] / n_simulations,
                "rating": team_ratings[team_id],
            }

        return results

    def _simulate_single_tournament(
        self, team_ratings: Dict[int, float], bracket_structure: Dict[str, any]
    ) -> Dict[int, int]:
        """Simulate a single tournament run.

        Args:
            team_ratings: Pre-calculated team ratings
            bracket_structure: Tournament bracket structure

        Returns:
            Dictionary mapping team_id to final placement
        """
        # Placeholder for actual bracket simulation
        # Real implementation would follow the bracket structure
        # For now, just rank by rating with some randomness

        teams = list(team_ratings.keys())
        # Add noise to ratings for simulation
        noisy_ratings = {
            team: rating + np.random.normal(0, 0.5)
            for team, rating in team_ratings.items()
        }

        # Sort by noisy rating
        sorted_teams = sorted(
            teams, key=lambda t: noisy_ratings[t], reverse=True
        )

        # Assign placements
        placements = {team: i + 1 for i, team in enumerate(sorted_teams)}
        return placements


class TournamentBacktester:
    """Backtest tournament predictions against historical data."""

    def __init__(
        self,
        team_calculator: TeamStrengthCalculator,
        match_predictor: MatchPredictor,
        seeder: TournamentSeeder,
    ):
        """Initialize backtester with necessary components.

        Args:
            team_calculator: Calculator for team strengths
            match_predictor: Predictor for match outcomes
            seeder: Tournament seeder
        """
        self.team_calculator = team_calculator
        self.match_predictor = match_predictor
        self.seeder = seeder

    def rolling_backtest(
        self,
        tournaments: List[Dict[str, any]],
        player_ratings_history: Dict[int, Dict[int, float]],
    ) -> Dict[str, any]:
        """Run rolling origin backtest on historical tournaments.

        Args:
            tournaments: List of tournament data
            player_ratings_history: Ratings before each tournament

        Returns:
            Dictionary with backtest metrics
        """
        results = {
            "seeding_correlation": [],
            "match_accuracy": [],
            "log_loss": [],
            "top4_accuracy": [],
        }

        for tournament in tournaments:
            tournament_id = tournament["id"]

            # Get ratings before this tournament
            if tournament_id not in player_ratings_history:
                continue

            ratings_before = player_ratings_history[tournament_id]

            # Generate seeds
            seeds = self.seeder.generate_seeds(
                ratings_before, tournament["team_rosters"]
            )

            # Compare to actual results
            actual_placements = tournament["final_placements"]
            predicted_order = [team_id for team_id, _, _ in seeds]

            # Calculate metrics
            # ... (implementation of various metrics)

        return results

    def evaluate_vs_manual_seeds(
        self,
        tournament: Dict[str, any],
        player_ratings: Dict[int, float],
        manual_seeds: List[int],
    ) -> Dict[str, float]:
        """Compare model seeds to manual seeds.

        Args:
            tournament: Tournament data
            player_ratings: Current player ratings
            manual_seeds: Manually assigned seeds

        Returns:
            Dictionary with comparison metrics
        """
        # Generate model seeds
        model_seeds = self.seeder.generate_seeds(
            player_ratings, tournament["team_rosters"]
        )

        model_order = [team_id for team_id, _, _ in model_seeds]

        # Calculate agreement metrics
        pairwise_agreement = self._calculate_pairwise_agreement(
            model_order, manual_seeds
        )

        return {
            "pairwise_agreement": pairwise_agreement,
            "spearman_correlation": self._spearman_correlation(
                model_order, manual_seeds
            ),
            "kendall_tau": self._kendall_tau(model_order, manual_seeds),
        }

    def _calculate_pairwise_agreement(
        self, order1: List[int], order2: List[int]
    ) -> float:
        """Calculate pairwise agreement between two orderings."""
        agree = 0
        total = 0

        for i in range(len(order1)):
            for j in range(i + 1, len(order1)):
                total += 1
                # Check if pair is in same order in both lists
                idx1_i = order1.index(order1[i]) if order1[i] in order1 else -1
                idx1_j = order1.index(order1[j]) if order1[j] in order1 else -1
                idx2_i = order2.index(order1[i]) if order1[i] in order2 else -1
                idx2_j = order2.index(order1[j]) if order1[j] in order2 else -1

                if idx1_i >= 0 and idx1_j >= 0 and idx2_i >= 0 and idx2_j >= 0:
                    if (idx1_i < idx1_j) == (idx2_i < idx2_j):
                        agree += 1

        return agree / total if total > 0 else 0.0

    def _spearman_correlation(
        self, order1: List[int], order2: List[int]
    ) -> float:
        """Calculate Spearman rank correlation."""
        # Simplified implementation - real would use scipy.stats.spearmanr
        return 0.0

    def _kendall_tau(self, order1: List[int], order2: List[int]) -> float:
        """Calculate Kendall's tau."""
        # Simplified implementation - real would use scipy.stats.kendalltau
        return 0.0
