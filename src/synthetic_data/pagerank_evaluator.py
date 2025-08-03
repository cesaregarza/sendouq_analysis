"""
PageRank evaluation module for synthetic tournament data.

This module evaluates how well PageRank rankings correlate with true player
skills in synthetic tournament circuits.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats

from rankings.analysis import RatingEngine
from rankings.core.constants import (
    DEFAULT_BETA,
    DEFAULT_DAMPING_FACTOR,
    DEFAULT_DECAY_RATE,
)
from synthetic_data.player_generator import SyntheticPlayer
from synthetic_data.tournament_circuit import CircuitResults, TournamentCircuit
from synthetic_data.weighted_correlation import (
    rank_difference_distribution,
    top_k_weighted_accuracy,
    weighted_spearman,
)


@dataclass
class PageRankEvaluation:
    """Results from PageRank evaluation on synthetic data."""

    # Correlation metrics
    spearman_correlation: float
    weighted_spearman: float  # New: weighted correlation
    kendall_tau: float
    pearson_correlation: float

    # Ranking accuracy metrics
    top_10_accuracy: float  # % of true top 10 in predicted top 10
    top_20_accuracy: float
    top_50_accuracy: float

    # Weighted accuracy metrics
    top_10_weighted_accuracy: float
    top_20_weighted_accuracy: float
    top_50_weighted_accuracy: float

    # Error metrics
    mean_rank_error: float
    median_rank_error: float
    rmse_rank: float
    top_10_mean_error: float  # New: error specifically for top players

    # PageRank specific metrics
    n_iterations: int
    convergence_error: float

    # Additional analysis
    rank_by_player_id: Dict[int, int]
    true_rank_by_player_id: Dict[int, int]
    pagerank_scores: Dict[int, float]
    participation_counts: Dict[int, int]


class PageRankEvaluator:
    """Evaluates PageRank performance on synthetic tournament circuits."""

    def __init__(
        self,
        damping_factor: float = DEFAULT_DAMPING_FACTOR,
        decay_rate: float = DEFAULT_DECAY_RATE,
        beta: float = DEFAULT_BETA,
        min_tournaments: int = 3,
        influence_agg_method: str = "mean",
    ):
        """
        Initialize the PageRank evaluator.

        Parameters
        ----------
        damping_factor : float
            PageRank damping factor
        decay_rate : float
            Time decay rate for matches
        beta : float
            Tournament strength weighting parameter
        min_tournaments : int
            Minimum tournaments for a player to be ranked
        influence_agg_method : str
            Method for aggregating participant ratings ("mean", "sum", "median", "top_20_sum")
        """
        self.damping_factor = damping_factor
        self.decay_rate = decay_rate
        self.beta = beta
        self.min_tournaments = min_tournaments
        self.influence_agg_method = influence_agg_method

        # Initialize rating engine
        self.engine = RatingEngine(
            damping_factor=damping_factor,
            decay_half_life_days=30.0
            if decay_rate == DEFAULT_DECAY_RATE
            else np.log(2) / decay_rate,
            beta=beta,
            influence_agg_method=influence_agg_method,
        )

    def evaluate_circuit(
        self,
        circuit: TournamentCircuit,
        circuit_results: CircuitResults,
    ) -> PageRankEvaluation:
        """
        Evaluate PageRank on a tournament circuit.

        Parameters
        ----------
        circuit : TournamentCircuit
            The tournament circuit generator
        circuit_results : CircuitResults
            Results from circuit simulation

        Returns
        -------
        PageRankEvaluation
            Evaluation metrics
        """
        # Convert circuit data to format expected by ranking engine
        matches_df = self._circuit_to_matches_df(circuit_results)

        # Run PageRank via RatingEngine using the players dataframe we created
        rankings = self.engine.rank_players(matches_df, self._players_df)

        # Filter to players with minimum participation
        active_players = self._get_active_players(circuit_results)

        # Calculate true rankings based on skill
        true_rankings = self._calculate_true_rankings(
            circuit.player_pool, active_players
        )

        # Compute evaluation metrics
        evaluation = self._compute_metrics(
            rankings,
            true_rankings,
            circuit_results,
            active_players,
            circuit.player_pool,
        )

        return evaluation

    def _circuit_to_matches_df(
        self, circuit_results: CircuitResults
    ) -> pl.DataFrame:
        """Convert circuit results to matches DataFrame."""
        matches_data = []

        for tournament in circuit_results.tournaments:
            for stage in tournament.stages:
                for round_matches in stage.rounds.values():
                    for match in round_matches:
                        if match.winner is None:
                            continue

                        # Determine winner and loser team IDs
                        if match.winner == match.team_a:
                            winner_team_id = match.team_a.team_id
                            loser_team_id = match.team_b.team_id
                        else:
                            winner_team_id = match.team_b.team_id
                            loser_team_id = match.team_a.team_id

                        # Create match record in the format expected by RatingEngine
                        match_record = {
                            "tournament_id": tournament.tournament_id,
                            "tournament_name": tournament.name,
                            "match_id": match.match_id,
                            "team1_id": match.team_a.team_id,
                            "team2_id": match.team_b.team_id,
                            "winner_team_id": winner_team_id,
                            "loser_team_id": loser_team_id,
                            "team1_score": match.score_a,
                            "team2_score": match.score_b,
                            "score_diff": abs(match.score_a - match.score_b),
                            "total_games": match.score_a + match.score_b,
                            "timestamp": match.timestamp,
                            "last_game_finished_at": int(
                                match.timestamp.timestamp()
                            )
                            if match.timestamp
                            else None,
                            "match_created_at": int(match.timestamp.timestamp())
                            if match.timestamp
                            else None,
                            "stage": match.stage,
                            "round_number": match.round_number,
                            "is_bye": False,
                        }

                        matches_data.append(match_record)

        # Convert to polars DataFrame
        matches_df = pl.DataFrame(matches_data)

        # Create players data for all teams
        players_data = []
        for tournament in circuit_results.tournaments:
            for team in tournament.all_teams:
                for player in team.players:
                    players_data.append(
                        {
                            "tournament_id": tournament.tournament_id,
                            "team_id": team.team_id,
                            "user_id": player.user_id,
                        }
                    )

        self._players_df = pl.DataFrame(players_data)

        return matches_df

    def _get_active_players(self, circuit_results: CircuitResults) -> List[int]:
        """Get players who meet minimum participation threshold."""
        active = []
        for (
            player_id,
            tournaments,
        ) in circuit_results.player_participation.items():
            if len(tournaments) >= self.min_tournaments:
                active.append(player_id)
        return active

    def _calculate_true_rankings(
        self,
        all_players: List[SyntheticPlayer],
        active_player_ids: List[int],
    ) -> Dict[int, int]:
        """Calculate true rankings based on player skills."""
        # Filter to active players
        active_players = [
            p for p in all_players if p.user_id in active_player_ids
        ]

        # Sort by true skill (descending)
        sorted_players = sorted(
            active_players, key=lambda p: p.true_skill, reverse=True
        )

        # Create rank mapping
        true_rankings = {}
        for rank, player in enumerate(sorted_players, 1):
            true_rankings[player.user_id] = rank

        return true_rankings

    def _compute_metrics(
        self,
        rankings: pl.DataFrame,
        true_rankings: Dict[int, int],
        circuit_results: CircuitResults,
        active_players: List[int],
        player_pool: List[SyntheticPlayer],
    ) -> PageRankEvaluation:
        """Compute evaluation metrics."""
        # Extract PageRank results for active players
        # Note: player_rank column contains PageRank SCORES, not rank positions
        pr_rankings = rankings.filter(pl.col("id").is_in(active_players)).sort(
            "player_rank", descending=True
        )  # Sort by score descending to assign ranks

        # Create rank mappings
        pr_rank_map = {}
        pr_score_map = {}

        # Assign actual rank positions based on sorted PageRank scores
        for rank_position, row in enumerate(
            pr_rankings.iter_rows(named=True), 1
        ):
            player_id = row["id"]
            pagerank_score = row[
                "player_rank"
            ]  # This is actually the PageRank score
            pr_rank_map[
                player_id
            ] = rank_position  # Actual rank position (1, 2, 3, ...)
            pr_score_map[player_id] = pagerank_score  # PageRank score

        # Align rankings for comparison
        player_ids = list(set(pr_rank_map.keys()) & set(true_rankings.keys()))
        pr_ranks = [pr_rank_map[pid] for pid in player_ids]
        true_ranks = [true_rankings[pid] for pid in player_ids]

        # Correlation metrics
        if len(player_ids) > 1:
            spearman_corr, _ = stats.spearmanr(true_ranks, pr_ranks)
            kendall_tau, _ = stats.kendalltau(true_ranks, pr_ranks)

            # Weighted Spearman - emphasizes top ranks
            weighted_spearman_corr = weighted_spearman(
                true_ranks,
                pr_ranks,
                weight_type="exponential",
                alpha=0.05,  # Moderate decay - top ranks ~20x more important than bottom
            )

            # For Pearson, use actual scores vs true skills
            player_skills = {
                p.user_id: p.true_skill
                for p in player_pool
                if p.user_id in player_ids
            }
            pr_scores_aligned = [pr_score_map[pid] for pid in player_ids]
            true_skills_aligned = [player_skills[pid] for pid in player_ids]
            pearson_corr, _ = stats.pearsonr(
                true_skills_aligned, pr_scores_aligned
            )
        else:
            spearman_corr = (
                kendall_tau
            ) = pearson_corr = weighted_spearman_corr = 0.0

        # Top-K accuracy
        def top_k_accuracy(k: int) -> float:
            if len(player_ids) < k:
                return 0.0

            true_top_k = set(
                pid for pid, rank in true_rankings.items() if rank <= k
            )
            pr_top_k = set(
                pid for pid, rank in pr_rank_map.items() if rank <= k
            )

            return len(true_top_k & pr_top_k) / k

        top_10_acc = top_k_accuracy(10)
        top_20_acc = top_k_accuracy(20)
        top_50_acc = top_k_accuracy(50)

        # Weighted accuracy metrics
        weighted_acc = top_k_weighted_accuracy(
            true_ranks, pr_ranks, [10, 20, 50]
        )
        top_10_weighted = weighted_acc.get("top_10_weighted", 0.0)
        top_20_weighted = weighted_acc.get("top_20_weighted", 0.0)
        top_50_weighted = weighted_acc.get("top_50_weighted", 0.0)

        # Rank error metrics
        rank_errors = [
            abs(pr_rank_map[pid] - true_rankings[pid]) for pid in player_ids
        ]

        if rank_errors:
            mean_rank_error = np.mean(rank_errors)
            median_rank_error = np.median(rank_errors)
            rmse_rank = np.sqrt(np.mean(np.square(rank_errors)))

            # Top 10 specific error
            top_10_errors = [
                abs(pr_rank_map[pid] - true_rankings[pid])
                for pid in player_ids
                if true_rankings[pid] <= 10
            ]
            top_10_mean_error = np.mean(top_10_errors) if top_10_errors else 0.0
        else:
            mean_rank_error = (
                median_rank_error
            ) = rmse_rank = top_10_mean_error = 0.0

        # PageRank convergence info (would need to be extracted from engine)
        n_iterations = 100  # Placeholder
        convergence_error = 1e-8  # Placeholder

        # Participation counts
        participation_counts = {
            pid: len(tournaments)
            for pid, tournaments in circuit_results.player_participation.items()
            if pid in active_players
        }

        return PageRankEvaluation(
            spearman_correlation=spearman_corr,
            weighted_spearman=weighted_spearman_corr,
            kendall_tau=kendall_tau,
            pearson_correlation=pearson_corr,
            top_10_accuracy=top_10_acc,
            top_20_accuracy=top_20_acc,
            top_50_accuracy=top_50_acc,
            top_10_weighted_accuracy=top_10_weighted,
            top_20_weighted_accuracy=top_20_weighted,
            top_50_weighted_accuracy=top_50_weighted,
            mean_rank_error=mean_rank_error,
            median_rank_error=median_rank_error,
            rmse_rank=rmse_rank,
            top_10_mean_error=top_10_mean_error,
            n_iterations=n_iterations,
            convergence_error=convergence_error,
            rank_by_player_id=pr_rank_map,
            true_rank_by_player_id=true_rankings,
            pagerank_scores=pr_score_map,
            participation_counts=participation_counts,
        )

    def evaluate_parameter_sensitivity(
        self,
        circuit: TournamentCircuit,
        circuit_results: CircuitResults,
        parameter_ranges: Dict[str, List[float]],
    ) -> Dict[Tuple[str, float], PageRankEvaluation]:
        """
        Evaluate PageRank across different parameter settings.

        Parameters
        ----------
        circuit : TournamentCircuit
            Tournament circuit
        circuit_results : CircuitResults
            Circuit results
        parameter_ranges : Dict[str, List[float]]
            Parameter names and values to test

        Returns
        -------
        Dict[Tuple[str, float], PageRankEvaluation]
            Evaluations for each parameter setting
        """
        results = {}

        # Save original parameters
        orig_damping = self.damping_factor
        orig_decay = self.decay_rate
        orig_beta = self.beta

        for param_name, values in parameter_ranges.items():
            for value in values:
                # Set parameter - recreate engine with new params
                if param_name == "damping_factor":
                    self.damping_factor = value
                    self.engine = RatingEngine(
                        damping_factor=value,
                        decay_half_life_days=30.0
                        if self.decay_rate == DEFAULT_DECAY_RATE
                        else np.log(2) / self.decay_rate,
                        beta=self.beta,
                    )
                elif param_name == "decay_rate":
                    self.decay_rate = value
                    self.engine = RatingEngine(
                        damping_factor=self.damping_factor,
                        decay_half_life_days=30.0
                        if value == DEFAULT_DECAY_RATE
                        else np.log(2) / value,
                        beta=self.beta,
                    )
                elif param_name == "beta":
                    self.beta = value
                    self.engine = RatingEngine(
                        damping_factor=self.damping_factor,
                        decay_half_life_days=30.0
                        if self.decay_rate == DEFAULT_DECAY_RATE
                        else np.log(2) / self.decay_rate,
                        beta=value,
                    )

                # Evaluate
                eval_result = self.evaluate_circuit(circuit, circuit_results)
                results[(param_name, value)] = eval_result

        # Restore original parameters
        self.damping_factor = orig_damping
        self.decay_rate = orig_decay
        self.beta = orig_beta
        self.engine = RatingEngine(
            damping_factor=orig_damping,
            decay_half_life_days=30.0
            if orig_decay == DEFAULT_DECAY_RATE
            else np.log(2) / orig_decay,
            beta=orig_beta,
        )

        return results
