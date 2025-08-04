"""
Improved tournament circuit with realistic skill-based match outcomes.

This module fixes the circular dependency issue by ensuring tournaments
have realistic skill distributions and match outcomes.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl

from synthetic_data.circuits.tournament_circuit import (
    CircuitResults,
    TournamentCircuit,
    TournamentConfig,
)
from synthetic_data.configs.realistic_tournament_config import (
    RealisticCircuitConfig,
    create_realistic_skill_distribution,
)
from synthetic_data.core.player_generator import PlayerGenerator


class RealisticTournamentCircuit(TournamentCircuit):
    """
    Tournament circuit that generates realistic patterns.

    Key improvements:
    1. Skill-based match outcomes (no artificial cycles)
    2. Realistic tournament participant selection
    3. Proper skill distribution in player pool
    4. Tournament quality varies based on participant skills
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        config: Optional[RealisticCircuitConfig] = None,
    ):
        """
        Initialize realistic tournament circuit.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        config : RealisticCircuitConfig, optional
            Circuit configuration
        """
        if config is None:
            config = RealisticCircuitConfig()

        self.config = config

        # Initialize base circuit with lognormal skill distribution
        super().__init__(
            seed=seed,
            player_pool_size=config.player_pool_size,
            skill_distribution=config.skill_distribution,
            skill_params=config.skill_params,
        )

        # Override player pool with realistic distribution
        self._create_realistic_player_pool()

    def _create_realistic_player_pool(self):
        """Create player pool with realistic skill distribution."""
        # Generate realistic skill values
        skills = create_realistic_skill_distribution(
            self.config.player_pool_size,
            self.config.skill_distribution,
            self.seed,
        )

        # Create players with these skills
        self.player_pool = []
        for i, skill in enumerate(skills):
            player = self.player_gen.generate_players(
                1,
                skill_distribution="uniform",
                skill_params={"low": skill, "high": skill},
            )[0]
            player.user_id = i + 1
            player.true_skill = skill
            self.player_pool.append(player)

        # Sort by skill for easier selection
        self.player_pool.sort(key=lambda p: p.true_skill, reverse=True)

        # Store skill array for percentile calculations
        self.player_skills = np.array([p.true_skill for p in self.player_pool])

    def generate_realistic_circuit(
        self,
        n_tournaments: Optional[int] = None,
        start_date: Optional[datetime] = None,
    ) -> CircuitResults:
        """
        Generate a realistic tournament circuit.

        Parameters
        ----------
        n_tournaments : int, optional
            Number of tournaments (uses config default if not specified)
        start_date : datetime, optional
            Circuit start date

        Returns
        -------
        CircuitResults
            Results of the circuit simulation
        """
        if n_tournaments is None:
            n_tournaments = self.config.n_tournaments

        if start_date is None:
            start_date = self.config.start_date or datetime.now()

        # Update config with actual tournament count
        self.config.n_tournaments = n_tournaments

        # Generate tournament configurations based on actual skill distribution
        configs = self.config.generate_tournament_configs(self.player_skills)[
            :n_tournaments
        ]

        # Run the circuit with realistic configurations
        return self.generate_circuit(configs, start_date)

    def _select_participants(self, config: TournamentConfig) -> list:
        """
        Select tournament participants with realistic patterns.

        Overrides base method to ensure proper skill-based selection.
        """
        eligible_players = self.player_pool.copy()

        # Apply skill restrictions
        if config.skill_floor is not None:
            eligible_players = [
                p
                for p in eligible_players
                if p.true_skill >= config.skill_floor
            ]

        if config.skill_cap is not None:
            eligible_players = [
                p for p in eligible_players if p.true_skill <= config.skill_cap
            ]

        if len(eligible_players) < config.n_teams * config.team_size:
            # Not enough eligible players, expand pool slightly
            if config.skill_floor is not None:
                # Lower the floor
                expanded_floor = config.skill_floor * 0.9
                eligible_players = [
                    p
                    for p in self.player_pool
                    if p.true_skill >= expanded_floor
                ]
            elif config.skill_cap is not None:
                # Raise the cap
                expanded_cap = config.skill_cap * 1.1
                eligible_players = [
                    p for p in self.player_pool if p.true_skill <= expanded_cap
                ]

        # Apply selection bias
        n_needed = min(config.n_teams * config.team_size, len(eligible_players))

        if config.selection_bias > 0 and len(eligible_players) > n_needed:
            # Bias towards higher skilled players
            # Sort by skill (already sorted in pool)
            eligible_sorted = sorted(
                eligible_players, key=lambda p: p.true_skill, reverse=True
            )

            # Use weighted selection
            weights = np.array(
                [
                    (1 + config.selection_bias) ** i
                    for i in range(len(eligible_sorted), 0, -1)
                ]
            )
            weights = weights / weights.sum()

            selected_indices = self.rng.choice(
                len(eligible_sorted), size=n_needed, replace=False, p=weights
            )

            return [eligible_sorted[i] for i in selected_indices]
        else:
            # Random selection
            return self.rng.choice(
                eligible_players, size=n_needed, replace=False
            ).tolist()


def analyze_circuit_quality(circuit_results: CircuitResults) -> pl.DataFrame:
    """
    Analyze the quality distribution of generated tournaments.

    Parameters
    ----------
    circuit_results : CircuitResults
        Results from circuit generation

    Returns
    -------
    pl.DataFrame
        Analysis of tournament qualities
    """
    tournament_data = []

    for tournament in circuit_results.tournaments:
        # Get all participants
        participants = []
        seen_ids = set()
        for team in tournament.all_teams:
            for player in team.players:
                if player.user_id not in seen_ids:
                    participants.append(player)
                    seen_ids.add(player.user_id)

        # Calculate tournament quality metrics
        skills = [p.true_skill for p in participants]

        tournament_data.append(
            {
                "tournament_id": tournament.tournament_id,
                "tournament_name": tournament.name,
                "n_participants": len(participants),
                "avg_skill": np.mean(skills) if skills else 0,
                "min_skill": np.min(skills) if skills else 0,
                "max_skill": np.max(skills) if skills else 0,
                "skill_std": np.std(skills) if skills else 0,
                "skill_range": np.max(skills) - np.min(skills) if skills else 0,
            }
        )

    return pl.DataFrame(tournament_data)


def validate_tournament_quality_differences(
    circuit_results: CircuitResults, min_quality_ratio: float = 1.5
) -> bool:
    """
    Validate that tournaments have sufficient quality differences.

    Parameters
    ----------
    circuit_results : CircuitResults
        Circuit results to validate
    min_quality_ratio : float
        Minimum ratio between highest and lowest quality tournaments

    Returns
    -------
    bool
        True if quality differences are sufficient
    """
    analysis = analyze_circuit_quality(circuit_results)

    if len(analysis) < 2:
        return False

    max_avg_skill = analysis["avg_skill"].max()
    min_avg_skill = analysis["avg_skill"].min()

    if min_avg_skill <= 0:
        return False

    quality_ratio = max_avg_skill / min_avg_skill

    print(f"Tournament quality ratio: {quality_ratio:.2f}")
    print(f"  Highest avg skill: {max_avg_skill:.2f}")
    print(f"  Lowest avg skill: {min_avg_skill:.2f}")

    return quality_ratio >= min_quality_ratio
