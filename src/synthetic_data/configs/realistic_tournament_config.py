"""
Realistic tournament configuration for synthetic data generation.

This module provides configurations that create realistic tournament patterns
with proper skill distributions and tournament types.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from synthetic_data.circuits.tournament_circuit import (
    TournamentConfig,
    TournamentType,
)
from synthetic_data.core.tournament_generator import TournamentFormat


@dataclass
class RealisticCircuitConfig:
    """Configuration for generating realistic tournament circuits."""

    # Player pool configuration
    player_pool_size: int = 500
    skill_distribution: str = (
        "lognormal"  # More realistic - most players clustered, few elite
    )
    skill_params: dict = None

    # Tournament types with realistic skill distributions
    elite_percentile: float = 0.85  # Top 15% of players for elite tournaments
    amateur_percentile: float = 0.40  # Bottom 40% for amateur tournaments

    # Circuit configuration
    n_tournaments: int = 20
    tournament_interval_days: float = 7.0
    start_date: Optional[datetime] = None

    def __post_init__(self):
        """Set up default skill parameters if not provided."""
        if self.skill_params is None:
            # Lognormal creates realistic distribution:
            # - Most players clustered around average
            # - Long tail of elite players
            # - Few very weak players
            self.skill_params = {
                "mean": 0.0,
                "sigma": 0.5,
                "scale": 1.0,
                "shift": 0.0,
            }

    def generate_tournament_configs(
        self, player_skills: np.ndarray
    ) -> list[TournamentConfig]:
        """
        Generate tournament configurations based on actual player skill distribution.

        Parameters
        ----------
        player_skills : np.ndarray
            Array of player skill values from the generated pool

        Returns
        -------
        list[TournamentConfig]
            List of tournament configurations
        """
        # Calculate skill thresholds based on actual distribution
        skill_floor_elite = np.percentile(
            player_skills, self.elite_percentile * 100
        )
        skill_cap_amateur = np.percentile(
            player_skills, self.amateur_percentile * 100
        )
        skill_median = np.median(player_skills)

        configs = []

        for i in range(self.n_tournaments):
            offset_days = i * self.tournament_interval_days

            # Create varied tournament types with realistic patterns
            if i % 7 == 0:
                # Major Championship - Elite double elimination
                config = TournamentConfig(
                    name=f"Major_Championship_{i+1}",
                    tournament_type=TournamentType.INVITATIONAL,
                    format=TournamentFormat.DOUBLE_ELIMINATION,
                    n_teams=16,
                    skill_floor=skill_floor_elite,
                    selection_bias=0.8,  # Strong bias towards highest skilled
                    seeded_bracket=True,
                    start_offset_days=offset_days,
                )
            elif i % 5 == 1:
                # Premier League - High level Swiss
                config = TournamentConfig(
                    name=f"Premier_League_{i+1}",
                    tournament_type=TournamentType.INVITATIONAL,
                    format=TournamentFormat.SWISS,
                    n_teams=24,
                    skill_floor=skill_median * 1.2,  # Above average players
                    swiss_rounds=7,
                    selection_bias=0.6,
                    start_offset_days=offset_days,
                )
            elif i % 4 == 2:
                # Amateur Cup - Lower skill bracket
                config = TournamentConfig(
                    name=f"Amateur_Cup_{i+1}",
                    tournament_type=TournamentType.SKILL_CAPPED,
                    format=TournamentFormat.SINGLE_ELIMINATION,
                    n_teams=32,
                    skill_cap=skill_cap_amateur,
                    selection_bias=0.0,  # Random selection within skill range
                    seeded_bracket=False,  # No seeding for amateur
                    start_offset_days=offset_days,
                )
            elif i % 3 == 0:
                # Open Qualifier - Mixed skills with slight upper bias
                config = TournamentConfig(
                    name=f"Open_Qualifier_{i+1}",
                    tournament_type=TournamentType.OPEN,
                    format=TournamentFormat.SWISS,
                    n_teams=48,
                    selection_bias=0.3,  # Slight bias to better players
                    swiss_rounds=6,
                    start_offset_days=offset_days,
                )
            else:
                # Regional Tournament - Mixed single elimination
                config = TournamentConfig(
                    name=f"Regional_Tournament_{i+1}",
                    tournament_type=TournamentType.MIXED,
                    format=TournamentFormat.SINGLE_ELIMINATION,
                    n_teams=16,
                    skill_floor=skill_median * 0.5,  # Some minimum skill
                    skill_cap=skill_median * 1.5,  # But not elite
                    selection_bias=0.2,
                    seeded_bracket=True,
                    start_offset_days=offset_days,
                )

            configs.append(config)

        return configs


def create_realistic_skill_distribution(
    n_players: int, distribution: str = "lognormal", seed: Optional[int] = None
) -> np.ndarray:
    """
    Create a realistic skill distribution for players.

    Parameters
    ----------
    n_players : int
        Number of players
    distribution : str
        Type of distribution ("lognormal", "normal", "bimodal")
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Array of skill values
    """
    rng = np.random.default_rng(seed)

    if distribution == "lognormal":
        # Lognormal: Most players average, few elite, very few terrible
        # This matches real competitive gaming populations
        base_skills = rng.lognormal(mean=0.0, sigma=0.5, size=n_players)
        # Shift and scale to reasonable range
        skills = (base_skills - base_skills.min()) * 2.0

    elif distribution == "bimodal":
        # Bimodal: Casual and competitive player populations
        n_casual = int(n_players * 0.6)
        n_competitive = n_players - n_casual

        casual_skills = rng.normal(0.0, 0.5, size=n_casual)
        competitive_skills = rng.normal(1.5, 0.4, size=n_competitive)
        skills = np.concatenate([casual_skills, competitive_skills])
        rng.shuffle(skills)

    else:  # normal
        # Standard normal distribution
        skills = rng.normal(1.0, 0.5, size=n_players)

    return skills


def calculate_expected_tournament_strength(
    tournament_participants: list[float], method: str = "top_20_sum"
) -> float:
    """
    Calculate expected tournament strength from participant skills.

    Parameters
    ----------
    tournament_participants : list[float]
        Skill values of tournament participants
    method : str
        Aggregation method matching RatingEngine

    Returns
    -------
    float
        Expected tournament strength
    """
    if not tournament_participants:
        return 0.0

    participants = np.array(tournament_participants)

    if method == "mean":
        return participants.mean()
    elif method == "median":
        return np.median(participants)
    elif method == "sum":
        return participants.sum()
    elif method == "top_20_sum":
        top_20 = np.sort(participants)[-20:]
        return top_20.sum()
    else:
        return participants.mean()
