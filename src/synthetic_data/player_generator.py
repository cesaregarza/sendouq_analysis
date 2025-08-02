"""
Player generation module for synthetic tournament data.

This module provides functionality to generate synthetic players with
configurable skill distributions for tournament simulations.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SyntheticPlayer:
    """Represents a synthetic player with skill attributes."""

    user_id: int
    username: str
    true_skill: float  # Internal skill rating for simulation
    skill_variance: float  # Performance variance
    team_affinity: Optional[str] = None  # Preferred team composition style

    def get_performance(
        self, rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Get player's performance for a match.

        Performance varies around true skill based on variance.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator for reproducibility

        Returns
        -------
        float
            Performance value for this match
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(self.true_skill, self.skill_variance)


class PlayerGenerator:
    """Generates synthetic players with configurable skill distributions."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the player generator.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._next_user_id = 1

    def generate_players(
        self,
        n_players: int,
        skill_distribution: str = "normal",
        skill_params: Optional[Dict] = None,
        variance_range: Tuple[float, float] = (0.1, 0.3),
    ) -> List[SyntheticPlayer]:
        """
        Generate a set of synthetic players.

        Parameters
        ----------
        n_players : int
            Number of players to generate
        skill_distribution : str
            Type of skill distribution: "normal", "uniform", "bimodal", "exponential"
        skill_params : dict, optional
            Parameters for the skill distribution
        variance_range : tuple
            Range for player performance variance

        Returns
        -------
        List[SyntheticPlayer]
            List of generated synthetic players
        """
        if skill_params is None:
            skill_params = self._get_default_params(skill_distribution)

        skills = self._generate_skills(
            n_players, skill_distribution, skill_params
        )
        players = []

        for i in range(n_players):
            player = SyntheticPlayer(
                user_id=self._next_user_id,
                username=f"player_{self._next_user_id}",
                true_skill=skills[i],
                skill_variance=self.rng.uniform(*variance_range),
                team_affinity=self._generate_team_affinity(),
            )
            players.append(player)
            self._next_user_id += 1

        return players

    def generate_elite_players(
        self, n_players: int, base_skill: float = 2.0
    ) -> List[SyntheticPlayer]:
        """
        Generate elite players with high skill levels.

        Parameters
        ----------
        n_players : int
            Number of elite players to generate
        base_skill : float
            Minimum skill level for elite players

        Returns
        -------
        List[SyntheticPlayer]
            List of elite synthetic players
        """
        skills = self.rng.normal(base_skill + 0.5, 0.2, n_players)
        skills = np.clip(skills, base_skill, None)  # Ensure minimum skill

        players = []
        for i in range(n_players):
            player = SyntheticPlayer(
                user_id=self._next_user_id,
                username=f"elite_{self._next_user_id}",
                true_skill=skills[i],
                skill_variance=self.rng.uniform(
                    0.05, 0.15
                ),  # Lower variance for elite
                team_affinity=self._generate_team_affinity(),
            )
            players.append(player)
            self._next_user_id += 1

        return players

    def _generate_skills(
        self, n_players: int, distribution: str, params: Dict
    ) -> np.ndarray:
        """Generate skill values based on specified distribution."""
        if distribution == "normal":
            return self.rng.normal(params["mean"], params["std"], n_players)
        elif distribution == "uniform":
            return self.rng.uniform(params["low"], params["high"], n_players)
        elif distribution == "bimodal":
            # Mix of two normal distributions
            n1 = n_players // 2
            n2 = n_players - n1
            skills1 = self.rng.normal(params["mean1"], params["std1"], n1)
            skills2 = self.rng.normal(params["mean2"], params["std2"], n2)
            return np.concatenate([skills1, skills2])
        elif distribution == "exponential":
            # Exponential decay from high skill
            return params["high"] - self.rng.exponential(
                params["scale"], n_players
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def _get_default_params(self, distribution: str) -> Dict:
        """Get default parameters for skill distributions."""
        defaults = {
            "normal": {"mean": 0.0, "std": 1.0},
            "uniform": {"low": -2.0, "high": 2.0},
            "bimodal": {"mean1": -0.5, "std1": 0.5, "mean2": 1.0, "std2": 0.5},
            "exponential": {"high": 2.0, "scale": 1.0},
        }
        return defaults.get(distribution, {})

    def _generate_team_affinity(self) -> str:
        """Generate random team affinity type."""
        affinities = ["aggressive", "defensive", "balanced", "specialist"]
        return self.rng.choice(affinities)

    def create_player_pool_with_categories(
        self, n_elite: int = 50, n_competitive: int = 200, n_casual: int = 500
    ) -> Dict[str, List[SyntheticPlayer]]:
        """
        Create a realistic player pool with different skill categories.

        Parameters
        ----------
        n_elite : int
            Number of elite players
        n_competitive : int
            Number of competitive players
        n_casual : int
            Number of casual players

        Returns
        -------
        Dict[str, List[SyntheticPlayer]]
            Dictionary mapping category to player list
        """
        player_pool = {
            "elite": self.generate_players(
                n_elite,
                skill_distribution="normal",
                skill_params={"mean": 2.0, "std": 0.3},
                variance_range=(0.05, 0.15),
            ),
            "competitive": self.generate_players(
                n_competitive,
                skill_distribution="normal",
                skill_params={"mean": 0.5, "std": 0.5},
                variance_range=(0.1, 0.25),
            ),
            "casual": self.generate_players(
                n_casual,
                skill_distribution="normal",
                skill_params={"mean": -0.5, "std": 0.7},
                variance_range=(0.2, 0.4),
            ),
        }

        return player_pool
