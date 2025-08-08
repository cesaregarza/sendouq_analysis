"""
Skill drift module implementing Ornstein-Uhlenbeck process.

This module provides functionality to simulate realistic skill evolution
over time using the Ornstein-Uhlenbeck process, which models mean-reverting
random walks suitable for player skill dynamics.
"""

from typing import Optional, Union

import numpy as np


class OrnsteinUhlenbeckDrift:
    """
    Implements Ornstein-Uhlenbeck process for skill drift.

    The OU process models skills that drift randomly but are pulled back
    toward a long-term mean, preventing unrealistic divergence while still
    allowing for temporal variation.

    Parameters
    ----------
    mu : float
        Long-term mean skill level
    half_life_days : float
        Half-life for mean reversion in days (should match PageRank decay)
    sigma : float
        Volatility parameter controlling drift magnitude
    dt_days : float
        Time step size in days
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        mu: float = 0.0,
        half_life_days: float = 28.0,
        sigma: float = 0.08,
        dt_days: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.mu = mu
        self.half_life_days = half_life_days
        self.sigma = sigma
        self.dt_days = dt_days
        self.rng = np.random.default_rng(seed)

        # Calculate mean reversion rate from half-life
        # lambda such that (1 - lambda)^(HL/dt) = 0.5
        self.lambda_ = 1 - 2 ** (-dt_days / half_life_days)

    def update_skills(
        self,
        skills: Union[np.ndarray, list],
        dt_days: Optional[float] = None,
    ) -> np.ndarray:
        """
        Update skills using Ornstein-Uhlenbeck process.

        The update follows:
        s_new = mu + (1 - lambda) * (s_old - mu) + sigma * sqrt(dt) * N(0,1)

        Parameters
        ----------
        skills : array-like
            Current skill values
        dt_days : float, optional
            Time step override (uses instance default if None)

        Returns
        -------
        np.ndarray
            Updated skill values
        """
        skills = np.asarray(skills)
        dt = dt_days if dt_days is not None else self.dt_days

        # Recalculate lambda if dt changed
        if dt != self.dt_days:
            lambda_ = 1 - 2 ** (-dt / self.half_life_days)
        else:
            lambda_ = self.lambda_

        # OU update: pull toward mean + random drift
        mean_reversion = self.mu + (1 - lambda_) * (skills - self.mu)
        random_drift = (
            self.sigma * np.sqrt(dt) * self.rng.normal(size=skills.shape)
        )

        return mean_reversion + random_drift

    def stationary_std(self) -> float:
        """
        Calculate the stationary standard deviation of the OU process.

        At equilibrium, the variance is sigma^2 * dt / (2 * lambda)

        Returns
        -------
        float
            Standard deviation at stationarity
        """
        return self.sigma * np.sqrt(self.dt_days / (2 * self.lambda_))

    def alpha_for_target_prob(
        self,
        p_star: float = 0.76,
        skill_std_multiplier: float = np.sqrt(2),
    ) -> float:
        """
        Calculate Bradley-Terry alpha for target win probability.

        Given a target probability p* for a 1 SD skill difference,
        calculates the appropriate alpha parameter.

        Parameters
        ----------
        p_star : float
            Target win probability for 1 SD skill difference
        skill_std_multiplier : float
            Multiplier for skill std to get typical difference (default sqrt(2))

        Returns
        -------
        float
            Bradley-Terry alpha parameter
        """
        delta = skill_std_multiplier * self.stationary_std()
        return np.log(p_star / (1 - p_star)) / delta

    def simulate_trajectory(
        self,
        initial_skills: Union[np.ndarray, list],
        n_steps: int,
        dt_days: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simulate a full skill trajectory over time.

        Parameters
        ----------
        initial_skills : array-like
            Starting skill values
        n_steps : int
            Number of time steps to simulate
        dt_days : float, optional
            Time step size (uses instance default if None)

        Returns
        -------
        np.ndarray
            Shape (n_steps + 1, n_players) with skill trajectories
        """
        skills = np.asarray(initial_skills)
        n_players = len(skills)
        trajectory = np.zeros((n_steps + 1, n_players))
        trajectory[0] = skills

        for t in range(n_steps):
            skills = self.update_skills(skills, dt_days)
            trajectory[t + 1] = skills

        return trajectory
