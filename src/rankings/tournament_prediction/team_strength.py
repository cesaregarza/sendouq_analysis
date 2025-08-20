"""Team strength calculation from individual player ratings."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from scipy.special import logsumexp


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
