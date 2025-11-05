"""
Team seeding from model outputs.

This module provides a simple interface to convert player rankings from the model
into team strength scores for tournament seeding.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from rankings.tournament_prediction.team_strength import (
    TeamRatingConfig,
    TeamStrengthCalculator,
)


@dataclass
class TeamSeedingConfig:
    """Configuration for team seeding from model outputs."""

    # Team strength calculation
    alpha: float = 1.0  # Diminishing returns (1.0 = no diminishing returns)
    use_top_k: int | None = 4  # Only use top-k players per team
    default_exposure: float = 1.0  # Default exposure weight

    # Input column names
    player_id_col: str = "id"
    score_col: str = "score"
    exposure_col: str | None = "exposure"


class TeamSeeding:
    """Convert model outputs to team strength scores."""

    def __init__(self, config: TeamSeedingConfig | None = None):
        """Initialize team seeding with configuration.

        Args:
            config: Configuration for team seeding (uses defaults if None)
        """
        self.config = config or TeamSeedingConfig()

        # Initialize team strength calculator
        team_config = TeamRatingConfig(
            alpha=self.config.alpha,
            use_top_k=self.config.use_top_k,
            default_weight=self.config.default_exposure,
        )
        self.calculator = TeamStrengthCalculator(team_config)

    def compute_team_strength(
        self,
        player_ratings: pl.DataFrame,
        team_roster: list[int],
        exposure_weights: dict[int, float] | None = None,
    ) -> float:
        """Compute strength score for a single team.

        Args:
            player_ratings: DataFrame with player_id and score columns
            team_roster: List of player IDs on the team
            exposure_weights: Optional exposure weights per player

        Returns:
            Team strength score (log-scale)
        """
        # Convert DataFrame to dict
        ratings_dict = self._extract_ratings(player_ratings)

        # Calculate team log-rating
        return self.calculator.team_log_rating(
            ratings_dict, team_roster, exposure_weights
        )

    def compute_all_teams(
        self,
        player_ratings: pl.DataFrame,
        teams: dict[int, list[int]],
        exposure_weights: dict[tuple[int, int], float] | None = None,
    ) -> pl.DataFrame:
        """Compute strength scores for all teams.

        Args:
            player_ratings: DataFrame with player rankings from model
            teams: Dictionary mapping team_id -> list of player_ids
            exposure_weights: Optional weights as (player_id, team_id) -> weight

        Returns:
            DataFrame with columns: team_id, strength, seed_rank
        """
        ratings_dict = self._extract_ratings(player_ratings)

        results = []
        for team_id, roster in teams.items():
            # Get exposure weights for this team
            team_weights = None
            if exposure_weights:
                team_weights = {
                    p: exposure_weights.get((p, team_id), self.config.default_exposure)
                    for p in roster
                }

            # Compute team strength
            strength = self.calculator.team_log_rating(
                ratings_dict, roster, team_weights
            )

            results.append({"team_id": team_id, "strength": strength})

        # Create DataFrame and assign seed ranks
        df = pl.DataFrame(results)
        df = df.sort("strength", descending=True)
        df = df.with_columns(
            pl.lit(range(1, len(df) + 1)).alias("seed_rank")
        )

        return df

    def compute_from_dataframe(
        self,
        player_ratings: pl.DataFrame,
        team_rosters: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute team strengths from DataFrames.

        Args:
            player_ratings: DataFrame with columns: id, score, (optional) exposure
            team_rosters: DataFrame with columns: team_id, player_id, (optional) exposure

        Returns:
            DataFrame with columns: team_id, strength, seed_rank
        """
        # Extract player ratings
        ratings_dict = self._extract_ratings(player_ratings)

        # Group rosters by team
        teams_grouped = team_rosters.group_by("team_id").agg(
            pl.col("player_id").alias("roster"),
            (
                pl.col("exposure").alias("weights")
                if "exposure" in team_rosters.columns
                else pl.lit(None).alias("weights")
            ),
        )

        results = []
        for row in teams_grouped.iter_rows(named=True):
            team_id = row["team_id"]
            roster = row["roster"]
            weights_list = row.get("weights")

            # Build exposure weights dict if available
            team_weights = None
            if weights_list and any(w is not None for w in weights_list):
                team_weights = dict(zip(roster, weights_list))

            # Compute strength
            strength = self.calculator.team_log_rating(
                ratings_dict, roster, team_weights
            )

            results.append({"team_id": team_id, "strength": strength})

        # Create DataFrame and assign seeds
        df = pl.DataFrame(results)
        df = df.sort("strength", descending=True)
        df = df.with_columns(
            pl.lit(range(1, len(df) + 1)).alias("seed_rank")
        )

        return df

    def _extract_ratings(self, player_ratings: pl.DataFrame) -> dict[int, float]:
        """Extract player ratings as dictionary.

        Args:
            player_ratings: DataFrame with player rankings

        Returns:
            Dictionary mapping player_id -> score
        """
        id_col = self.config.player_id_col
        score_col = self.config.score_col

        # Validate columns exist
        if id_col not in player_ratings.columns:
            raise ValueError(f"Column '{id_col}' not found in player_ratings")
        if score_col not in player_ratings.columns:
            raise ValueError(f"Column '{score_col}' not found in player_ratings")

        # Convert to dict
        return dict(
            zip(
                player_ratings[id_col].to_list(),
                player_ratings[score_col].to_list(),
            )
        )


# Convenience function for simple cases
def seed_teams(
    player_ratings: pl.DataFrame,
    teams: dict[int, list[int]],
    alpha: float = 1.0,
    use_top_k: int | None = 4,
) -> pl.DataFrame:
    """Simple function to seed teams from player ratings.

    Args:
        player_ratings: DataFrame with 'id' and 'score' columns from model
        teams: Dictionary mapping team_id -> list of player_ids
        alpha: Diminishing returns factor (1.0 = no diminishing returns)
        use_top_k: Only use top-k players per team (None = use all)

    Returns:
        DataFrame with columns: team_id, strength, seed_rank

    Example:
        >>> player_ratings = pl.DataFrame({
        ...     "id": [1, 2, 3, 4, 5, 6, 7, 8],
        ...     "score": [2.5, 2.1, 1.8, 1.5, 1.2, 1.0, 0.8, 0.5]
        ... })
        >>> teams = {
        ...     101: [1, 2, 3, 4],
        ...     102: [5, 6, 7, 8]
        ... }
        >>> seeds = seed_teams(player_ratings, teams)
        >>> print(seeds)
        ┌─────────┬──────────┬───────────┐
        │ team_id ┆ strength ┆ seed_rank │
        ├─────────┼──────────┼───────────┤
        │ 101     ┆ 3.45     ┆ 1         │
        │ 102     ┆ 2.12     ┆ 2         │
        └─────────┴──────────┴───────────┘
    """
    config = TeamSeedingConfig(alpha=alpha, use_top_k=use_top_k)
    seeder = TeamSeeding(config)
    return seeder.compute_all_teams(player_ratings, teams)
