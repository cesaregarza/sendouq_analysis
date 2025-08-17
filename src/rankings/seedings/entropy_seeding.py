"""
Entropy-controlled division assignment system.

This module implements a parameter-free approach to team division assignment
using Shannon entropy to measure team balance and adjust ratings accordingly.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


def compute_team_entropy(
    skill_values: List[float], exposure_weights: Optional[List[float]] = None
) -> Tuple[float, float, np.ndarray]:
    """
    Compute Shannon entropy for a team's skill distribution.

    Args:
        skill_values: List of player skill values (e.g., log-odds ratings)
        exposure_weights: Optional exposure weights for each player

    Returns:
        Tuple of (entropy, normalized_lambda, probability_distribution)
    """
    if len(skill_values) == 0:
        raise ValueError("Team must have at least one player")

    # Convert to numpy array
    skills = np.array(skill_values)

    # Apply exposure weights if provided
    if exposure_weights is not None:
        weights = np.array(exposure_weights)
        if len(weights) != len(skills):
            raise ValueError("Exposure weights must match skill values length")
    else:
        weights = np.ones(len(skills))

    # Convert to probabilities using softmax
    exp_skills = np.exp(skills)
    weighted_exp_skills = exp_skills * weights
    p = weighted_exp_skills / np.sum(weighted_exp_skills)

    # Compute Shannon entropy
    H = -np.sum(p * np.log(p + 1e-10))  # Add small epsilon to avoid log(0)

    # Normalize by maximum possible entropy (uniform distribution)
    # For 4 players: H_max = log(4)
    H_max = np.log(min(len(skills), 4))  # Cap at 4 for consistency

    # Lambda controls how much of the ace surplus to retain
    lambda_val = H / H_max if H_max > 0 else 0
    lambda_val = np.clip(lambda_val, 0, 1)  # Ensure in [0, 1]

    return H, lambda_val, p


def compute_entropy_controlled_rating(
    skill_values: List[float],
    exposure_weights: Optional[List[float]] = None,
    top_n: int = 4,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute entropy-controlled team rating.

    This implements the mathematical formula:
    R_EC = O + log(1 + λ·ρ)

    Where:
    - O is the baseline (average of top-N skills)
    - ρ is the ace surplus (difference between LSE and baseline)
    - λ is the entropy-based retention factor

    Args:
        skill_values: List of player skill values
        exposure_weights: Optional exposure weights
        top_n: Number of top players to consider (default 4)

    Returns:
        Tuple of (final_rating, debug_info)
    """
    if len(skill_values) == 0:
        raise ValueError("Team must have at least one player")

    # Sort skills in descending order and take top N
    skills = np.array(skill_values)
    sorted_skills = np.sort(skills)[::-1]
    top_skills = sorted_skills[: min(top_n, len(sorted_skills))]

    # Pad with very low values if fewer than top_n players
    if len(top_skills) < top_n:
        padding = np.full(top_n - len(top_skills), -10.0)
        top_skills = np.concatenate([top_skills, padding])

    # Compute baseline (average of top skills)
    O = np.mean(top_skills)

    # Compute log-sum-exp (team strength without penalty)
    lse = np.log(np.sum(np.exp(top_skills)))

    # Compute ace surplus
    rho = np.exp(lse - O) - 1.0

    # Compute entropy and lambda
    H, lambda_val, p = compute_team_entropy(
        top_skills[: min(top_n, len(skills))]
    )

    # Apply entropy-controlled shrinkage
    # R_EC = O + log(1 + λ·ρ)
    R_EC = O + np.log(1.0 + lambda_val * rho)

    # Gather debug information
    debug_info = {
        "baseline_O": O,
        "lse": lse,
        "ace_surplus_rho": rho,
        "entropy_H": H,
        "lambda": lambda_val,
        "shrinkage": lse - R_EC,
        "p_distribution": p.tolist() if len(p) <= 4 else p[:4].tolist(),
        "top_skills": top_skills.tolist(),
    }

    return R_EC, debug_info


def assign_divisions(
    teams: pl.DataFrame,
    rating_column: str = "entropy_rating",
    division_sizes: Optional[Dict[str, int]] = None,
) -> pl.DataFrame:
    """
    Assign teams to divisions based on their entropy-controlled ratings.

    Args:
        teams: Polars DataFrame with team information and ratings
        rating_column: Column name containing the ratings
        division_sizes: Optional dict mapping division names to sizes
                       Default: {'X': 12, '1': 24, '2': 24, ...}

    Returns:
        DataFrame with division assignments added
    """
    if division_sizes is None:
        division_sizes = {
            "X": 12,
            "1": 24,
            "2": 24,
            "3": 24,
            "4": 48,
            "5": 48,
            "6": 48,
            "7": 48,
            "8": 48,
        }

    # Sort teams by rating (descending)
    teams_sorted = teams.sort(rating_column, descending=True)

    # Create division assignments
    divisions = []
    current_idx = 0

    for div_name in ["X", "1", "2", "3", "4", "5", "6", "7", "8"]:
        if div_name not in division_sizes:
            continue

        div_size = division_sizes[div_name]
        end_idx = min(current_idx + div_size, len(teams_sorted))

        for i in range(current_idx, end_idx):
            divisions.append(div_name)

        current_idx = end_idx

        if current_idx >= len(teams_sorted):
            break

    # Handle any remaining teams
    while len(divisions) < len(teams_sorted):
        divisions.append("8")

    # Add division column
    teams_with_divs = teams_sorted.with_columns(
        pl.Series("assigned_division", divisions)
    )

    return teams_with_divs


class EntropyDivisionAssigner:
    """
    Main class for entropy-controlled division assignment.
    """

    def __init__(self, top_n: int = 4):
        """
        Initialize the division assigner.

        Args:
            top_n: Number of top players to consider for each team
        """
        self.top_n = top_n
        self.teams_data = None
        self.division_assignments = None

    def compute_team_ratings(
        self,
        teams: List[Dict],
        skill_field: str = "score",
        exposure_field: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Compute entropy-controlled ratings for all teams.

        Args:
            teams: List of team dictionaries, each containing:
                   - 'team_id': Team identifier
                   - 'players': List of player dicts with skill values
            skill_field: Field name in player dict containing skill value
            exposure_field: Optional field name for exposure weights

        Returns:
            Polars DataFrame with team ratings and debug information
        """
        results = []

        for team in teams:
            team_id = team["team_id"]
            players = team["players"]

            # Extract skill values
            skills = [p[skill_field] for p in players if skill_field in p]

            # Extract exposure weights if specified
            if exposure_field:
                exposures = [p.get(exposure_field, 1.0) for p in players]
            else:
                exposures = None

            # Compute entropy-controlled rating
            rating, debug_info = compute_entropy_controlled_rating(
                skills, exposures, self.top_n
            )

            # Store results
            result = {
                "team_id": team_id,
                "entropy_rating": rating,
                "num_players": len(skills),
                **debug_info,
            }

            results.append(result)

        self.teams_data = pl.DataFrame(results)
        return self.teams_data

    def compute_from_dataframe(
        self,
        teams_df: pl.DataFrame,
        players_df: pl.DataFrame,
        team_id_col: str = "team_id",
        player_team_col: str = "team_id",
        skill_col: str = "score",
        exposure_col: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Compute ratings from existing polars DataFrames.

        Args:
            teams_df: DataFrame with team information
            players_df: DataFrame with player information
            team_id_col: Column name for team ID in teams_df
            player_team_col: Column name for team ID in players_df
            skill_col: Column name for skill values
            exposure_col: Optional column name for exposure weights

        Returns:
            Polars DataFrame with team ratings
        """
        results = []

        # Get unique team IDs
        team_ids = teams_df.select(team_id_col).to_series().to_list()

        for team_id in team_ids:
            # Get players for this team
            team_players = players_df.filter(pl.col(player_team_col) == team_id)

            # Extract skill values
            skills = team_players.select(skill_col).to_series().to_list()

            if len(skills) == 0:
                continue

            # Extract exposure weights if specified
            if exposure_col and exposure_col in team_players.columns:
                exposures = (
                    team_players.select(exposure_col).to_series().to_list()
                )
            else:
                exposures = None

            # Compute entropy-controlled rating
            rating, debug_info = compute_entropy_controlled_rating(
                skills, exposures, self.top_n
            )

            # Store results
            result = {
                "team_id": team_id,
                "entropy_rating": rating,
                "num_players": len(skills),
                **debug_info,
            }

            results.append(result)

        self.teams_data = pl.DataFrame(results)
        return self.teams_data

    def assign_divisions(
        self, division_sizes: Optional[Dict[str, int]] = None
    ) -> pl.DataFrame:
        """
        Assign teams to divisions based on computed ratings.

        Args:
            division_sizes: Optional custom division sizes

        Returns:
            Polars DataFrame with division assignments
        """
        if self.teams_data is None:
            raise ValueError("Must compute team ratings first")

        self.division_assignments = assign_divisions(
            self.teams_data, "entropy_rating", division_sizes
        )

        return self.division_assignments

    def get_division_statistics(self) -> pl.DataFrame:
        """
        Get statistics for each division.

        Returns:
            Polars DataFrame with division-level statistics
        """
        if self.division_assignments is None:
            raise ValueError("Must assign divisions first")

        # Group by division and compute statistics
        stats = (
            self.division_assignments.group_by("assigned_division")
            .agg(
                [
                    pl.count("team_id").alias("num_teams"),
                    pl.mean("entropy_rating").alias("avg_rating"),
                    pl.min("entropy_rating").alias("min_rating"),
                    pl.max("entropy_rating").alias("max_rating"),
                    pl.mean("entropy_H").alias("avg_entropy"),
                    pl.mean("lambda").alias("avg_lambda"),
                    pl.mean("shrinkage").alias("avg_shrinkage"),
                ]
            )
            .sort("assigned_division")
        )

        return stats

    def evaluate_accuracy(
        self,
        actual_divisions: pl.DataFrame,
        team_id_col: str = "team_id",
        actual_div_col: str = "actual_division",
    ) -> Dict[str, float]:
        """
        Evaluate assignment accuracy against actual divisions.

        Args:
            actual_divisions: DataFrame with actual division assignments
            team_id_col: Column name for team ID
            actual_div_col: Column name for actual division

        Returns:
            Dictionary with accuracy metrics
        """
        if self.division_assignments is None:
            raise ValueError("Must assign divisions first")

        # Join with actual divisions
        comparison = self.division_assignments.join(
            actual_divisions.select([team_id_col, actual_div_col]),
            left_on="team_id",
            right_on=team_id_col,
            how="inner",
        )

        # Convert division names to numbers for comparison
        def div_to_num(div):
            return 0 if div == "X" else int(div) if div.isdigit() else 9

        comparison = comparison.with_columns(
            [
                pl.col("assigned_division")
                .map_elements(div_to_num, return_dtype=pl.Int32)
                .alias("pred_num"),
                pl.col(actual_div_col)
                .map_elements(div_to_num, return_dtype=pl.Int32)
                .alias("actual_num"),
            ]
        )

        # Calculate metrics
        total = len(comparison)
        exact = (
            comparison["assigned_division"] == comparison[actual_div_col]
        ).sum()

        comparison = comparison.with_columns(
            (pl.col("pred_num") - pl.col("actual_num")).abs().alias("diff")
        )

        within_1 = (comparison["diff"] <= 1).sum()
        within_2 = (comparison["diff"] <= 2).sum()

        return {
            "exact_accuracy": exact / total,
            "within_1_accuracy": within_1 / total,
            "within_2_accuracy": within_2 / total,
            "total_teams": total,
            "correct_exact": exact,
            "correct_within_1": within_1,
            "correct_within_2": within_2,
        }
