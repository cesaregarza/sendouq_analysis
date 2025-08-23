"""Simple time-based splitting for cross-validation.

This module provides clear, straightforward temporal splitting logic for
cross-validation. The approach is:
1. Divide time into folds
2. For each fold, pick multiple cutoff points
3. For each cutoff: train on all before, test on next N tournaments
"""

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def create_simple_time_splits(
    matches_df: pl.DataFrame,
    n_splits: int = 5,
    test_tournaments_per_split: int = 20,
    test_samples_per_split: int = 5,
    min_train_tournaments: int = 50,
) -> list[tuple[pl.DataFrame, pl.DataFrame, list[str]]]:
    """Create simple time-based train/test splits for cross-validation.

    This function implements temporal cross-validation where:
    - Training data always comes before test data in time
    - Each test set contains a fixed number of consecutive tournaments
    - Multiple samples are taken within each fold to reduce noise

    Args:
        matches_df: DataFrame with matches including 'tournament_id' and 'start_time'
        n_splits: Number of cross-validation folds
        test_tournaments_per_split: Number of consecutive tournaments in each test set
        test_samples_per_split: Number of different cutoff points to test per fold
        min_train_tournaments: Minimum number of tournaments required for training

    Returns:
        List of (train_df, test_df, test_tournament_ids) tuples
    """
    # Get tournaments sorted by time
    # Use last_game_finished_at as it has actual timestamp data
    tournaments = (
        matches_df.group_by("tournament_id")
        .agg(pl.col("last_game_finished_at").min().alias("start_time"))
        .sort("start_time")
    )

    tournament_ids = tournaments["tournament_id"].to_list()
    n_tournaments = len(tournament_ids)

    logger.info(f"Total tournaments: {n_tournaments}")

    # Calculate fold boundaries
    # Leave space for test tournaments at the end
    usable_tournaments = n_tournaments - test_tournaments_per_split

    # Ensure we have enough tournaments
    if usable_tournaments < min_train_tournaments:
        raise ValueError(
            f"Not enough tournaments: need at least {min_train_tournaments + test_tournaments_per_split}, "
            f"but only have {n_tournaments}"
        )

    # Create folds
    fold_size = (usable_tournaments - min_train_tournaments) // n_splits

    splits = []

    for fold_idx in range(n_splits):
        # Define the range for this fold
        fold_start = min_train_tournaments + fold_idx * fold_size
        fold_end = min_train_tournaments + (fold_idx + 1) * fold_size

        # For the last fold, extend to use all remaining tournaments
        if fold_idx == n_splits - 1:
            fold_end = usable_tournaments

        logger.info(
            f"Fold {fold_idx + 1}: Testing {test_samples_per_split} cutoff points in range [{fold_start}, {fold_end}]"
        )

        # Create multiple samples within this fold
        if test_samples_per_split == 1:
            # Single sample at the midpoint
            cutoff_indices = [(fold_start + fold_end) // 2]
        else:
            # Multiple samples spread across the fold
            cutoff_indices = np.linspace(
                fold_start,
                fold_end - 1,  # -1 to ensure we don't go past the boundary
                test_samples_per_split,
                dtype=int,
            ).tolist()

        for cutoff_idx in cutoff_indices:
            # Training: all tournaments up to cutoff
            train_ids = tournament_ids[:cutoff_idx]

            # Test: next N consecutive tournaments
            test_ids = tournament_ids[
                cutoff_idx : cutoff_idx + test_tournaments_per_split
            ]

            # Filter matches
            train_df = matches_df.filter(
                pl.col("tournament_id").is_in(train_ids)
            )
            test_df = matches_df.filter(pl.col("tournament_id").is_in(test_ids))

            splits.append((train_df, test_df, test_ids))

    logger.info(
        f"Created {len(splits)} total train/test pairs from {n_splits} folds"
    )

    return splits


def visualize_splits(
    splits: list[tuple[pl.DataFrame, pl.DataFrame, list[str]]],
) -> None:
    """Visualize train/test splits with ASCII art.

    Args:
        splits: List of (train_df, test_df, test_tournament_ids) tuples
    """
    print("\nCross-validation splits visualization:")
    print("=" * 60)

    for i, (train_df, test_df, test_ids) in enumerate(splits):
        # Get tournament counts
        n_train_tournaments = train_df["tournament_id"].n_unique()
        n_test_tournaments = len(test_ids)

        # Get match counts
        n_train_matches = train_df.height
        n_test_matches = test_df.height

        # Calculate time gap between train and test
        if train_df.height > 0 and test_df.height > 0:
            max_train_time = train_df["last_game_finished_at"].max()
            min_test_time = test_df["last_game_finished_at"].min()

            # Handle None values or check if times are valid
            if max_train_time is not None and min_test_time is not None:
                # These are Unix timestamps in seconds
                gap_days = (min_test_time - max_train_time) / 86400
            else:
                gap_days = 0
        else:
            gap_days = 0

        # Create visual representation
        # Scale to max 50 characters width
        max_width = 40
        total_tournaments = n_train_tournaments + n_test_tournaments
        train_width = int((n_train_tournaments / total_tournaments) * max_width)
        test_width = int((n_test_tournaments / total_tournaments) * max_width)

        # Ensure at least 1 character for visibility
        train_width = max(1, train_width)
        test_width = max(1, test_width)

        print(f"\nSplit {i + 1}:")
        print(
            f"  Train: {'█' * train_width} ({n_train_tournaments} tournaments)"
        )
        print(f"  Test:  {'▓' * test_width} ({n_test_tournaments} tournaments)")
        print(f"  Train matches: {n_train_matches:,}")
        print(f"  Test matches: {n_test_matches:,}")
        print(f"  Gap between train/test: {gap_days:.1f} days")

    print("\n" + "=" * 60)


def get_split_info(train_df: pl.DataFrame, test_df: pl.DataFrame) -> dict:
    """Get information about a train/test split.

    Args:
        train_df: Training data
        test_df: Test data

    Returns:
        Dictionary with split statistics
    """
    info = {
        "n_train_tournaments": train_df["tournament_id"].n_unique(),
        "n_test_tournaments": test_df["tournament_id"].n_unique(),
        "n_train_matches": train_df.height,
        "n_test_matches": test_df.height,
    }

    # Time statistics
    if train_df.height > 0:
        info["train_start"] = train_df["last_game_finished_at"].min()
        info["train_end"] = train_df["last_game_finished_at"].max()
        if info["train_start"] is not None and info["train_end"] is not None:
            # These are Unix timestamps in seconds
            info["train_duration_days"] = (
                info["train_end"] - info["train_start"]
            ) / 86400

    if test_df.height > 0:
        info["test_start"] = test_df["last_game_finished_at"].min()
        info["test_end"] = test_df["last_game_finished_at"].max()
        if info["test_start"] is not None and info["test_end"] is not None:
            # These are Unix timestamps in seconds
            info["test_duration_days"] = (
                info["test_end"] - info["test_start"]
            ) / 86400

    # Gap between train and test
    if (
        train_df.height > 0
        and test_df.height > 0
        and "train_end" in info
        and "test_start" in info
    ):
        if info["train_end"] is not None and info["test_start"] is not None:
            # These are Unix timestamps in seconds
            info["gap_days"] = (info["test_start"] - info["train_end"]) / 86400

    return info
