"""
Example: Converting model outputs to team seeding.

This demonstrates how to take player rankings from the model and convert them
into team strength scores for tournament seeding.
"""

import polars as pl

from rankings.seedings.team_seeding import TeamSeeding, TeamSeedingConfig, seed_teams


def example_simple_seeding():
    """Simple example: Seed teams from player ratings."""
    print("=" * 60)
    print("Example 1: Simple Team Seeding")
    print("=" * 60)

    # Simulate model output: player rankings with scores
    player_ratings = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "score": [
                2.5,  # Best player
                2.1,
                1.8,
                1.5,
                1.2,
                1.0,
                0.8,
                0.5,
                0.3,
                0.0,
                -0.5,
                -1.0,  # Worst player
            ],
        }
    )

    # Define teams (team_id -> list of player_ids)
    teams = {
        101: [1, 5, 9, 12],  # Mixed team: 1 ace, 3 average
        102: [2, 3, 4, 10],  # Balanced team: 3 good, 1 weak
        103: [6, 7, 8, 11],  # Weaker team
    }

    # Compute seeding
    seeds = seed_teams(player_ratings, teams, use_top_k=4)

    print("\nPlayer Ratings:")
    print(player_ratings)

    print("\nTeam Rosters:")
    for team_id, roster in teams.items():
        print(f"  Team {team_id}: {roster}")

    print("\nSeeding Results:")
    print(seeds)

    print("\nInterpretation:")
    print("  - seed_rank 1 = strongest team (should be top seed)")
    print("  - strength = log-scale team rating from player scores")


def example_with_exposure():
    """Example with exposure weights (e.g., partial participation)."""
    print("\n" + "=" * 60)
    print("Example 2: Seeding with Exposure Weights")
    print("=" * 60)

    # Player ratings from model
    player_ratings = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "score": [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5],
        }
    )

    teams = {
        201: [1, 2, 3, 4],  # Has star player (1) but at reduced exposure
        202: [5, 6, 7, 8],  # All players at full exposure
    }

    # Exposure weights: (player_id, team_id) -> weight
    # Player 1 only played 50% of matches on team 201
    exposure_weights = {
        (1, 201): 0.5,  # Star player at 50% exposure
    }

    config = TeamSeedingConfig(use_top_k=4)
    seeder = TeamSeeding(config)

    # Compute with exposure
    seeds_with_exposure = seeder.compute_all_teams(
        player_ratings, teams, exposure_weights
    )

    # Compare to no exposure adjustment
    seeds_no_exposure = seeder.compute_all_teams(player_ratings, teams, None)

    print("\nWith exposure adjustment (player 1 at 50%):")
    print(seeds_with_exposure)

    print("\nWithout exposure adjustment:")
    print(seeds_no_exposure)

    print("\nNote:")
    print("  Team 201's strength is lower when accounting for reduced exposure")


def example_from_dataframe():
    """Example using DataFrame input (typical for database queries)."""
    print("\n" + "=" * 60)
    print("Example 3: Seeding from DataFrames")
    print("=" * 60)

    # Player ratings (from model output)
    player_ratings = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "score": [2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2],
            "exposure": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    # Team rosters (from database)
    team_rosters = pl.DataFrame(
        {
            "team_id": [301, 301, 301, 301, 302, 302, 302, 302],
            "player_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "exposure": [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
        }
    )

    seeder = TeamSeeding()
    seeds = seeder.compute_from_dataframe(player_ratings, team_rosters)

    print("\nPlayer Ratings:")
    print(player_ratings.select(["id", "score"]))

    print("\nTeam Rosters:")
    print(team_rosters)

    print("\nSeeding Results:")
    print(seeds)


def example_diminishing_returns():
    """Example showing effect of diminishing returns parameter."""
    print("\n" + "=" * 60)
    print("Example 4: Diminishing Returns (alpha parameter)")
    print("=" * 60)

    # One team with a superstar, one with balanced players
    player_ratings = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "score": [4.0, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5],
            #          ^ superstar      balanced team ->
        }
    )

    teams = {
        401: [1, 2, 3, 4],  # Superstar + 3 weak
        402: [5, 6, 7, 8],  # All balanced
    }

    print("\nTeam 401: Superstar (4.0) + 3 weak (0.5 each)")
    print("Team 402: All balanced (1.5 each)")

    # No diminishing returns (alpha=1.0)
    seeds_no_dim = seed_teams(player_ratings, teams, alpha=1.0)
    print("\nNo diminishing returns (alpha=1.0):")
    print(seeds_no_dim)

    # Strong diminishing returns (alpha=0.5)
    seeds_dim = seed_teams(player_ratings, teams, alpha=0.5)
    print("\nWith diminishing returns (alpha=0.5):")
    print(seeds_dim)

    print("\nInterpretation:")
    print("  - Lower alpha = more penalty for skill imbalance")
    print("  - alpha=1.0 favors superstar teams")
    print("  - alpha<1.0 favors balanced teams")


if __name__ == "__main__":
    # Run all examples
    example_simple_seeding()
    example_with_exposure()
    example_from_dataframe()
    example_diminishing_returns()

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print("""
The team seeding module converts model outputs to team strength scores:

1. Input: Player ratings (id, score) from ranking model
2. Process: Aggregate player skills using log-sum-exp
3. Output: Team strength scores and seed rankings

Key parameters:
- use_top_k: Only use top K players (default 4)
- alpha: Diminishing returns factor (1.0 = none, <1.0 = favor balance)
- exposure: Weight for partial participation

Use cases:
- Tournament bracket seeding
- Division assignment
- Matchmaking balance
    """)
