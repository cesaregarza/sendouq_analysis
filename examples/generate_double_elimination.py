"""
Example script for generating double elimination tournaments.

This script demonstrates how to generate various double elimination
tournament configurations with synthetic data.
"""

import json
from datetime import datetime

from synthetic_data import (
    DataSerializer,
    MatchSimulator,
    PlayerGenerator,
    TournamentGenerator,
)
from synthetic_data.tournament_generator import TournamentFormat
from synthetic_data.validator import DataValidator


def generate_small_double_elimination():
    """Generate a small double elimination tournament (8 teams)."""
    print("Generating small double elimination tournament (8 teams)...")

    # Create players with varied skill levels
    player_gen = PlayerGenerator(seed=100)
    players = player_gen.generate_players(
        n_players=32,  # 8 teams x 4 players
        skill_distribution="normal",
        skill_params={"mean": 0.0, "std": 1.2},
    )

    # Generate tournament
    tournament_gen = TournamentGenerator(seed=100)
    tournament = tournament_gen.generate_tournament(
        players=players,
        format=TournamentFormat.DOUBLE_ELIMINATION,
        team_size=4,
        name="Small Double Elimination Tournament",
        start_date=datetime(2024, 3, 1, 10, 0, 0),
        seeded=True,
    )

    # Simulate matches with skill-based outcomes
    match_sim = MatchSimulator(seed=100, alpha=1.5, skill_weight=0.8)

    # Serialize
    serializer = DataSerializer()
    data = serializer.serialize_tournament(
        tournament, simulate_matches=True, match_simulator=match_sim
    )

    # Validate
    validator = DataValidator()
    if validator.validate_serialized_data(data):
        print("✓ Small tournament data is valid!")
        _print_tournament_stats(tournament)
    else:
        print("✗ Validation failed:")
        print(validator.get_validation_report())

    return [data]


def generate_medium_double_elimination():
    """Generate a medium double elimination tournament (16 teams)."""
    print("\nGenerating medium double elimination tournament (16 teams)...")

    # Create tiered players
    player_gen = PlayerGenerator(seed=200)
    player_pool = player_gen.create_player_pool_with_categories(
        n_elite=12, n_competitive=40, n_casual=12
    )

    # Combine all players
    all_players = []
    for players in player_pool.values():
        all_players.extend(players)

    # Generate tournament
    tournament_gen = TournamentGenerator(seed=200)
    tournament = tournament_gen.generate_tournament(
        players=all_players[:64],  # 16 teams x 4 players
        format=TournamentFormat.DOUBLE_ELIMINATION,
        team_size=4,
        name="Medium Double Elimination Championship",
        start_date=datetime(2024, 3, 15, 9, 0, 0),
        seeded=True,
    )

    # Simulate with balanced settings
    match_sim = MatchSimulator(seed=200, alpha=1.2, skill_weight=0.75)

    # Serialize
    serializer = DataSerializer()
    data = serializer.serialize_tournament(
        tournament, simulate_matches=True, match_simulator=match_sim
    )

    # Validate
    validator = DataValidator()
    if validator.validate_serialized_data(data):
        print("✓ Medium tournament data is valid!")
        _print_tournament_stats(tournament)
    else:
        print("✗ Validation failed:")
        print(validator.get_validation_report())

    return [data]


def generate_large_double_elimination_with_byes():
    """Generate a large double elimination tournament with byes (24 teams)."""
    print(
        "\nGenerating large double elimination tournament with byes (24 teams)..."
    )

    # Create diverse player pool
    player_gen = PlayerGenerator(seed=300)
    player_pool = player_gen.create_player_pool_with_categories(
        n_elite=20, n_competitive=60, n_casual=16
    )

    all_players = []
    for players in player_pool.values():
        all_players.extend(players)

    # Generate tournament - 24 teams means 8 teams get byes
    tournament_gen = TournamentGenerator(seed=300)
    tournament = tournament_gen.generate_tournament(
        players=all_players[:96],  # 24 teams x 4 players
        format=TournamentFormat.DOUBLE_ELIMINATION,
        team_size=4,
        name="Major Double Elimination Tournament",
        start_date=datetime(2024, 4, 1, 8, 0, 0),
        seeded=True,
    )

    # Simulate with some upset potential
    match_sim = MatchSimulator(seed=300, alpha=1.0, skill_weight=0.7)

    # Serialize
    serializer = DataSerializer()
    data = serializer.serialize_tournament(
        tournament, simulate_matches=True, match_simulator=match_sim
    )

    # Validate
    validator = DataValidator()
    if validator.validate_serialized_data(data):
        print("✓ Large tournament data is valid!")
        _print_tournament_stats(tournament)
    else:
        print("✗ Validation failed:")
        print(validator.get_validation_report())

    return [data]


def generate_unseeded_double_elimination():
    """Generate an unseeded double elimination tournament."""
    print("\nGenerating unseeded double elimination tournament (12 teams)...")

    # Create players with similar skills
    player_gen = PlayerGenerator(seed=400)
    players = player_gen.generate_players(
        n_players=48,  # 12 teams x 4 players
        skill_distribution="normal",
        skill_params={"mean": 0.0, "std": 0.8},  # Lower variance
    )

    # Generate unseeded tournament
    tournament_gen = TournamentGenerator(seed=400)
    tournament = tournament_gen.generate_tournament(
        players=players,
        format=TournamentFormat.DOUBLE_ELIMINATION,
        team_size=4,
        name="Open Double Elimination Tournament",
        start_date=datetime(2024, 4, 15, 12, 0, 0),
        seeded=False,  # Random initial matchups
    )

    # Simulate with high randomness
    match_sim = MatchSimulator(seed=400, alpha=0.8, skill_weight=0.6)

    # Serialize
    serializer = DataSerializer()
    data = serializer.serialize_tournament(
        tournament, simulate_matches=True, match_simulator=match_sim
    )

    # Validate
    validator = DataValidator()
    if validator.validate_serialized_data(data):
        print("✓ Unseeded tournament data is valid!")
        _print_tournament_stats(tournament)
    else:
        print("✗ Validation failed:")
        print(validator.get_validation_report())

    return [data]


def _print_tournament_stats(tournament):
    """Print statistics about the generated tournament."""
    stage = tournament.stages[0]

    print(f"  - Teams: {len(stage.teams)}")
    print(f"  - Winners bracket rounds: {len(stage.winners_bracket)}")
    print(f"  - Losers bracket rounds: {len(stage.losers_bracket)}")
    print(f"  - Grand finals matches: {len(stage.grand_finals)}")

    total_matches = sum(len(matches) for matches in stage.rounds.values())
    print(f"  - Total matches: {total_matches}")

    # Count matches by bracket type
    wb_matches = sum(len(matches) for matches in stage.winners_bracket.values())
    lb_matches = sum(len(matches) for matches in stage.losers_bracket.values())
    gf_matches = len(stage.grand_finals)

    print(f"  - Winners bracket matches: {wb_matches}")
    print(f"  - Losers bracket matches: {lb_matches}")
    print(f"  - Grand finals matches: {gf_matches}")


def test_parser_compatibility(tournaments):
    """Test that generated tournaments work with the parser."""
    print("\nTesting parser compatibility...")

    try:
        from rankings.core import parse_tournaments_data

        tables = parse_tournaments_data(tournaments)
        print("✓ Parser compatibility confirmed")
        print(f"  - Matches: {len(tables['matches'])}")
        print(f"  - Players: {len(tables['players'])}")
        print(f"  - Teams: {len(tables['teams'])}")
        print(f"  - Tournaments: {len(tables['tournaments'])}")

    except Exception as e:
        print(f"✗ Parser compatibility failed: {e}")


def save_tournaments_to_file(tournaments, filename):
    """Save tournament data to JSON file."""
    with open(filename, "w") as f:
        json.dump(tournaments, f, indent=2, default=str)
    print(f"\nSaved {len(tournaments)} tournaments to {filename}")


if __name__ == "__main__":
    # Generate different tournament configurations
    small_data = generate_small_double_elimination()
    medium_data = generate_medium_double_elimination()
    large_data = generate_large_double_elimination_with_byes()
    unseeded_data = generate_unseeded_double_elimination()

    # Save individual examples
    save_tournaments_to_file(small_data, "synthetic_double_elim_small.json")
    save_tournaments_to_file(medium_data, "synthetic_double_elim_medium.json")
    save_tournaments_to_file(large_data, "synthetic_double_elim_large.json")
    save_tournaments_to_file(
        unseeded_data, "synthetic_double_elim_unseeded.json"
    )

    # Create combined dataset
    all_tournaments = small_data + medium_data + large_data + unseeded_data
    save_tournaments_to_file(all_tournaments, "synthetic_double_elim_all.json")

    # Test parser compatibility
    test_parser_compatibility(all_tournaments)

    print("\nAll double elimination examples completed!")
