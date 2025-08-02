"""
Example script demonstrating synthetic tournament data generation.

This script shows how to generate various tournament formats with synthetic data
and validate the output for use with the ranking system.
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


def generate_simple_swiss_tournament():
    """Generate a simple Swiss tournament with 16 teams."""
    print("Generating Swiss tournament...")
    
    # Create player pool
    player_gen = PlayerGenerator(seed=42)
    players = player_gen.generate_players(
        n_players=64,  # 16 teams x 4 players
        skill_distribution="normal",
        skill_params={"mean": 0.0, "std": 1.0}
    )
    
    # Generate tournament
    tournament_gen = TournamentGenerator(seed=42)
    tournament = tournament_gen.generate_tournament(
        players=players,
        format=TournamentFormat.SWISS,
        team_size=4,
        name="Example Swiss Tournament",
        start_date=datetime(2024, 1, 15, 10, 0, 0),
        n_rounds=5
    )
    
    # Simulate matches
    match_sim = MatchSimulator(seed=42, alpha=1.0, bo_games=7)
    
    # Serialize to JSON
    serializer = DataSerializer()
    data = serializer.serialize_tournament(tournament, simulate_matches=True, match_simulator=match_sim)
    
    # Validate
    validator = DataValidator()
    if validator.validate_serialized_data(data):
        print("✓ Tournament data is valid!")
    else:
        print("✗ Validation failed:")
        print(validator.get_validation_report())
        
    return [data]  # Return as list for parser compatibility


def generate_competitive_tournament():
    """Generate a more complex tournament with multiple skill tiers."""
    print("\nGenerating competitive tournament with skill tiers...")
    
    # Create tiered player pool
    player_gen = PlayerGenerator(seed=123)
    player_pool = player_gen.create_player_pool_with_categories(
        n_elite=20,
        n_competitive=80,
        n_casual=100
    )
    
    # Combine all players
    all_players = []
    for category, players in player_pool.items():
        all_players.extend(players)
        
    # Generate multi-stage tournament
    tournament_gen = TournamentGenerator(seed=123)
    tournament = tournament_gen.generate_multi_stage_tournament(
        players=all_players[:128],  # Use 128 players for 32 teams
        stages_config=[
            {
                "format": "group_stage",
                "n_groups": 8,
                "advance_count": 16  # Top 2 from each group
            },
            {
                "format": "single_elimination",
                "seeded": True
            }
        ],
        team_size=4,
        name="Sendouq Championship Series",
        start_date=datetime(2024, 2, 1, 9, 0, 0)
    )
    
    # Simulate with match simulator
    match_sim = MatchSimulator(seed=123, alpha=1.2, skill_weight=0.75)
    
    # Serialize
    serializer = DataSerializer()
    data = serializer.serialize_tournament(tournament, simulate_matches=True, match_simulator=match_sim)
    
    # Validate
    validator = DataValidator()
    is_valid = validator.validate_serialized_data(data)
    print(f"Tournament validation: {'✓ Passed' if is_valid else '✗ Failed'}")
    
    if not is_valid:
        print(validator.get_validation_report())
        
    return [data]


def generate_tournament_series():
    """Generate a series of tournaments to test cross-validation."""
    print("\nGenerating tournament series for cross-validation testing...")
    
    # Create persistent player pool
    player_gen = PlayerGenerator(seed=999)
    base_players = player_gen.create_player_pool_with_categories(
        n_elite=30,
        n_competitive=120,
        n_casual=250
    )
    
    all_players = []
    for players in base_players.values():
        all_players.extend(players)
        
    tournaments = []
    tournament_gen = TournamentGenerator(seed=999)
    match_sim = MatchSimulator(seed=999)
    serializer = DataSerializer()
    
    # Generate 5 tournaments over time
    formats = [
        TournamentFormat.SWISS,
        TournamentFormat.ROUND_ROBIN,
        TournamentFormat.SINGLE_ELIMINATION,
        TournamentFormat.SWISS,
        TournamentFormat.GROUP_STAGE
    ]
    
    for i, format in enumerate(formats):
        # Sample subset of players (simulating different participation)
        import random
        rng = random.Random(999 + i)
        participating_players = rng.sample(all_players, k=min(64, len(all_players)))
        
        tournament = tournament_gen.generate_tournament(
            players=participating_players,
            format=format,
            team_size=4,
            name=f"Weekly Tournament #{i+1}",
            start_date=datetime(2024, 1, 1 + i * 7, 18, 0, 0)
        )
        
        # Serialize with simulation
        data = serializer.serialize_tournament(
            tournament,
            simulate_matches=True,
            match_simulator=match_sim
        )
        
        tournaments.append(data)
        
    print(f"Generated {len(tournaments)} tournaments")
    
    # Validate all tournaments
    validator = DataValidator()
    all_valid = all(validator.validate_serialized_data(t) for t in tournaments)
    print(f"All tournaments valid: {'✓ Yes' if all_valid else '✗ No'}")
    
    # Test parser compatibility
    try:
        from rankings.core import parse_tournaments_data
        tables = parse_tournaments_data(tournaments)
        print(f"✓ Parser compatibility confirmed")
        print(f"  - Matches: {len(tables['matches'])}")
        print(f"  - Players: {len(tables['players'])}")
        print(f"  - Teams: {len(tables['teams'])}")
    except Exception as e:
        print(f"✗ Parser compatibility failed: {e}")
        
    return tournaments


def save_tournaments_to_file(tournaments, filename):
    """Save tournament data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(tournaments, f, indent=2, default=str)
    print(f"\nSaved {len(tournaments)} tournaments to {filename}")


if __name__ == "__main__":
    # Generate different tournament types
    swiss_data = generate_simple_swiss_tournament()
    competitive_data = generate_competitive_tournament()
    series_data = generate_tournament_series()
    
    # Save examples
    save_tournaments_to_file(swiss_data, "synthetic_swiss_tournament.json")
    save_tournaments_to_file(competitive_data, "synthetic_competitive_tournament.json")
    save_tournaments_to_file(series_data, "synthetic_tournament_series.json")
    
    print("\nAll examples completed!")