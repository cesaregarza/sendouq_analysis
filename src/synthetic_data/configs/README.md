# Synthetic Data Configuration

This directory contains configuration and recipe modules for generating realistic synthetic tournament data that mirrors actual Sendou.ink tournament patterns.

## Files

### `realistic_recipe.py`

The main tournament generation recipe that creates synthetic tournaments matching real-world characteristics observed in ~1800 Sendou.ink tournaments.

#### Key Features

**Tournament Characteristics:**
- **Team sizes**: 4-5 players (60% teams of 4, 40% teams of 5)
- **Tournament sizes**: 4-32 teams per tournament (most common: 6-12 teams)
- **Tournament formats**: 
  - Single elimination (51%)
  - Double elimination (27%)
  - Round Robin (15%)
  - Swiss (7%)
- **Match formats**: Best-of-3 (70%), Best-of-5 (25%), Best-of-7 (5%)
- **Scheduling**: Multiple tournaments per day with peak hours 18:00-22:00

**Player Population:**
- **Skill distribution**: Normal distribution (mean=20, std=8) on a 5-35 scale
- **Participation patterns**:
  - Casual players (25%): 1-5% participation rate
  - Occasional players (27%): 5-20% participation rate  
  - Regular players (48%): 20-80% participation rate
- **Country distribution**: Realistic mix of US, JP, FR, DE, UK, etc.

**Match Simulation:**
- Win probability based on average team skill using logistic function
- Skill difference of 5 points ≈ 62% win probability
- Small upset factor (±2%) to add realistic variance

#### Usage

```python
from synthetic_data.configs.realistic_recipe import create_realistic_tournament_dataset
from datetime import datetime

# Generate 100 tournaments over a year with 2000 players
tournaments = create_realistic_tournament_dataset(
    n_tournaments=100,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    n_players=2000,
    seed=42
)

# Each tournament contains:
# - id: Tournament identifier
# - date: ISO format datetime
# - format: Tournament format (single_elimination, etc.)
# - match_format: Match format (bo3, bo5, bo7)
# - teams: List of teams with players and skills
# - matches: List of match results
```

#### Default Parameters

The `create_realistic_tournament_dataset` function defaults:
- `n_tournaments`: 100 (not 10 - that's just for the test in __main__)
- `n_players`: 2000
- `start_date`: January 1, 2024
- `end_date`: December 31, 2024

The scheduling uses a Poisson distribution to create realistic tournament clustering (some days have multiple tournaments, others have none). The function generates more tournament time slots than requested but only creates exactly `n_tournaments` tournaments from the schedule.

### `recipe_validator.py`

A validation script that compares synthetic tournament data against real Sendou.ink data to ensure realistic characteristics.

#### Features

- Loads real tournament data from scraped Sendou.ink tournaments
- Generates synthetic tournaments using the realistic recipe
- Compares key statistics:
  - Team count distributions
  - Match count distributions
  - Tournament format distributions
  - Team size distributions
  - Score distributions
  - Player pool sizes

#### Usage

```python
from synthetic_data.configs.recipe_validator import validate_recipe

# Run validation comparing 100 synthetic tournaments to real data
comparison = validate_recipe(
    n_synthetic_tournaments=100,
    n_synthetic_players=1000
)
```

## Data Model

### Tournament Structure

```python
{
    "id": 1,
    "date": "2024-01-01T18:00:00",
    "format": "single_elimination",
    "match_format": "bo3",
    "teams": [
        {
            "id": 1,
            "name": "Team_1",
            "seed": 1,
            "players": [
                {"id": 101, "username": "Player_101"},
                # ...
            ],
            "avg_skill": 25.3
        },
        # ...
    ],
    "matches": [
        {
            "round": 1,
            "team1_id": 1,
            "team2_id": 8,
            "score1": 2,
            "score2": 0,
            "winner_id": 1
        },
        # ...
    ]
}
```

### Player Structure

```python
{
    "id": 1,
    "username": "Player_1",
    "skill_ordinal": 22.5,
    "participation_rate": 0.35,
    "frequency_type": "regular",
    "country": "US",
    "created_at": datetime(2023, 3, 15)
}
```

## Validation Results

Based on comparison with ~1800 real tournaments:

| Metric | Real Data | Synthetic Data |
|--------|-----------|----------------|
| Teams per tournament | Mean: 11.3 | Mean: 10.8 |
| Matches per tournament | Mean: 28.5 | Mean: 26.2 |
| Most common team size | 4 players (40%) | 4 players (40%) |
| Single elimination | 51% | 51% |
| Player pool | 8401 unique | Configurable |

## Notes

- The recipe uses a seeded random number generator for reproducibility
- Team formation is currently random within tournaments (not skill-based) to match observed patterns
- Player skills remain static within the dataset (no growth over time in base recipe)
- For experiments with player growth, skill-based teams, or other variations, see the evaluation modules