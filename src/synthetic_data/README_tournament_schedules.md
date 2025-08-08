# Tournament Schedules Module

The `tournament_schedules.py` module provides utilities for generating realistic tournament schedules for synthetic data generation.

## Quick Start

```python
from synthetic_data.circuits import (
    create_dense_schedule,
    create_sparse_schedule,
    create_competitive_season
)

# Create a dense schedule (~500 tournaments over 180 days)
dense_configs = create_dense_schedule(days=180)

# Create a sparse schedule (~100 tournaments over 180 days)
sparse_configs = create_sparse_schedule(days=180)

# Create a competitive season with majors
season_configs = create_competitive_season(months=6)
```

## Features

### Pre-configured Schedule Types

1. **Dense Schedule** - High tournament frequency
   - ~3 tournaments per day average
   - Mix of small to mega tournaments
   - Addresses the correlation issues found in testing

2. **Sparse Schedule** - Lower tournament frequency
   - ~0.5 tournaments per day average
   - Larger tournaments on average
   - Good for testing with limited data

3. **Competitive Season** - Structured competitive circuit
   - Monthly major championships
   - Open qualifiers before each major
   - Regular tournaments throughout

### Tournament Size Distribution

All schedules respect minimum team sizes (default: 20 teams) to avoid unrealistically small tournaments:
- Small: 20-48 teams
- Medium: 64-128 teams
- Large: 192-384 teams
- Mega: 512+ teams

### Tournament Types

Schedules include a realistic mix of:
- **Skill-capped** (~50%) - Limited by skill rating
- **Open** (~35%) - Anyone can enter
- **Invitational** (~15%) - Top players only

### Advanced Usage

For more control, use the `TournamentScheduleGenerator` class:

```python
from synthetic_data.tournament_schedules import TournamentScheduleGenerator

generator = TournamentScheduleGenerator(seed=42)

# Custom dense schedule
configs = generator.create_dense_schedule(
    days=180,
    min_team_size=32,  # Larger minimum
    tournaments_per_day=(2, 6),  # 2-6 per day
    size_distribution={
        "small": 0.3,
        "medium": 0.4,
        "large": 0.2,
        "mega": 0.1,
    }
)

# Custom weekly series
custom_series = [{
    "name": "Pro_League",
    "type": TournamentType.INVITATIONAL,
    "teams": 16,
    "format": TournamentFormat.DOUBLE_ELIMINATION,
}]
configs = generator.create_weekly_series(weeks=26, series_configs=custom_series)
```

## Integration with Tournament Circuit

```python
from synthetic_data.tournament_circuit import TournamentCircuit
from synthetic_data.tournament_schedules import create_dense_schedule

# Create circuit
circuit = TournamentCircuit(
    seed=42,
    player_pool_size=1000,
    skill_distribution="normal",
    skill_params={"mean": 10.0, "std": 3.0}
)

# Use dense schedule
configs = create_dense_schedule(days=180)

# Generate tournaments
results = circuit.generate_circuit(configs)
```

## Key Findings from Testing

This module was created based on extensive testing that revealed:
1. Tournament density is critical for good PageRank correlations
2. Very small tournaments (< 20 teams) are unrealistic and hurt rankings
3. A mix of tournament types and sizes better represents real communities
4. Dense schedules (~500 tournaments/6 months) match real-world data better than sparse schedules (~100 tournaments/6 months)