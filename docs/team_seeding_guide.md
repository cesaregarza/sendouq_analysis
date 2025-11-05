# Team Seeding from Model Outputs

This guide explains how to convert player rankings from the model into team strength scores for tournament seeding.

## Overview

The team seeding system takes **player ratings** (individual skill scores) and converts them into **team strength scores** using log-sum-exp aggregation. This allows you to:

1. Seed teams for tournament brackets
2. Assign teams to skill divisions
3. Balance matchmaking

## Quick Start

### Simple Usage

```python
import polars as pl
from rankings.seedings.team_seeding import seed_teams

# Player ratings from model output
player_ratings = pl.DataFrame({
    "id": [1, 2, 3, 4, 5, 6, 7, 8],
    "score": [2.5, 2.1, 1.8, 1.5, 1.2, 1.0, 0.8, 0.5]
})

# Define teams
teams = {
    101: [1, 2, 3, 4],  # Team 101 has players 1,2,3,4
    102: [5, 6, 7, 8],  # Team 102 has players 5,6,7,8
}

# Compute seeding
seeds = seed_teams(player_ratings, teams)
print(seeds)
```

**Output:**
```
┌─────────┬──────────┬───────────┐
│ team_id ┆ strength ┆ seed_rank │
├─────────┼──────────┼───────────┤
│ 101     ┆ 3.45     ┆ 1         │
│ 102     ┆ 2.12     ┆ 2         │
└─────────┴──────────┴───────────┘
```

## Input Format

### Player Ratings (from Model)

The model outputs player rankings as a Polars DataFrame with:

- **`id`**: Player ID (integer)
- **`score`**: Log-odds skill rating (float)
- **`exposure`** (optional): Activity/exposure metric

Example from `run_rankings`:
```python
pl.DataFrame({
    "id": [1001, 1002, 1003],
    "score": [2.5, 1.8, 0.3],
    "exposure": [1.0, 0.9, 1.0]
})
```

### Team Rosters

Teams can be provided in two formats:

#### 1. Dictionary (simple)
```python
teams = {
    team_id: [player_ids],
    101: [1, 2, 3, 4],
    102: [5, 6, 7, 8]
}
```

#### 2. DataFrame (from database)
```python
team_rosters = pl.DataFrame({
    "team_id": [101, 101, 101, 101, 102, 102],
    "player_id": [1, 2, 3, 4, 5, 6],
    "exposure": [1.0, 1.0, 0.8, 1.0, 1.0, 1.0]  # optional
})
```

## Output Format

The seeding output is a Polars DataFrame with:

- **`team_id`**: Team identifier
- **`strength`**: Team strength score (log-scale)
- **`seed_rank`**: Tournament seed (1 = strongest)

Teams are automatically sorted by strength (descending).

## Configuration Options

### Parameters

```python
from rankings.seedings.team_seeding import TeamSeeding, TeamSeedingConfig

config = TeamSeedingConfig(
    alpha=1.0,              # Diminishing returns factor
    use_top_k=4,            # Only use top-k players
    default_exposure=1.0,   # Default exposure weight
    player_id_col="id",     # Column name for player IDs
    score_col="score",      # Column name for scores
    exposure_col="exposure" # Column name for exposure
)

seeder = TeamSeeding(config)
```

### Alpha (Diminishing Returns)

Controls how much superstar players are worth:

- **`alpha = 1.0`** (default): No diminishing returns - linear aggregation
- **`alpha < 1.0`**: Diminishing returns - favors balanced teams
- **`alpha = 0.5`**: Strong penalty for skill imbalance

Example:
```python
# Favor superstar teams
seeds_linear = seed_teams(ratings, teams, alpha=1.0)

# Favor balanced teams
seeds_balanced = seed_teams(ratings, teams, alpha=0.5)
```

### Top-K Players

Limit team strength calculation to top-K players:

```python
# Only use top 4 players per team
seeds = seed_teams(ratings, teams, use_top_k=4)

# Use all players
seeds = seed_teams(ratings, teams, use_top_k=None)
```

### Exposure Weights

Account for partial participation (e.g., player only played 50% of matches):

```python
exposure_weights = {
    (player_id, team_id): weight,
    (1001, 101): 0.5,  # Player 1001 at 50% exposure
    (1002, 101): 1.0,  # Player 1002 at full exposure
}

seeder = TeamSeeding()
seeds = seeder.compute_all_teams(ratings, teams, exposure_weights)
```

## Advanced Usage

### From DataFrames (Database Query)

```python
from rankings.seedings.team_seeding import TeamSeeding

# Load from database
player_ratings = pl.read_database("SELECT id, score FROM player_rankings", conn)
team_rosters = pl.read_database("SELECT team_id, player_id FROM roster", conn)

# Compute seeding
seeder = TeamSeeding()
seeds = seeder.compute_from_dataframe(player_ratings, team_rosters)
```

### Single Team Strength

```python
seeder = TeamSeeding()
strength = seeder.compute_team_strength(
    player_ratings,
    team_roster=[1, 2, 3, 4],
    exposure_weights={1: 0.8, 2: 1.0, 3: 1.0, 4: 1.0}
)
print(f"Team strength: {strength:.2f}")
```

### Custom Column Names

```python
# If your DataFrame uses different column names
config = TeamSeedingConfig(
    player_id_col="player_id",
    score_col="rating",
    exposure_col="activity"
)
seeder = TeamSeeding(config)
```

## Mathematical Details

### Team Strength Formula

Team strength is calculated using **log-sum-exp** aggregation:

```
R_team = log(Σ w_i * exp(α * r_i))
```

Where:
- `r_i` = player log-rating (from model)
- `w_i` = exposure weight (default 1.0)
- `α` = diminishing returns exponent (default 1.0)

### Why Log-Sum-Exp?

- **Stable**: Numerically stable calculation in log-space
- **Interpretable**: Result is on same scale as player ratings
- **Principled**: Corresponds to probabilistic skill aggregation
- **Flexible**: Supports exposure weighting and top-K filtering

## Integration with Pipeline

### Step 1: Run Rankings

```bash
# Generate player rankings
rankings run --output data/compiled/
```

### Step 2: Load Rankings

```python
import polars as pl

# Load from compiled output
player_ratings = pl.read_parquet("data/compiled/rankings.parquet")

# Or load from database
from sqlalchemy import create_engine
engine = create_engine("postgresql://...")
player_ratings = pl.read_database_uri(
    "SELECT player_id as id, score FROM player_rankings",
    engine
)
```

### Step 3: Define Teams

```python
# From tournament registration
teams = {
    101: [1001, 1002, 1003, 1004],
    102: [1005, 1006, 1007, 1008],
    # ...
}

# Or from database
team_rosters = pl.read_database_uri(
    "SELECT team_id, player_id FROM tournament_rosters WHERE tournament_id = ?",
    engine,
    params=[tournament_id]
)
```

### Step 4: Generate Seeds

```python
from rankings.seedings.team_seeding import seed_teams

# Simple case
seeds = seed_teams(player_ratings, teams, use_top_k=4)

# From DataFrame
from rankings.seedings.team_seeding import TeamSeeding
seeder = TeamSeeding()
seeds = seeder.compute_from_dataframe(player_ratings, team_rosters)
```

### Step 5: Use Seeds

```python
# Export to CSV
seeds.write_csv("tournament_seeds.csv")

# Write to database
seeds.write_database("tournament_seeds", engine, if_exists="replace")

# Use directly
for row in seeds.iter_rows(named=True):
    assign_bracket_seed(
        tournament_id=tournament_id,
        team_id=row["team_id"],
        seed=row["seed_rank"]
    )
```

## Example Workflows

### Tournament Bracket Seeding

```python
# Load data
player_ratings = pl.read_parquet("rankings.parquet")
teams = load_tournament_teams(tournament_id)

# Generate seeds
seeds = seed_teams(player_ratings, teams, use_top_k=4, alpha=1.0)

# Assign to bracket
for team_id, strength, seed in seeds.iter_rows():
    bracket.assign_seed(team_id, seed)
```

### Division Assignment

```python
# Compute all team strengths
seeds = seed_teams(player_ratings, teams, use_top_k=4)

# Assign divisions based on strength
seeds = seeds.with_columns([
    pl.when(pl.col("seed_rank") <= 8).then(pl.lit("Premier"))
      .when(pl.col("seed_rank") <= 24).then(pl.lit("Division 1"))
      .when(pl.col("seed_rank") <= 56).then(pl.lit("Division 2"))
      .otherwise(pl.lit("Division 3"))
      .alias("division")
])
```

### Exposure-Weighted Seeding

```python
# Load roster with playtime data
rosters = pl.DataFrame({
    "team_id": [101, 101, 101, 101],
    "player_id": [1, 2, 3, 4],
    "maps_played": [12, 12, 8, 10],  # Out of 12 total
})

# Calculate exposure weights
rosters = rosters.with_columns(
    (pl.col("maps_played") / 12.0).alias("exposure")
)

# Seed with exposure
seeder = TeamSeeding()
seeds = seeder.compute_from_dataframe(
    player_ratings,
    rosters.select(["team_id", "player_id", "exposure"])
)
```

## See Also

- `examples/team_seeding_example.py` - Complete working examples
- `src/rankings/tournament_prediction/team_strength.py` - Core calculation
- `src/rankings/seedings/entropy_seeding.py` - Advanced entropy-based seeding
- `src/rankings/postprocess/rankings.py` - Model output processing
