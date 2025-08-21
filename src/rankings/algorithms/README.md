# Rankings Algorithms

This module contains the core ranking algorithm implementations for the Sendouq analysis system. It provides multiple approaches for computing player and team rankings from tournament match data.

## Overview

The algorithms module implements various ranking engines that transform match results into numerical ratings:

- **ExposureLogOddsEngine**: The recommended production engine using exposure-weighted log-odds ratios
- **TickTockEngine**: The core iterative PageRank-based engine with tournament strength feedback
- **TTLEngine**: Experimental hybrid engine combining tick-tock with log-odds (in development)
- **Backends**: Low-level computation backends for the tick-tock orchestration

All engines follow a consistent interface and can be used interchangeably depending on your requirements.

## Algorithms

### ExposureLogOddsEngine

The primary production engine that uses exposure-weighted log-odds for robust ranking:

```python
from rankings.algorithms import ExposureLogOddsEngine
from rankings.core import ExposureLogOddsConfig

config = ExposureLogOddsConfig(
    damping_factor=0.85,
    tolerance=1e-6,
    max_iterations=100,
    lambda_param=None,  # Auto-tuned if not specified
    inactivity_days=30,
    inactivity_decay=0.95
)

engine = ExposureLogOddsEngine(config)
rankings = engine.rank_players(matches_df, players_df)
```

**Key Features:**
- Exposure-based teleport vectors (sum of match pair 'share' per player)
- Dual PageRank computation with consistent teleport vectors
- Auto-tuned lambda parameter (~2.5% of median PageRank per node)
- Optional time-decay for inactive players
- Robust to sparse data and outliers

**Output Schema:**
- `id`: Player or team identifier
- `player_rank`: Computed ranking score
- Additional metadata (exposure, match counts, etc.)

### TickTockEngine

The core iterative engine using PageRank with tournament strength feedback:

```python
from rankings.algorithms import TickTockEngine
from rankings.core import TickTockConfig

config = TickTockConfig(
    convergence_tol=1e-4,
    max_iterations=50,
    beta=1.0,  # Tournament strength exponent
    influence_agg_method="top_20_sum",
    teleport_type="uniform"  # or "volume_inverse"
)

engine = TickTockEngine(config)
rankings = engine.rank_players(matches_df, players_df)
```

**Key Features:**
- Iterative "tick-tock" between PageRank and tournament strength
- Multiple tournament strength aggregation methods
- Configurable teleport vector strategies
- Mean-normalized influences for stability
- Tight convergence control (default 1e-4)

**Aggregation Methods:**
- `"top_20_sum"`: Sum of top 20 player ratings
- `"log_top_20_sum"`: Log-transformed sum of top 20 ratings (log1p after normalization) **Recommended**
- `"sqrt_top_20_sum"`: Square root of top 20 sum
- `"top_10_sum"`: Sum of top 10 player ratings
- `"top_20_mean"`: Average of top 20 player ratings
- `"mean"` or `"arithmetic"`: Average rating of all players
- `"sum"`: Sum of all player ratings
- `"median"`: Median rating

### TTLEngine (Experimental)

Tick-Tock over Log-Odds (TTL) engine - an experimental hybrid approach:

```python
from rankings.algorithms import TTLEngine
from rankings.algorithms.backends import LogOddsBackend

# Uses log-odds backend by default
engine = TTLEngine(
    config=ExposureLogOddsConfig(
        tick_tock=TickTockConfig(
            max_ticks=20,
            convergence_tol=1e-4,
            influence_method="log_top_20_sum"
        )
    )
)

rankings = engine.rank_players(matches_df, players_df)
```

**⚠️ Note**: TTLEngine is still in development and currently produces worse results than the standard ExposureLogOddsEngine. Use ExposureLogOddsEngine for production.

**Key Features:**
- **Outer loop**: Tick-tock iteration for tournament influence (anti-gaming)
- **Inner loop**: Exposure log-odds computation for volume-neutral ratings
- Uses `quality_mass` from log-odds instead of win-based PageRank
- Attempts to combine volume neutrality with gaming resistance
- Damped influence updates for stability (default μ=0.5)
- Theoretical advantages not yet realized in practice

## Backends

The `backends` submodule provides low-level computation engines:

### LogOddsBackend

Computes ratings using log-odds ratios from win/loss statistics:

```python
from rankings.algorithms.backends import LogOddsBackend

backend = LogOddsBackend(smoothing_factor=1.0)
ratings = backend.compute(edges_df, teleport_vector)
```

### RowPRBackend

Row-stochastic PageRank implementation for comparison:

```python
from rankings.algorithms.backends import RowPRBackend

backend = RowPRBackend(damping_factor=0.85, tolerance=1e-6)
ratings = backend.compute(transition_matrix, teleport_vector)
```

## Usage Examples

### Basic Player Rankings

```python
from rankings.core import parse_tournaments_data
from rankings.algorithms import ExposureLogOddsEngine

# Load and parse tournament data
tables = parse_tournaments_data(tournaments)

# Create engine and compute rankings
engine = ExposureLogOddsEngine()
player_rankings = engine.rank_players(
    tables["matches"], 
    tables["players"]
)

# Display top players
top_players = player_rankings.sort("player_rank", descending=True).head(20)
print(top_players)
```

### Team Rankings with Custom Configuration

```python
from rankings.algorithms import TickTockEngine
from rankings.core import TickTockConfig

config = TickTockConfig(
    beta=1.5,  # Higher weight on tournament strength
    influence_agg_method="log_top_20_sum",  # Log-transformed top 20
    convergence_tol=1e-5
)

engine = TickTockEngine(config)
team_rankings = engine.rank_teams(tables["matches"])
```

### Comparing Multiple Algorithms

```python
from rankings.algorithms import ExposureLogOddsEngine, TickTockEngine

# Compute with both engines
elo_engine = ExposureLogOddsEngine()
elo_rankings = elo_engine.rank_players(matches, players)

tt_engine = TickTockEngine()
tt_rankings = tt_engine.rank_players(matches, players)

# Compare results
comparison = elo_rankings.join(
    tt_rankings, 
    on="id", 
    suffix="_tt"
).select([
    "id",
    "player_rank",
    "player_rank_tt",
    (pl.col("player_rank") - pl.col("player_rank_tt")).alias("diff")
])
```

## Configuration

All engines support extensive configuration through their respective config classes:

- **ExposureLogOddsConfig**: Controls PageRank parameters, lambda tuning, and decay
- **TickTockConfig**: Sets convergence criteria, aggregation methods, and teleport strategies
- **TTLConfig**: Defines activity windows and decay parameters

See `rankings.core.config` for complete configuration options.

## Performance Considerations

- All engines use sparse matrix operations for efficiency
- Polars DataFrames provide vectorized operations
- Convergence tolerances can be adjusted for speed vs. accuracy
- The ExposureLogOddsEngine is generally fastest for large datasets
- TickTockEngine provides more interpretable tournament strength metrics

## Algorithm Selection Guide

Choose **ExposureLogOddsEngine** when:
- You need production-ready, robust rankings (**Recommended for production**)
- Dataset has varying tournament sizes and participation
- Auto-tuning of parameters is desired
- Volume neutrality is important
- Good balance of speed and accuracy

Choose **TickTockEngine** when:
- You need explicit tournament strength metrics
- Interpretability of the ranking process is important
- You want fine control over aggregation methods
- You're comparing with legacy systems
- Tournament gaming resistance is more important than volume neutrality

Choose **TTLEngine** when:
- You want to experiment with hybrid approaches
- Testing new ranking methodologies
- Research and development purposes
- **Note**: Currently not recommended for production use

## Extending the Algorithms

To add a new ranking algorithm:

1. Create a new class inheriting from a base engine interface
2. Implement `rank_players()` and/or `rank_teams()` methods
3. Add configuration class in `rankings.core.config`
4. Export from `__init__.py`

Example structure:
```python
class MyCustomEngine:
    def __init__(self, config=None):
        self.config = config or MyCustomConfig()
    
    def rank_players(self, matches_df, players_df):
        # Implementation
        return rankings_df
```

## See Also

- `rankings.core`: Core utilities and data structures
- `rankings.evaluation`: Metrics and validation tools
- `rankings.analysis`: Analysis and visualization utilities