# Rankings Core

The core module provides foundational components for the rankings system, including data parsing, configuration management, algorithm building blocks, and utility functions.

## Overview

This module is the foundation of the ranking system, providing:

- **Data Parsing**: Convert Sendou.ink tournament JSON into structured Polars DataFrames
- **Configuration**: Centralized configuration classes for all ranking engines
- **Algorithm Components**: PageRank, teleport strategies, smoothing methods, and edge building
- **Time Utilities**: Decay calculations, time windows, and activity tracking
- **Results**: Structured result types for engines and pipelines
- **Logging**: Consistent logging and performance monitoring

The core module is engine-agnostic and used by all ranking algorithms.

## Key Components

### Data Parser

The parser normalizes Sendou.ink tournament data into structured tables:

```python
from rankings.core import parse_tournaments_data

tables = parse_tournaments_data(tournaments)
```

Returns a dictionary with up to 7 DataFrames:
- `tournaments`: Tournament metadata and settings
- `stages`: Stage configuration per tournament
- `groups`: Group structure within stages
- `rounds`: Round-level information
- `teams`: Team rosters and attributes
- `players`: Player details per team/tournament
- `matches`: Normalized match outcomes

See the Data Model section below for detailed schemas.

### Configuration System

Comprehensive configuration classes for all engines:

```python
from rankings.core import (
    ExposureLogOddsConfig,
    TickTockConfig,
    PageRankConfig,
    DecayConfig
)

# Configure exposure log-odds engine
elo_config = ExposureLogOddsConfig(
    damping_factor=0.85,
    tolerance=1e-6,
    lambda_param=None,  # Auto-tuned
    inactivity_days=30
)

# Configure tick-tock engine
tt_config = TickTockConfig(
    beta=1.0,
    convergence_tol=1e-4,
    influence_agg_method="top_20_sum"
)
```

### PageRank Implementations

Efficient PageRank computations for ranking algorithms:

```python
from rankings.core import pagerank_sparse, pagerank_dense

# Sparse implementation (recommended for large graphs)
ratings = pagerank_sparse(
    edges,
    damping_factor=0.85,
    teleport=teleport_vector,
    tolerance=1e-6
)

# Dense implementation (for small graphs or testing)
ratings = pagerank_dense(
    transition_matrix,
    damping_factor=0.85,
    teleport=teleport_vector
)
```

### Teleport Strategies

Multiple teleport vector strategies for PageRank:

```python
from rankings.core import (
    UniformTeleport,
    VolumeInverseTeleport,
    ActivePlayersTeleport,
    CustomTeleport
)

# Uniform teleport (equal probability)
teleport = UniformTeleport()
vector = teleport.compute(num_nodes)

# Volume-inverse (bias against high-volume players)
teleport = VolumeInverseTeleport()
vector = teleport.compute(match_counts)

# Active players only
teleport = ActivePlayersTeleport(min_matches=5)
vector = teleport.compute(player_activity)
```

### Smoothing Strategies

Edge weight smoothing for handling sparse data:

```python
from rankings.core import (
    WinsProportional,
    ConstantSmoothing,
    AdaptiveSmoothing,
    HybridSmoothing
)

# Proportional to wins
smoother = WinsProportional(factor=1.0)
smoothed_edges = smoother.apply(edges)

# Adaptive based on data density
smoother = AdaptiveSmoothing(min_factor=0.5, max_factor=2.0)
smoothed_edges = smoother.apply(edges, density_metrics)
```

### Time and Decay Utilities

Handle temporal aspects of rankings:

```python
from rankings.core import (
    Clock,
    compute_decay_factor,
    apply_inactivity_decay,
    create_time_windows
)

# Fixed time for reproducibility
clock = Clock(now_ts=1234567890.0)

# Compute exponential decay
decay = compute_decay_factor(
    days_elapsed=30,
    half_life=30
)

# Apply inactivity penalties
rankings = apply_inactivity_decay(
    rankings_df,
    last_activity_df,
    inactivity_days=30,
    decay_rate=0.95
)

# Create time windows for analysis
windows = create_time_windows(
    start_date="2024-01-01",
    end_date="2024-12-31",
    window_size_days=30
)
```

### Edge Building

Convert match results into graph edges:

```python
from rankings.core import (
    build_player_edges,
    build_team_edges,
    build_exposure_triplets,
    normalize_edges
)

# Build player-vs-player edges
player_edges = build_player_edges(
    matches_df,
    players_df,
    decay_config
)

# Build team-vs-team edges
team_edges = build_team_edges(
    matches_df,
    weight_method="score_differential"
)

# Build exposure triplets for log-odds
triplets = build_exposure_triplets(
    matches_df,
    players_df
)

# Normalize edge weights
normalized = normalize_edges(edges, method="sum_to_one")
```

### Tournament Influence

Compute tournament strength metrics:

```python
from rankings.core import (
    compute_tournament_influence,
    aggregate_multi_round_influence,
    normalize_influence
)

# Compute influence from current ratings
influence = compute_tournament_influence(
    tournament_players,
    current_ratings,
    aggregation_method="top_20_sum"
)

# Multi-round aggregation
final_influence = aggregate_multi_round_influence(
    round_influences,
    decay_factor=0.9
)

# Normalize to mean=1.0
normalized = normalize_influence(influence)
```

## Data Model

### Tournaments Table
- `tournament_id`: Unique tournament identifier
- `event_id`: Associated event ID
- `name`, `description`: Tournament details
- `start_time`, `is_finalized`: Timing and status
- `team_count`, `match_count`: Participation metrics
- Various metadata fields for rules, settings, staff, etc.

### Matches Table
- `tournament_id`, `stage_id`, `group_id`, `round_id`, `match_id`: Full hierarchy
- `team1_id`, `team2_id`: Competing teams
- `team1_score`, `team2_score`: Match scores
- `winner_team_id`, `loser_team_id`: Derived outcomes
- `score_diff`, `total_games`: Computed metrics
- `is_bye`: Bye/forfeit indicator
- `last_game_finished_at`, `match_created_at`: Timestamps

### Players Table
- `tournament_id`, `team_id`: Association keys
- `user_id`, `username`: Player identifiers
- `discord_id`, `in_game_name`: Additional IDs
- `country`, `twitch`: Profile information
- `is_owner`: Team ownership flag
- `roster_created_at`: Timestamp

### Teams Table
- `tournament_id`, `team_id`: Identifiers
- `team_name`: Display name
- `seed`: Initial seeding
- `dropped_out`, `no_screen`: Status flags
- `created_at`: Registration time

## Logging and Monitoring

Unified logging across all components:

```python
from rankings.core.logging import (
    setup_logging,
    get_logger,
    log_timing,
    log_dataframe_stats,
    ProgressLogger
)

# Setup logging
setup_logging(level="INFO", format="detailed")
logger = get_logger(__name__)

# Time operations
with log_timing(logger, "compute rankings"):
    rankings = engine.rank_players(matches, players)

# Log DataFrame statistics
log_dataframe_stats(logger, matches_df, "match data")

# Progress tracking for long operations
with ProgressLogger(logger, total=1000) as progress:
    for item in items:
        process(item)
        progress.update(1)
```

## Result Types

Structured result classes for type safety:

```python
from rankings.core import (
    RankResult,
    TickTockResult,
    ExposureLogOddsResult,
    ValidationResult,
    BenchmarkResult
)

# Engine results include rankings and metadata
result = ExposureLogOddsResult(
    rankings=rankings_df,
    metadata={
        "lambda": 0.025,
        "iterations": 47,
        "convergence": 8.3e-7
    }
)

# Validation results for metrics
validation = ValidationResult(
    metrics={"accuracy": 0.85, "log_loss": 0.42},
    predictions=predictions_df,
    confusion_matrix=confusion
)
```

## Constants and Defaults

Centralized configuration defaults:

```python
from rankings.core.constants import (
    DEFAULT_DECAY_HALF_LIFE_DAYS,  # 30
    DEFAULT_DAMPING_FACTOR,  # 0.85
    DEFAULT_PAGERANK_TOLERANCE,  # 1e-6
    DEFAULT_BETA,  # 1.0
    MIN_TOURNAMENTS_FOR_RANKING,  # 3
    MIN_MATCHES_FOR_EDGE,  # 1
)
```

## Performance Optimization

The core module is optimized for performance:

- **Polars DataFrames**: Columnar storage and vectorized operations
- **Sparse Matrices**: Efficient graph representations via scipy.sparse
- **Lazy Evaluation**: Deferred computation where possible
- **Caching**: Reusable intermediate results
- **Parallel Processing**: Multi-threaded operations in Polars

## Common Usage Patterns

### Basic Pipeline

```python
from rankings.core import parse_tournaments_data
from rankings.algorithms import ExposureLogOddsEngine

# Parse data
tables = parse_tournaments_data(tournament_json)

# Configure and run engine
engine = ExposureLogOddsEngine()
rankings = engine.rank_players(
    tables["matches"],
    tables["players"]
)
```

### Custom Configuration

```python
from rankings.core import (
    ExposureLogOddsConfig,
    DecayConfig,
    VolumeInverseTeleport
)

# Custom decay
decay = DecayConfig(
    half_life_days=14,
    reference_date="2024-01-01"
)

# Custom teleport
teleport = VolumeInverseTeleport(epsilon=1e-4)

# Combined config
config = ExposureLogOddsConfig(
    decay_config=decay,
    teleport_strategy=teleport,
    damping_factor=0.9
)
```

### Time-Window Analysis

```python
from rankings.core import create_time_windows, filter_by_recency

# Create monthly windows
windows = create_time_windows(
    start="2024-01-01",
    end="2024-12-31",
    window_size_days=30
)

# Analyze each window
for window in windows:
    window_matches = filter_by_recency(
        matches_df,
        window.start,
        window.end
    )
    rankings = engine.rank_players(window_matches, players)
    save_rankings(rankings, window.label)
```

## Extending Core

To add new core functionality:

1. **New Parser Fields**: Extend row dictionaries in `parser.py`
2. **New Config Options**: Add to relevant config class in `config.py`
3. **New Constants**: Define in `constants.py` with clear documentation
4. **New Strategies**: Implement protocol interface (e.g., `TeleportStrategy`)
5. **New Utilities**: Add to appropriate module with tests

## Testing

Core components have comprehensive test coverage:

```bash
# Run core tests
pytest tests/test_core/

# Test specific component
pytest tests/test_core/test_pagerank.py

# Performance benchmarks
pytest tests/benchmarks/test_core_performance.py
```

## Common Pitfalls

- **Byes/Forfeits**: Filter `is_bye=True` matches in analysis
- **Missing Winners**: Some matches have `winner_team_id=None`
- **ID Types**: User IDs may be strings in synthetic data
- **Time Zones**: Ensure consistent timezone handling
- **Empty Tables**: Parser returns `None` for empty tables

## See Also

- `rankings.algorithms`: Ranking engine implementations
- `rankings.evaluation`: Metrics and validation
- `rankings.analysis`: Analysis and visualization tools
- `rankings.continuous`: Continuous ranking updates