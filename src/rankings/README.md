# Sendou.ink Tournament Rankings

This module provides comprehensive tournament ranking capabilities for Sendou.ink data.

Key engines:
- **ExposureLogOddsEngine (Recommended)**: log-odds PageRank on mirrored graphs that removes volume bias ("grinding") and focuses on conversion quality.
- **RatingEngine (Core)**: tick-tock PageRank with tournament strength modeling.

## Current Status (v0.2.0)

### Recent Updates
- **Bradley-Terry Probability Model**: Implemented natural probability model `P(A beats B) = s_A / (s_A + s_B)` replacing legacy logistic model
- **Improved Alpha Bounds**: Updated from extreme values (1e3-1e5) to reasonable range (0.1-5.0)
- **Enhanced Evaluation Suite**: 7 orthogonal metrics including discrimination, calibration, ranking order, confidence performance, and stability indicators

### Known Issues & In-Progress Improvements
Based on test_tour_7.ipynb analysis, the current loss function needs improvements:
1. **Poor Calibration**: Low accuracy (~10%) and poor upset O/E ratio (~0.2) indicate probability calibration issues
2. **Unknown Player Handling**: Current approach injects 0.0 ratings for unknown players, causing extreme probabilities
3. **Incomplete Roster Handling**: Teams with many unrated players skew predictions
4. **Loss Function Optimization**: Need better weighting scheme and aggregation methods

### Planned Fixes (from plan.md)
- Implement global prior for unknown players (5th percentile of ratings)
- Add row-skipping rule for teams with ≥2 unrated players
- Switch to log-centered aggregation for team ratings
- Implement entropy-weighted loss as primary objective
- Add bye/forfeit match filtering
- Improve teleport vector defaults

## Features

### Core Functionality
- **Tournament Data Parsing**: Parse Sendou.ink JSON exports into structured DataFrames
- **Team & Player Rankings**: Rank both teams and individual players
- **Time Decay**: Exponential decay weighting for match age
- **Tournament Strength**: Dynamic tournament importance calculation
- **Multiple Algorithms**: Basic PageRank and advanced tick-tock algorithm

### Algorithms

#### 1. Basic PageRank (`basic_rankings.py`)
- Straightforward PageRank implementation
- Optional tournament strength weighting based on participant count
- Fast and simple for most use cases

#### 2. Exposure Log-Odds Engine (`analysis/engine/exposure_logodds.py`)
- Removes volume bias by computing separate win/loss PageRanks with the same exposure-based teleport, then taking smoothed log-ratios
- Optional surprisal weighting to reward upsets
- Time decay and tournament strength weighting via inherited parameters
- Outputs win PR, loss PR, exposure, and log-odds score

#### 3. Advanced Rating Engine (`analysis/engine/core.py`)
- Tick-tock algorithm that iteratively refines ratings and tournament strengths
- Tournament influence calculated based on participant skill levels
- Multiple aggregation methods for tournament strength
- Configurable teleport vectors and decay models

## Quick Start

```python
import json
from rankings import parse_tournaments_data
from rankings.analysis.engine.exposure_logodds import ExposureLogOddsEngine

# Load and parse tournament data
with open("tournament_data.json") as f:
    raw_data = json.load(f)

tables = parse_tournaments_data(raw_data)
matches_df = tables["matches"]
players_df = tables["players"]

# Recommended: Exposure Log-Odds player rankings (volume-bias free)
engine = ExposureLogOddsEngine(beta=1.0)
rankings = engine.rank_players(matches_df, players_df)

# Optionally post-process (min tournaments, grades, display score)
final = engine.post_process_rankings(
    rankings,
    players_df,
    min_tournaments=3,
)

# Access tournament influence/strength computed during initialization run
tournament_influence = engine.tournament_influence
tournament_strength = engine.tournament_strength

# Core alternative: tick-tock engine
# from rankings import RatingEngine
# engine = RatingEngine(beta=1.0, influence_agg_method="top_20_sum")
# rankings = engine.rank_players(matches_df, players_df)
```

## API Reference

### Data Parsing

#### `parse_tournaments_data(tournaments: list[dict]) -> dict`
Parses Sendou.ink tournament JSON into structured DataFrames.

**Returns:**
- `stages`: Tournament stages information
- `groups`: Tournament groups  
- `rounds`: Tournament rounds
- `teams`: Team information and rosters
- `players`: Individual player records
- `matches`: Match results with winners/losers

### Basic Rankings

#### `rank_teams_basic(matches_df, **kwargs) -> pl.DataFrame`
Basic PageRank team rankings.

#### `rank_players_basic(matches_df, players_df, **kwargs) -> pl.DataFrame`
Basic PageRank player rankings.

#### `rank_teams_with_strength(matches_df, teams_df, base_weight=1.0, **kwargs) -> pl.DataFrame`
Enhanced team rankings with tournament strength weighting.

#### `rank_players_with_strength(matches_df, players_df, teams_df, base_weight=1.0, **kwargs) -> pl.DataFrame`
Enhanced player rankings with tournament strength weighting.

### Exposure Log-Odds Engine (Recommended)

#### `class ExposureLogOddsEngine(RatingEngine)`

Inherits core parameters (time decay, damping, beta, influence aggregation) and adds:

**Key Parameters:**
- `lambda_smooth`: Optional smoothing tied to exposure teleport (auto-tuned if None)
- `use_surprisal`: Whether to add upset-aware weighting
- `surprisal_T`: Temperature for surprisal
- `surprisal_iters`: Iterations of surprisal refinement
- `min_exposure`: Minimum exposure to be included in final rankings
- `score_decay_delay_days` / `score_decay_rate`: Post-ranking inactivity decay

**Methods:**
- `rank_players(matches_df, players_df) -> pl.DataFrame`: Columns: `id`, `player_rank`, `win_pr`, `loss_pr`, `exposure`
- `rank_teams(matches_df) -> pl.DataFrame`: Columns: `id`, `team_rank`, `win_pr`, `loss_pr`, `exposure`
- `post_process_rankings(rankings, players_df, ...) -> pl.DataFrame`: Grades and display scores

**Stored Attributes (after run):**
- `win_pagerank_`, `loss_pagerank_`, `exposure_teleport_`, `logodds_scores_`, `lambda_used_`

### Core Engine (Alternative)

#### `class RatingEngine`

Tick-tock PageRank with tournament strength modeling.

## Configuration

### Time Decay
Control how much recent matches matter vs historical ones:
```python
# Recent matches matter more (shorter half-life)
engine = RatingEngine(decay_half_life_days=15.0)

# Historical matches remain relevant longer
engine = RatingEngine(decay_half_life_days=60.0)
```

### Tournament Strength
Control how much tournament "strength" affects ratings:
```python
# No tournament strength weighting
engine = RatingEngine(beta=0.0)

# Moderate tournament strength weighting  
engine = RatingEngine(beta=0.5)

# Full tournament strength weighting
engine = RatingEngine(beta=1.0)
```

### Influence Aggregation
How to compute tournament strength from participant ratings:
```python
# Mean skill of all participants
engine = RatingEngine(influence_agg_method="mean")

# Sum of top 20 participant ratings
engine = RatingEngine(influence_agg_method="top_20_sum")
```

## Utilities

The `utils.py` module provides helper functions:

- `prepare_player_summary()`: Format player rankings with statistics
- `prepare_team_summary()`: Format team rankings with statistics  
- `prepare_tournament_summary()`: Tournament metadata with strength metrics
- `format_top_rankings()`: Pretty-print top rankings
- `compare_rankings()`: Compare two ranking systems
- `get_head_to_head_record()`: Head-to-head statistics between entities

## Example Output

```
Top 10 Players (Advanced Engine):
 1. Grey                  100.0 (69 tournaments)
 2. Jared                  99.0 (20 tournaments)
 3. Noah                   96.9 (78 tournaments)
 4. phoenix                91.4 (227 tournaments)
 5. Crowny                 80.8 (187 tournaments)
 6. .q                     76.7 (29 tournaments)
 7. Silver                 76.7 (181 tournaments)
 8. kiki                   72.4 (311 tournaments)
 9. bran                   66.1 (14 tournaments)
10. Aaron                  65.2 (54 tournaments)

Tournament Analysis:
Top 5 Strongest Tournaments:
1. SendouQ Season 7 Finale         Influence: 4.88, Strength: 6.77
2. Barnacle Bash #7                Influence: 4.01, Strength: 5.91
3. Fry Basket Invitational #1      Influence: 3.99, Strength: 5.41
4. Barnacle Bash #6.5              Influence: 4.66, Strength: 5.14
5. LUTI: Season 16 - Division X    Influence: 3.98, Strength: 5.07
```

## Algorithm Details

### Basic Algorithm
1. Build weighted directed graph from match results
2. Apply exponential time decay to edge weights
3. Optionally weight by tournament strength (based on team count)
4. Run PageRank to get final rankings

### Advanced Algorithm (Tick-Tock)
1. **Initialize**: Set all tournament strengths to 1.0
2. **Tick**: Compute player/team ratings using current tournament strengths
3. **Tock**: Recompute tournament strengths based on participant skill levels
4. **Repeat**: Until tournament strengths converge
5. **Output**: Final ratings and retrospective tournament strength metrics

The tick-tock algorithm produces mutually consistent ratings where strong tournaments are those with strong participants, and strong participants are those who perform well in strong tournaments.

## Performance Notes

- Basic algorithms are fast and suitable for real-time applications
- Advanced engine is more computationally intensive but provides richer insights
- Both scale well to thousands of players and hundreds of tournaments
- Polars DataFrames provide efficient memory usage and fast operations

## Dependencies

- `polars`: Fast DataFrame library
- `numpy`: Numerical computations for PageRank
- `zoneinfo`: Timezone handling for time decay
- `requests`: HTTP requests for API interactions
- `tqdm`: Progress bars for batch operations
- `scipy`: Scientific computing (optimization and statistics)

## Utility Functions

The module provides several convenience functions for enhancing data with human-readable names and timestamps:

### Name Resolution Functions
- `add_player_names()`: Add player usernames to DataFrames with user IDs
- `add_team_names()`: Add team names to DataFrames with team IDs  
- `add_tournament_names()`: Add tournament names from raw JSON data
- `add_match_timestamps()`: Convert Unix timestamps to readable dates

### Analysis Functions
- `get_most_influential_matches()`: Find matches with highest impact on a player's ranking
- `get_player_match_history()`: Get complete match history with names and context
- `create_match_summary_with_names()`: Create comprehensive match summaries
- `prepare_player_summary()`: Create player rankings with statistics
- `prepare_tournament_summary()`: Create tournament summaries with strength metrics

### Example Usage
```python
# Add names to rankings
rankings_with_names = add_player_names(rankings_df, players_df)

# Get influential matches with timestamps
influential = get_most_influential_matches(player_id=123, ...)
wins_with_names = add_tournament_names(influential["wins"], tournament_data)
wins_with_dates = add_match_timestamps(wins_with_names)

# Get full match history
history = get_player_match_history(
    player_id=123,
    matches_df=matches_df,
    players_df=players_df,
    teams_df=teams_df,
    tournament_data=raw_data,
    limit=50
)
```

## Files

### Core Module Structure
- `core/`
  - `__init__.py`: Core module exports
  - `parser.py`: Tournament data parsing from JSON
  - `constants.py`: Configuration constants and defaults

### Scraping Module
- `scraping/`
  - `__init__.py`: Scraping module exports
  - `api.py`: Sendou.ink API interface
  - `batch.py`: Batch scraping operations
  - `discovery.py`: Tournament discovery via calendar
  - `storage.py`: Data persistence utilities

### Analysis Module  
- `analysis/`
  - `__init__.py`: Analysis module exports
  - `engine.py`: Advanced RatingEngine with tick-tock algorithm
  - `utils.py`: Analysis utilities and convenience functions

- `README.md`: This documentation 