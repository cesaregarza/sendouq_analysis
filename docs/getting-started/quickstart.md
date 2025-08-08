# Quick Start Guide

Get up and running with Sendouq Analysis in minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/sendouq/analysis.git
cd sendouq_analysis

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Load Your Data

```python
import polars as pl
from rankings.evaluation.cross_validation import cross_validate_simple

# Load match data
matches_df = pl.read_csv("matches.csv")
players_df = pl.read_csv("players.csv")

# Required columns for matches_df:
# - tournament_id: Tournament identifier
# - last_game_finished_at: Match timestamp
# - winner_player_id: Winner's ID
# - loser_player_id: Loser's ID
```

### 2. Run Cross-Validation

```python
# Simple evaluation with default settings
results = cross_validate_simple(
    matches_df=matches_df,
    players_df=players_df,
    n_splits=5,
    verbose=True
)

print(f"Average loss: {results['avg_loss']:.4f}")
print(f"Concordance: {results['concordance']:.4f}")
```

### 3. Optimize Parameters

```python
from rankings.evaluation.optimizer import optimize_rating_engine

# Find best hyperparameters
best_params = optimize_rating_engine(
    matches_df=matches_df,
    players_df=players_df,
    method="grid",
    param_space={
        "decay_half_life_days": [7, 14, 30, 60],
        "damping_factor": [0.8, 0.85, 0.9]
    }
)

print(f"Best parameters: {best_params['best_params']}")
```

## Complete Example

Here's a full working example:

```python
import polars as pl
from rankings.evaluation.cross_validation import cross_validate_ratings
from rankings.evaluation.optimizer import optimize_rating_engine
from rankings.analysis.engine import RatingEngine

# 1. Load data
matches_df = pl.read_csv("data/matches.csv")
players_df = pl.read_csv("data/players.csv")

# 2. Quick evaluation
print("Running initial evaluation...")
initial_results = cross_validate_simple(
    matches_df=matches_df,
    players_df=players_df,
    n_splits=3,  # Fewer splits for speed
    verbose=True
)

# 3. Parameter optimization
print("\nOptimizing parameters...")
optimization_results = optimize_rating_engine(
    matches_df=matches_df,
    players_df=players_df,
    method="grid",
    param_space={
        "decay_half_life_days": [14, 30, 60],
        "damping_factor": [0.85, 0.9],
        "beta": [0.0, 0.5, 1.0]
    },
    n_splits=5
)

best_params = optimization_results["best_params"]
print(f"\nBest parameters found: {best_params}")

# 4. Full evaluation with best parameters
print("\nRunning full evaluation with best parameters...")
final_results = cross_validate_ratings(
    matches_df=matches_df,
    players_df=players_df,
    engine_params=best_params,
    n_splits=5,
    compute_extras=True,  # Get all metrics
    verbose=True
)

# 5. Display comprehensive results
print("\n=== Final Results ===")
print(f"Log Loss: {final_results['avg_loss']:.4f} ± {final_results['std_loss']:.4f}")
print(f"Concordance: {final_results['concordance']:.4f}")
print(f"Skill Score: {final_results['skill_score']:.4f}")
print(f"Calibration (70%): {final_results['accuracy_70']:.4f}")
print(f"Upset O/E: {final_results['upset_oe']:.3f}")

# 6. Train final model
print("\nTraining final model...")
engine = RatingEngine(**best_params)
engine.fit(matches_df, players_df)

# Get current ratings
current_ratings = engine.get_current_ratings()
print(f"\nTop 10 players:")
for i, (player_id, rating) in enumerate(current_ratings.head(10).items()):
    print(f"{i+1}. Player {player_id}: {rating:.1f}")
```

## Data Format Requirements

### Matches DataFrame

Required columns:
- `tournament_id` (int): Tournament identifier
- `last_game_finished_at` (datetime): Match timestamp
- `winner_player_id` (int): Winner's player ID
- `loser_player_id` (int): Loser's player ID

Optional columns:
- `winner_team_id` (int): Winner's team ID
- `loser_team_id` (int): Loser's team ID
- `stage` (str): Tournament stage (pools, bracket, finals)
- `match_importance` (float): Custom importance weight

### Players DataFrame

Required columns:
- `player_id` (int): Player identifier

Optional columns:
- `name` (str): Player name
- `region` (str): Player region
- `team_id` (int): Current team

### Teams DataFrame (Optional)

Required columns:
- `team_id` (int): Team identifier

Optional columns:
- `name` (str): Team name
- `roster` (list): List of player IDs

## Common Workflows

### 1. Evaluating a Custom Rating System

```python
from my_module import MyCustomEngine

# Evaluate custom engine
results = cross_validate_simple(
    matches_df=matches_df,
    engine_class=MyCustomEngine,
    engine_params={"my_param": 42},
    n_splits=5
)
```

### 2. Team-Based Predictions

```python
# Rank players but predict on teams
results = cross_validate_ratings(
    matches_df=matches_df,
    players_df=players_df,
    teams_df=teams_df,
    ranking_entity="player",    # Rank individual players
    prediction_entity="team",   # But predict team outcomes
    agg_func="mean"            # Average player ratings for teams
)
```

### 3. Temporal Analysis

```python
from rankings.evaluation.cross_validation import create_time_based_folds

# Analyze performance over time
folds = create_time_based_folds(matches_df, n_splits=10)

fold_performances = []
for i, (train, test, _) in enumerate(folds):
    # Train on historical data
    engine = RatingEngine()
    engine.fit(train)
    
    # Evaluate on future data
    loss = evaluate_on_test(engine, test)
    fold_performances.append({
        "fold": i,
        "test_date": test["last_game_finished_at"].max(),
        "loss": loss
    })

# Plot temporal trends
plot_temporal_performance(fold_performances)
```

## Tips for Better Results

1. **Data Quality**
   - Ensure consistent player/team IDs
   - Remove duplicate matches
   - Handle missing timestamps

2. **Parameter Tuning**
   - Start with coarse grids
   - Refine around best values
   - Consider computational cost

3. **Evaluation Strategy**
   - Use more splits for final evaluation
   - Check temporal stability
   - Validate on recent tournaments

## Next Steps

- [Complete Examples](examples.md) - More detailed workflows
- [API Reference](../api/cross-validation.md) - Full function documentation
- [Advanced Topics](../advanced/custom-engines.md) - Build custom rating systems