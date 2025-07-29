# Tournament Rankings Evaluation Module

This module implements the cross-validated loss function and optimization framework described in `plan.md`. It provides tools for evaluating and optimizing tournament rating systems using probabilistic predictions.

## Key Components

### 1. Loss Functions (`loss.py`)
- **Cross-entropy loss**: Measures how well ratings predict match outcomes
- **Match probability**: Sigmoid transformation of rating differences
- **Tournament aggregation**: Weighted average loss across matches
- **Alpha fitting**: Maximum likelihood estimation of temperature parameter

### 2. Cross-Validation (`cross_validation.py`)
- **Time-based splits**: Rolling-origin CV respecting temporal order
- **Frozen look-back**: No data leakage from future tournaments
- **Split evaluation**: Complete train/test evaluation pipeline
- **Full CV**: Aggregate results across multiple splits

### 3. Optimization (`optimizer.py`)
- **Grid Search**: Exhaustive search over discrete parameter combinations
- **Bayesian Optimization**: Intelligent search for continuous parameters
- **Regularization**: L2 penalty on deviations from defaults
- **Results tracking**: Detailed logging of all evaluations

### 4. Diagnostic Metrics (`metrics.py`)
- **Brier Score**: Calibration of probabilistic predictions
- **Accuracy**: Simple classification performance
- **Spearman Correlation**: Rating vs final placement correlation
- **Round Analysis**: Performance breakdown by tournament stage

## Usage Example

```python
from rankings import (
    RatingEngine,
    parse_tournaments_data,
    optimize_rating_engine,
    cross_validate_ratings
)

# Load and parse tournament data
tournaments = load_scraped_tournaments("data/tournaments")
tables = parse_tournaments_data(tournaments)

# Quick evaluation with default parameters
cv_results = cross_validate_ratings(
    engine_class=RatingEngine,
    engine_params={
        "decay_half_life_days": 30.0,
        "damping_factor": 0.85,
        "beta": 1.0
    },
    matches_df=tables["matches"],
    teams_df=tables["teams"],
    n_splits=5
)

print(f"Average loss: {cv_results['avg_loss']:.4f}")

# Full hyperparameter optimization
best_params = optimize_rating_engine(
    matches_df=tables["matches"],
    teams_df=tables["teams"],
    method="grid",
    param_space={
        "decay_half_life_days": [7, 14, 30, 60],
        "damping_factor": [0.8, 0.85, 0.9],
        "beta": [0.0, 0.5, 1.0]
    }
)

print(f"Best parameters: {best_params['best_params']}")
```

## Key Concepts

### Loss Function
The loss is based on the Bernoulli negative log-likelihood:
```
ℓ = -y·log(p) - (1-y)·log(1-p)
```
where `p` is the predicted probability and `y` is the actual outcome.

### Probability Model
Match outcomes are modeled using a sigmoid function:
```
P(A wins) = 1 / (1 + exp(-α(r_A - r_B)))
```
where `α` controls how deterministic the predictions are.

### Cross-Validation Strategy
- **Rolling-origin**: Each split uses all data before test tournaments
- **No leakage**: Strict temporal ordering prevents future information
- **Tournament-level**: Hold out entire tournaments, not individual matches

### Optimization Approach
1. Define parameter search space
2. For each parameter combination:
   - Run cross-validation
   - Compute average loss
   - Track results
3. Return best parameters

## Best Practices

1. **Data Requirements**
   - Minimum ~20 tournaments for meaningful CV
   - Sufficient match history before each test tournament
   - Balanced representation of team/player skill levels

2. **Parameter Tuning**
   - Start with coarse grid, refine around best values
   - Use regularization to prevent overfitting
   - Consider computational cost vs accuracy gains

3. **Evaluation**
   - Always use held-out tournaments for final evaluation
   - Check diagnostic metrics beyond just loss
   - Verify predictions are well-calibrated

4. **Implementation Tips**
   - Cache computed ratings when possible
   - Parallelize CV splits if needed
   - Monitor convergence of iterative algorithms