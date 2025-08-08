# Cross-Validation Documentation

Cross-validation is critical for evaluating tournament ranking systems. This module provides specialized implementations that respect the temporal nature of tournament data.

## Overview

The cross-validation module offers two main approaches:

1. **Simple Cross-Validation** (`cross_validate_simple`) - Fast, streamlined evaluation
2. **Advanced Cross-Validation** (`cross_validate_ratings`) - Comprehensive evaluation with multiple metrics

## Simple Cross-Validation

### Purpose

Simple CV is designed for:
- Quick iteration during development
- Large-scale parameter sweeps
- Situations where speed is prioritized

### Key Features

- **Fast alpha optimization** using sampling (10-100x faster)
- **Minimal dependencies** for maximum performance
- **Clear, simple API** with sensible defaults

### Usage

```python
from rankings.evaluation.cross_validation import cross_validate_simple

results = cross_validate_simple(
    matches_df=matches,
    players_df=players,
    teams_df=teams,  # Optional
    ranking_entity="player",  # or "team"
    prediction_entity="team",  # or "player"
    agg_func="mean",  # How to aggregate player ratings to teams
    n_splits=5,
    fit_alpha=True,
    alpha_sample_size=10000,  # Matches to sample for alpha fitting
    engine_class=RatingEngine,
    engine_params={"decay_half_life_days": 30},
    verbose=True
)
```

### Output Structure

```python
{
    "avg_loss": 0.6234,      # Average log loss across folds
    "std_loss": 0.0145,      # Standard deviation of fold losses
    "fold_losses": [...],    # Individual fold losses
    "alpha": 1.0234,         # Optimal alpha parameter
    "concordance": 0.743,    # Average concordance
    "fold_details": [...]    # Detailed results per fold
}
```

## Advanced Cross-Validation

### Purpose

Advanced CV provides:
- Comprehensive metric evaluation
- Detailed diagnostic information
- Tournament-level analysis
- Multiple loss computation strategies

### Key Features

- **Multiple metrics** beyond simple loss
- **Tournament-aware evaluation** 
- **Flexible loss computation** (optimized, sampled, or full)
- **Diagnostic outputs** for debugging

### Usage

```python
from rankings.evaluation.cross_validation import cross_validate_ratings

results = cross_validate_ratings(
    matches_df=matches,
    players_df=players,
    ranking_entity="player",
    prediction_entity="team",
    n_splits=5,
    fit_alpha=True,
    alpha_method="optimized",  # or "sampled", "full"
    compute_extras=True,       # Include additional metrics
    min_test_tournaments=1,
    verbose=True
)
```

### Extended Metrics

When `compute_extras=True`, additional metrics are computed:

```python
{
    # Basic metrics
    "avg_loss": 0.6234,
    "std_loss": 0.0145,
    
    # Extended metrics
    "concordance": 0.743,        # Ranking correlation
    "skill_score": 0.234,        # Improvement over baseline
    "upset_oe": 1.023,          # Upset over/under expectation
    "alpha_std": 0.045,         # Alpha parameter stability
    "accuracy_70": 0.715,       # Accuracy at 70% confidence threshold
    "placement_spearman": 0.823, # Tournament placement correlation
    
    # Detailed results
    "fold_details": [...],
    "tournament_metrics": {...}
}
```

## Splitting Strategies

### Time-Based Splits

Both CV approaches use temporal splitting:

```python
from rankings.evaluation.cross_validation import create_time_based_folds

folds = create_time_based_folds(
    matches_df=matches,
    n_splits=5,
    min_test_tournaments=1,      # Minimum tournaments in test set
    min_tournaments_before=10    # Minimum tournaments before first test
)

for train_df, test_df, test_tournament_ids in folds:
    # Train and evaluate on each fold
    pass
```

### Split Visualization

```python
from rankings.evaluation.cross_validation import visualize_splits

# Visualize how data is split
fig = visualize_splits(
    matches_df=matches,
    n_splits=5
)
fig.show()
```

## Alpha Parameter Optimization

The alpha parameter scales rating differences to probabilities. Three optimization methods are available:

### 1. Optimized (Fastest)

Uses closed-form solution when possible:

```python
results = cross_validate_simple(
    matches_df=matches,
    fit_alpha=True,
    alpha_method="optimized"  # Default in simple CV
)
```

### 2. Sampled (Balanced)

Samples matches for faster optimization:

```python
results = cross_validate_simple(
    matches_df=matches,
    fit_alpha=True,
    alpha_sample_size=10000  # Number of matches to sample
)
```

### 3. Full (Most Accurate)

Uses all matches (slower but most precise):

```python
results = cross_validate_ratings(
    matches_df=matches,
    fit_alpha=True,
    alpha_method="full"
)
```

## Performance Considerations

### Memory Usage

- Simple CV: ~2-3x size of input data
- Advanced CV: ~3-5x size of input data
- Use sampling for datasets >1M matches

### Speed Optimization

1. **Use Simple CV** for initial exploration
2. **Enable sampling** for alpha optimization
3. **Reduce splits** if computation is slow
4. **Parallelize** multiple CV runs

### Example: Large Dataset

```python
# For large datasets (>1M matches)
results = cross_validate_simple(
    matches_df=large_matches,
    n_splits=3,  # Fewer splits
    alpha_sample_size=50000,  # Larger sample for stability
    verbose=True
)
```

## Advanced Usage

### Custom Evaluation Functions

```python
def custom_evaluate(engine, test_matches, test_ratings):
    """Custom evaluation logic."""
    predictions = engine.predict(test_matches, test_ratings)
    # Custom metric computation
    return custom_metric

# Use with advanced CV
results = cross_validate_ratings(
    matches_df=matches,
    custom_eval_func=custom_evaluate
)
```

### Tournament-Level Analysis

```python
# Get tournament-level metrics
results = cross_validate_ratings(
    matches_df=matches,
    compute_extras=True,
    return_tournament_metrics=True
)

# Analyze per-tournament performance
tournament_metrics = results["tournament_metrics"]
for tournament_id, metrics in tournament_metrics.items():
    print(f"Tournament {tournament_id}: Loss={metrics['loss']:.3f}")
```

### Debugging CV Issues

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Get detailed fold information
results = cross_validate_simple(
    matches_df=matches,
    verbose=True,
    return_diagnostics=True
)

# Inspect fold compositions
for i, fold_info in enumerate(results["fold_diagnostics"]):
    print(f"Fold {i}: Train={fold_info['n_train']}, Test={fold_info['n_test']}")
```

## Common Patterns

### 1. Parameter Sensitivity Analysis

```python
# Test sensitivity to decay parameter
decay_values = [7, 14, 30, 60, 90]
results = []

for decay in decay_values:
    cv_result = cross_validate_simple(
        matches_df=matches,
        engine_params={"decay_half_life_days": decay}
    )
    results.append((decay, cv_result["avg_loss"]))

# Plot sensitivity curve
plot_parameter_sensitivity(results)
```

### 2. Model Comparison

```python
# Compare multiple models
models = {
    "Bradley-Terry": {"engine_class": RatingEngine, "params": {}},
    "Elo": {"engine_class": EloEngine, "params": {"k": 32}},
    "Glicko": {"engine_class": GlickoEngine, "params": {"rd": 350}}
}

for name, config in models.items():
    result = cross_validate_simple(
        matches_df=matches,
        engine_class=config["engine_class"],
        engine_params=config["params"]
    )
    print(f"{name}: {result['avg_loss']:.4f}")
```

### 3. Temporal Stability

```python
# Analyze performance over time
n_splits = 10  # More splits for temporal analysis
results = cross_validate_ratings(
    matches_df=matches,
    n_splits=n_splits,
    compute_extras=True
)

# Plot loss over time
import matplotlib.pyplot as plt
plt.plot(results["fold_losses"])
plt.xlabel("Time Period")
plt.ylabel("Log Loss")
plt.title("Model Performance Over Time")
```

## Troubleshooting

### Issue: High Loss Values

**Symptoms**: Average loss >1.0 or increasing over folds

**Solutions**:
1. Check data quality (missing values, duplicates)
2. Verify rating engine convergence
3. Adjust regularization parameters
4. Ensure sufficient training data per fold

### Issue: Unstable Alpha Values

**Symptoms**: Alpha varies significantly between folds

**Solutions**:
1. Increase sample size for alpha optimization
2. Use full optimization method
3. Check for outliers in rating differences
4. Consider fixing alpha to reasonable value

### Issue: Memory Errors

**Symptoms**: Out of memory during CV

**Solutions**:
1. Use simple CV instead of advanced
2. Reduce number of splits
3. Enable sampling for large datasets
4. Process folds sequentially, not in parallel

## Best Practices

1. **Start Simple**: Use `cross_validate_simple` for initial exploration
2. **Validate Splits**: Always check split sizes and temporal ordering
3. **Multiple Metrics**: Don't rely solely on log loss
4. **Stability Checks**: Run CV multiple times with different random seeds
5. **Hold-Out Testing**: Keep final tournament for ultimate validation

## Next Steps

- [Loss Functions Guide](loss-functions.md)
- [Metrics Reference](metrics.md)
- [Optimization Tutorial](optimization.md)