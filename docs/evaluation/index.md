# Evaluation Module Documentation

The evaluation module is the core of the Sendouq Analysis system, providing comprehensive tools for assessing and optimizing tournament ranking algorithms.

## Overview

The evaluation module (`src.rankings.evaluation`) consists of four main components:

1. **Cross-Validation** - Temporal splitting and validation strategies
2. **Loss Functions** - Various loss metrics for model evaluation
3. **Extra Metrics** - Additional performance measures beyond standard loss
4. **Optimization** - Hyperparameter tuning for rating engines

## Module Structure

```
evaluation/
├── cross_validation/
│   ├── __init__.py              # Module exports
│   ├── cross_validation_simple.py    # Fast, simple CV implementation
│   ├── cross_validation_advanced.py  # Full-featured CV with multiple metrics
│   └── simple_splits.py         # Splitting utilities
├── loss.py                      # Loss functions and alpha optimization
├── metrics_extras.py            # Additional evaluation metrics
└── optimizer.py                 # Hyperparameter optimization
```

## Key Concepts

### 1. Temporal Cross-Validation

Tournament data has a natural temporal structure. Our cross-validation approaches respect this:

- **Forward Chaining**: Train on past data, test on future data
- **No Data Leakage**: Strict temporal boundaries prevent future information from influencing past predictions
- **Tournament Boundaries**: Splits align with tournament boundaries, not arbitrary time points

### 2. Confidence-Weighted Evaluation

Not all predictions are equal. The system weights evaluation metrics by:

- **Match Importance**: Tournament stage, stakes
- **Prediction Confidence**: Model certainty
- **Data Quality**: Number of prior observations

### 3. Multi-Objective Evaluation

Beyond simple accuracy, we evaluate:

- **Calibration**: Do 70% confidence predictions win 70% of the time?
- **Discrimination**: Can the model distinguish skill levels?
- **Stability**: How consistent are ratings over time?

## Quick Start

### Simple Cross-Validation

For quick evaluation with sensible defaults:

```python
from rankings.evaluation.cross_validation import cross_validate_simple

results = cross_validate_simple(
    matches_df=matches,
    players_df=players,
    n_splits=5,
    fit_alpha=True,  # Automatically optimize alpha parameter
    verbose=True
)

print(f"Average loss: {results['avg_loss']:.4f}")
print(f"Optimal alpha: {results['alpha']:.4f}")
```

### Advanced Cross-Validation

For comprehensive evaluation with multiple metrics:

```python
from rankings.evaluation.cross_validation import cross_validate_ratings

results = cross_validate_ratings(
    matches_df=matches,
    players_df=players,
    ranking_entity="player",
    prediction_entity="team",
    n_splits=5,
    fit_alpha=True,
    compute_extras=True  # Include all additional metrics
)

# Access detailed metrics
print(f"Concordance: {results['concordance']:.4f}")
print(f"Skill Score: {results['skill_score']:.4f}")
print(f"Upset O/E: {results['upset_oe']:.4f}")
```

### Hyperparameter Optimization

Find the best parameters for your rating engine:

```python
from rankings.evaluation.optimizer import optimize_rating_engine

best_params = optimize_rating_engine(
    matches_df=matches,
    players_df=players,
    method="grid",  # or "bayesian"
    param_space={
        "decay_half_life_days": [7, 14, 30, 60],
        "damping_factor": [0.8, 0.85, 0.9],
        "beta": [0.0, 0.5, 1.0]
    },
    n_splits=5
)

print(f"Best parameters: {best_params['best_params']}")
print(f"Best CV loss: {best_params['best_score']:.4f}")
```

## Design Philosophy

### 1. Flexibility with Sensible Defaults

- Simple API for common use cases
- Advanced options for power users
- Reasonable defaults based on empirical testing

### 2. Performance at Scale

- Efficient implementations for large datasets
- Sampling strategies for expensive computations
- Parallel processing where beneficial

### 3. Statistical Rigor

- Proper handling of temporal dependencies
- Multiple hypothesis correction where needed
- Confidence intervals and uncertainty quantification

## Common Use Cases

### 1. Comparing Rating Systems

```python
# Compare different rating engines
from rankings.analysis.engine import RatingEngine, EloEngine

for engine_class in [RatingEngine, EloEngine]:
    results = cross_validate_simple(
        matches_df=matches,
        engine_class=engine_class,
        n_splits=5
    )
    print(f"{engine_class.__name__}: {results['avg_loss']:.4f}")
```

### 2. Feature Importance Analysis

```python
# Test impact of different features
base_params = {"decay_half_life_days": 30}

for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
    params = {**base_params, "beta": beta}
    results = cross_validate_simple(
        matches_df=matches,
        engine_params=params
    )
    print(f"Beta={beta}: Loss={results['avg_loss']:.4f}")
```

### 3. Temporal Stability Analysis

```python
# Analyze performance over time
from rankings.evaluation.cross_validation import create_time_based_folds

folds = create_time_based_folds(matches, n_splits=10)
fold_losses = []

for train, test, _ in folds:
    # Evaluate on each fold
    loss = evaluate_fold(train, test)
    fold_losses.append(loss)

# Plot temporal trends
plot_temporal_performance(fold_losses)
```

## Best Practices

### 1. Data Preparation

- Ensure consistent player/team IDs across datasets
- Handle missing data appropriately
- Verify temporal ordering of matches

### 2. Validation Strategy

- Use at least 5 folds for stable estimates
- Consider computational cost vs. statistical precision
- Always fit alpha parameter for probabilistic predictions

### 3. Interpretation

- Look beyond average metrics to distributions
- Consider practical significance, not just statistical
- Validate findings on held-out tournament data

## Troubleshooting

### Common Issues

1. **High Loss Values**
   - Check data quality and consistency
   - Verify rating engine convergence
   - Consider adjusting regularization

2. **Slow Performance**
   - Use simple CV for initial exploration
   - Enable sampling for alpha optimization
   - Reduce parameter grid size

3. **Unstable Results**
   - Increase number of CV folds
   - Check for data leakage
   - Verify temporal ordering

## Next Steps

- [Cross-Validation Deep Dive](cross-validation.md)
- [Loss Functions Reference](loss-functions.md)
- [Metrics Guide](metrics.md)
- [Optimization Tutorial](optimization.md)