# Loss Functions Documentation

Loss functions are fundamental to evaluating ranking system performance. This module provides various loss metrics tailored for tournament prediction scenarios.

## Overview

The loss module (`src.rankings.evaluation.loss`) provides:

1. **Log Loss Computation** - Standard and weighted variants
2. **Tournament-Based Loss** - Contextual evaluation within tournaments
3. **Alpha Parameter Optimization** - Scaling rating differences to probabilities
4. **Filtering and Thresholding** - Focus evaluation on confident predictions

## Core Loss Functions

### 1. Weighted Log Loss

The primary loss function with confidence weighting:

```python
from src.rankings.evaluation.loss import compute_weighted_log_loss

loss = compute_weighted_log_loss(
    y_true=actuals,           # Binary outcomes (1 for win, 0 for loss)
    y_pred=predictions,       # Predicted probabilities
    weights=confidence_weights # Optional confidence weights
)
```

#### Weight Calculation

Weights can incorporate multiple factors:

```python
# Simple confidence weighting
weights = np.abs(predictions - 0.5) * 2  # 0 at 50%, 1 at 0%/100%

# Custom weighting function
def custom_weights(predictions, rating_diffs):
    confidence = np.abs(predictions - 0.5) * 2
    importance = np.log1p(np.abs(rating_diffs))
    return confidence * importance

weights = custom_weights(predictions, rating_differences)
```

### 2. Tournament Loss

Evaluate predictions within tournament contexts:

```python
from src.rankings.evaluation.loss import compute_tournament_loss

tournament_loss = compute_tournament_loss(
    matches_df=tournament_matches,
    winner_ratings=winner_rating_dict,
    loser_ratings=loser_rating_dict,
    alpha=1.0,
    weight_func="confidence",  # or "uniform", "entropy"
    confidence_threshold=0.6   # Only evaluate confident predictions
)
```

### 3. Cross-Tournament Loss

Aggregate loss across multiple tournaments:

```python
from src.rankings.evaluation.loss import compute_cross_tournament_loss

results = compute_cross_tournament_loss(
    matches_df=all_matches,
    ratings_dict=player_ratings,
    ranking_entity="player",
    prediction_entity="team",
    agg_func="mean",
    alpha=1.0,
    by_tournament=True  # Get per-tournament breakdown
)

print(f"Overall loss: {results['overall_loss']:.4f}")
print(f"Per-tournament: {results['tournament_losses']}")
```

## Alpha Parameter Optimization

The alpha parameter scales rating differences to probabilities using the Bradley-Terry model:

```
P(A beats B) = 1 / (1 + exp(-alpha * (rating_A - rating_B)))
```

### Optimization Methods

#### 1. Optimized (Closed-Form)

Fastest method using Newton-Raphson:

```python
from src.rankings.evaluation.loss import fit_alpha_parameter_optimized

optimal_alpha = fit_alpha_parameter_optimized(
    matches_df=matches,
    ratings_dict=ratings,
    initial_alpha=1.0,
    tolerance=1e-6,
    max_iterations=100
)
```

#### 2. Sampled

Balance between speed and accuracy:

```python
from src.rankings.evaluation.loss import fit_alpha_parameter_sampled

optimal_alpha = fit_alpha_parameter_sampled(
    matches_df=matches,
    ratings_dict=ratings,
    sample_size=10000,  # Number of matches to sample
    n_iterations=50
)
```

#### 3. Grid Search

Most thorough but slowest:

```python
from src.rankings.evaluation.loss import fit_alpha_parameter

optimal_alpha = fit_alpha_parameter(
    matches_df=matches,
    ratings_dict=ratings,
    alpha_range=(0.5, 2.0),
    n_points=100
)
```

## Filtering and Thresholding

### Confidence-Based Filtering

Focus evaluation on predictions where the model is confident:

```python
from src.rankings.evaluation.loss import filter_matches_by_confidence

filtered_matches = filter_matches_by_confidence(
    matches_df=matches,
    predictions=predicted_probabilities,
    confidence_threshold=0.6  # Only keep |p - 0.5| > 0.1
)

# Evaluate on confident predictions only
loss = compute_weighted_log_loss(
    y_true=filtered_matches["actual"],
    y_pred=filtered_matches["predicted"]
)
```

### Ranked Player Filtering

Evaluate only matches involving ranked players:

```python
from src.rankings.evaluation.loss import filter_matches_by_ranked_threshold

filtered_matches = filter_matches_by_ranked_threshold(
    matches_df=matches,
    ratings_dict=ratings,
    ranking_entity="player",
    ranked_threshold=20.0,  # Minimum rating to be considered "ranked"
    min_matches_ratio=0.5   # At least 50% of players must be ranked
)
```

## Weight Functions

### Built-in Weight Functions

```python
# Uniform weights (no weighting)
loss = compute_weighted_log_loss(y_true, y_pred, weight_func="uniform")

# Confidence weighting (default)
loss = compute_weighted_log_loss(y_true, y_pred, weight_func="confidence")

# Entropy-based weighting
loss = compute_weighted_log_loss(y_true, y_pred, weight_func="entropy")

# Squared confidence weighting
loss = compute_weighted_log_loss(y_true, y_pred, weight_func="confidence_squared")
```

### Custom Weight Functions

```python
def tournament_stage_weights(matches_df):
    """Weight by tournament stage importance."""
    stage_weights = {
        "pools": 1.0,
        "bracket": 2.0,
        "finals": 3.0
    }
    return matches_df["stage"].map(stage_weights)

# Use custom weights
weights = tournament_stage_weights(matches)
loss = compute_weighted_log_loss(y_true, y_pred, weights=weights)
```

## Advanced Usage

### 1. Multi-Objective Loss

Combine multiple loss components:

```python
def multi_objective_loss(matches_df, predictions, ratings):
    # Standard log loss
    log_loss = compute_weighted_log_loss(
        matches_df["actual"], 
        predictions
    )
    
    # Calibration loss
    calibration_loss = compute_calibration_loss(
        matches_df["actual"],
        predictions
    )
    
    # Ranking stability loss
    stability_loss = compute_stability_loss(ratings)
    
    # Combined loss
    return (
        0.7 * log_loss + 
        0.2 * calibration_loss + 
        0.1 * stability_loss
    )
```

### 2. Time-Weighted Loss

Give more weight to recent matches:

```python
def time_weighted_loss(matches_df, predictions):
    # Calculate days since match
    days_ago = (pd.Timestamp.now() - matches_df["date"]).dt.days
    
    # Exponential decay weights
    time_weights = np.exp(-days_ago / 30.0)  # 30-day half-life
    
    return compute_weighted_log_loss(
        matches_df["actual"],
        predictions,
        weights=time_weights
    )
```

### 3. Stratified Loss Analysis

Analyze loss across different segments:

```python
def stratified_loss_analysis(matches_df, predictions):
    results = {}
    
    # By ranking difference
    for min_diff, max_diff in [(0, 50), (50, 100), (100, 200), (200, float('inf'))]:
        mask = (matches_df["rating_diff"] >= min_diff) & (matches_df["rating_diff"] < max_diff)
        results[f"diff_{min_diff}_{max_diff}"] = compute_weighted_log_loss(
            matches_df[mask]["actual"],
            predictions[mask]
        )
    
    # By tournament stage
    for stage in ["pools", "bracket", "finals"]:
        mask = matches_df["stage"] == stage
        results[f"stage_{stage}"] = compute_weighted_log_loss(
            matches_df[mask]["actual"],
            predictions[mask]
        )
    
    return results
```

## Performance Optimization

### 1. Vectorized Operations

```python
# Efficient loss computation for large datasets
def vectorized_log_loss(y_true, y_pred, weights=None):
    # Clip predictions to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Vectorized log loss
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    if weights is not None:
        return np.average(loss, weights=weights)
    return np.mean(loss)
```

### 2. Batch Processing

```python
def batch_compute_loss(matches_df, ratings_dict, batch_size=10000):
    """Compute loss in batches for memory efficiency."""
    total_loss = 0
    total_weight = 0
    
    for i in range(0, len(matches_df), batch_size):
        batch = matches_df[i:i + batch_size]
        
        # Compute predictions for batch
        predictions = compute_predictions_batch(batch, ratings_dict)
        
        # Compute weighted loss
        weights = compute_confidence_weights(predictions)
        batch_loss = compute_weighted_log_loss(
            batch["actual"], 
            predictions, 
            weights
        )
        
        # Accumulate
        batch_weight = np.sum(weights)
        total_loss += batch_loss * batch_weight
        total_weight += batch_weight
    
    return total_loss / total_weight
```

## Common Patterns

### 1. Loss Decomposition

Understanding where loss comes from:

```python
def decompose_loss(matches_df, predictions):
    # Overall loss
    total_loss = compute_weighted_log_loss(matches_df["actual"], predictions)
    
    # Decompose by factors
    decomposition = {
        "total": total_loss,
        "favorites": compute_loss_on_favorites(matches_df, predictions),
        "upsets": compute_loss_on_upsets(matches_df, predictions),
        "close_matches": compute_loss_on_close_matches(matches_df, predictions),
        "blowouts": compute_loss_on_blowouts(matches_df, predictions)
    }
    
    return decomposition
```

### 2. Loss Trajectory Analysis

Track how loss changes over time:

```python
def analyze_loss_trajectory(matches_df, engine):
    # Sort by date
    matches_df = matches_df.sort("date")
    
    # Compute rolling loss
    window_size = 1000
    losses = []
    
    for i in range(window_size, len(matches_df)):
        window = matches_df[i-window_size:i]
        
        # Get current ratings
        ratings = engine.get_ratings_at_date(window["date"].max())
        
        # Compute predictions
        predictions = compute_predictions(window, ratings)
        
        # Compute loss
        loss = compute_weighted_log_loss(window["actual"], predictions)
        losses.append(loss)
    
    return losses
```

## Best Practices

### 1. Always Clip Predictions

```python
# Prevent numerical instability
predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
```

### 2. Handle Missing Ratings

```python
def safe_compute_loss(matches_df, ratings_dict):
    # Filter matches with ratings for both players
    has_ratings = matches_df.filter(
        pl.col("winner_id").is_in(ratings_dict.keys()) &
        pl.col("loser_id").is_in(ratings_dict.keys())
    )
    
    if len(has_ratings) < len(matches_df):
        print(f"Warning: {len(matches_df) - len(has_ratings)} matches skipped due to missing ratings")
    
    return compute_weighted_log_loss(has_ratings["actual"], predictions)
```

### 3. Validate Weight Distributions

```python
def validate_weights(weights):
    """Ensure weights are properly normalized and distributed."""
    assert np.all(weights >= 0), "Weights must be non-negative"
    assert np.sum(weights) > 0, "At least some weights must be positive"
    
    # Check distribution
    if np.std(weights) / np.mean(weights) > 2:
        print("Warning: High weight variance may lead to unstable results")
    
    return weights / np.sum(weights)
```

## Next Steps

- [Metrics Guide](metrics.md) - Additional evaluation metrics
- [Optimization Tutorial](optimization.md) - Using loss for hyperparameter tuning
- [API Reference](../api/loss-functions.md) - Detailed function documentation