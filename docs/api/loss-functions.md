# Loss Functions API Reference

Complete API documentation for loss computation and alpha optimization functions.

## Module: `src.rankings.evaluation.loss`

### Loss Computation Functions

#### `compute_weighted_log_loss`

Compute weighted binary cross-entropy loss.

```python
def compute_weighted_log_loss(
    y_true: Union[np.ndarray, pl.Series],
    y_pred: Union[np.ndarray, pl.Series],
    weights: Optional[Union[np.ndarray, pl.Series, str]] = None,
    weight_func: str = "confidence",
    eps: float = 1e-15
) -> float
```

**Parameters:**

- `y_true`: Binary outcomes (1 for win, 0 for loss)
- `y_pred`: Predicted probabilities [0, 1]
- `weights`: Sample weights, can be:
  - Array of weights
  - String specifying weight function
  - None for uniform weights
- `weight_func`: Weight function if weights is string:
  - `"uniform"`: Equal weights
  - `"confidence"`: |p - 0.5| * 2
  - `"confidence_squared"`: (|p - 0.5| * 2)²
  - `"entropy"`: -p*log(p) - (1-p)*log(1-p)
- `eps`: Small value to avoid log(0)

**Returns:**

Float: Weighted average log loss

**Example:**

```python
loss = compute_weighted_log_loss(
    y_true=[1, 0, 1, 1, 0],
    y_pred=[0.8, 0.2, 0.9, 0.6, 0.1],
    weight_func="confidence"
)
```

---

#### `compute_tournament_loss`

Compute loss for matches within a tournament.

```python
def compute_tournament_loss(
    matches_df: pl.DataFrame,
    winner_ratings: Dict[int, float],
    loser_ratings: Dict[int, float],
    alpha: float = 1.0,
    weight_func: str = "confidence",
    confidence_threshold: Optional[float] = None,
    return_details: bool = False
) -> Union[float, Dict[str, Any]]
```

**Parameters:**

- `matches_df`: Tournament matches DataFrame
- `winner_ratings`: Dictionary mapping winner IDs to ratings
- `loser_ratings`: Dictionary mapping loser IDs to ratings
- `alpha`: Bradley-Terry scaling parameter
- `weight_func`: Weight function for loss computation
- `confidence_threshold`: Only evaluate predictions above this confidence
- `return_details`: Return detailed results

**Returns:**

- Float: Tournament loss (if return_details=False)
- Dict: Detailed results including predictions (if return_details=True)

---

#### `compute_cross_tournament_loss`

Aggregate loss across multiple tournaments.

```python
def compute_cross_tournament_loss(
    matches_df: pl.DataFrame,
    ratings_dict: Dict[int, float],
    ranking_entity: str = "player",
    prediction_entity: str = "team",
    agg_func: str = "mean",
    alpha: float = 1.0,
    weight_func: str = "confidence",
    by_tournament: bool = False,
    min_matches_per_tournament: int = 10
) -> Dict[str, Any]
```

**Parameters:**

- `matches_df`: All matches DataFrame
- `ratings_dict`: Entity ratings dictionary
- `ranking_entity`: Entity type being ranked
- `prediction_entity`: Entity type for predictions
- `agg_func`: Team rating aggregation method
- `alpha`: Bradley-Terry parameter
- `weight_func`: Weight function
- `by_tournament`: Return per-tournament breakdown
- `min_matches_per_tournament`: Minimum matches to include tournament

**Returns:**

Dictionary containing:
- `overall_loss`: Aggregate loss
- `n_tournaments`: Number of tournaments
- `n_matches`: Total matches evaluated
- `tournament_losses`: Per-tournament losses (if by_tournament=True)

---

### Alpha Optimization Functions

#### `fit_alpha_parameter`

Fit optimal alpha using grid search.

```python
def fit_alpha_parameter(
    matches_df: pl.DataFrame,
    ratings_dict: Dict[int, float],
    entity_type: str = "player",
    alpha_range: Tuple[float, float] = (0.5, 2.0),
    n_points: int = 100,
    weight_func: str = "confidence"
) -> float
```

**Parameters:**

- `matches_df`: Match data
- `ratings_dict`: Current ratings
- `entity_type`: "player" or "team"
- `alpha_range`: Range to search
- `n_points`: Number of points to evaluate
- `weight_func`: Loss weight function

**Returns:**

Float: Optimal alpha value

---

#### `fit_alpha_parameter_optimized`

Fast alpha fitting using Newton-Raphson method.

```python
def fit_alpha_parameter_optimized(
    matches_df: pl.DataFrame,
    ratings_dict: Dict[int, float],
    entity_type: str = "player",
    initial_alpha: float = 1.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    weight_func: str = "confidence"
) -> float
```

**Parameters:**

- `matches_df`: Match data
- `ratings_dict`: Current ratings
- `entity_type`: Entity type
- `initial_alpha`: Starting point for optimization
- `tolerance`: Convergence tolerance
- `max_iterations`: Maximum iterations
- `weight_func`: Weight function

**Returns:**

Float: Optimal alpha value

**Example:**

```python
alpha = fit_alpha_parameter_optimized(
    matches_df=test_matches,
    ratings_dict=current_ratings,
    initial_alpha=1.0
)
```

---

#### `fit_alpha_parameter_sampled`

Alpha fitting with match sampling for speed.

```python
def fit_alpha_parameter_sampled(
    matches_df: pl.DataFrame,
    ratings_dict: Dict[int, float],
    entity_type: str = "player",
    sample_size: int = 10000,
    n_iterations: int = 50,
    initial_alpha: float = 1.0,
    learning_rate: float = 0.1,
    seed: int = 42
) -> float
```

**Parameters:**

- `matches_df`: Match data
- `ratings_dict`: Current ratings
- `entity_type`: Entity type
- `sample_size`: Number of matches to sample
- `n_iterations`: Optimization iterations
- `initial_alpha`: Starting alpha
- `learning_rate`: Step size for optimization
- `seed`: Random seed for sampling

**Returns:**

Float: Optimal alpha value

---

### Filtering Functions

#### `filter_matches_by_ranked_threshold`

Filter matches based on player ranking status.

```python
def filter_matches_by_ranked_threshold(
    matches_df: pl.DataFrame,
    ratings_dict: Dict[int, float],
    ranking_entity: str = "player",
    ranked_threshold: float = 20.0,
    min_matches_ratio: float = 0.5,
    return_mask: bool = False
) -> Union[pl.DataFrame, np.ndarray]
```

**Parameters:**

- `matches_df`: Match data
- `ratings_dict`: Current ratings
- `ranking_entity`: Entity type
- `ranked_threshold`: Minimum rating to be "ranked"
- `min_matches_ratio`: Minimum ratio of ranked players
- `return_mask`: Return boolean mask instead of filtered DataFrame

**Returns:**

- Filtered DataFrame (if return_mask=False)
- Boolean mask array (if return_mask=True)

**Example:**

```python
# Keep only matches with mostly ranked players
ranked_matches = filter_matches_by_ranked_threshold(
    matches_df=all_matches,
    ratings_dict=player_ratings,
    ranked_threshold=25.0,
    min_matches_ratio=0.75  # 75% of players must be ranked
)
```

---

#### `filter_matches_by_confidence`

Filter matches by prediction confidence.

```python
def filter_matches_by_confidence(
    matches_df: pl.DataFrame,
    predictions: Union[np.ndarray, pl.Series],
    confidence_threshold: float = 0.6,
    return_mask: bool = False
) -> Union[pl.DataFrame, np.ndarray]
```

**Parameters:**

- `matches_df`: Match data
- `predictions`: Predicted probabilities
- `confidence_threshold`: Minimum |p - 0.5| to include
- `return_mask`: Return boolean mask

**Returns:**

- Filtered DataFrame with predictions (if return_mask=False)
- Boolean mask array (if return_mask=True)

---

### Utility Functions

#### `bradley_terry_probability`

Compute Bradley-Terry win probability.

```python
def bradley_terry_probability(
    rating_a: float,
    rating_b: float,
    alpha: float = 1.0
) -> float
```

**Parameters:**

- `rating_a`: Rating of player/team A
- `rating_b`: Rating of player/team B
- `alpha`: Scaling parameter

**Returns:**

Float: Probability that A beats B

**Example:**

```python
prob = bradley_terry_probability(
    rating_a=1500,
    rating_b=1400,
    alpha=1.0
)
# prob ≈ 0.731
```

---

#### `compute_loss_gradient`

Compute gradient of log loss with respect to alpha.

```python
def compute_loss_gradient(
    rating_diffs: np.ndarray,
    outcomes: np.ndarray,
    alpha: float,
    weights: Optional[np.ndarray] = None
) -> float
```

**Parameters:**

- `rating_diffs`: Rating differences (winner - loser)
- `outcomes`: Binary outcomes
- `alpha`: Current alpha value
- `weights`: Sample weights

**Returns:**

Float: Gradient value

---

#### `compute_loss_hessian`

Compute second derivative for Newton's method.

```python
def compute_loss_hessian(
    rating_diffs: np.ndarray,
    alpha: float,
    weights: Optional[np.ndarray] = None
) -> float
```

**Parameters:**

- `rating_diffs`: Rating differences
- `alpha`: Current alpha value
- `weights`: Sample weights

**Returns:**

Float: Hessian value

---

## Advanced Usage

### Custom Weight Functions

```python
def custom_weight_function(predictions, matches_df):
    """Custom weight based on match importance."""
    base_confidence = np.abs(predictions - 0.5) * 2
    
    # Weight by tournament stage
    stage_weights = {
        "pools": 1.0,
        "bracket": 2.0,
        "finals": 3.0
    }
    stage_multiplier = matches_df["stage"].map(stage_weights)
    
    return base_confidence * stage_multiplier

# Use custom weights
loss = compute_weighted_log_loss(
    y_true=outcomes,
    y_pred=predictions,
    weights=custom_weight_function(predictions, matches_df)
)
```

### Batch Loss Computation

```python
def batch_compute_losses(matches_df, ratings_dict, batch_size=10000):
    """Compute loss in batches for memory efficiency."""
    
    total_loss = 0
    total_weight = 0
    
    for batch_start in range(0, len(matches_df), batch_size):
        batch = matches_df[batch_start:batch_start + batch_size]
        
        # Get predictions for batch
        winner_ratings = [ratings_dict.get(id, 0) for id in batch["winner_id"]]
        loser_ratings = [ratings_dict.get(id, 0) for id in batch["loser_id"]]
        
        predictions = [
            bradley_terry_probability(w, l) 
            for w, l in zip(winner_ratings, loser_ratings)
        ]
        
        # Compute weighted loss
        weights = np.abs(np.array(predictions) - 0.5) * 2
        batch_loss = compute_weighted_log_loss(
            y_true=np.ones(len(batch)),
            y_pred=predictions,
            weights=weights
        )
        
        batch_weight = np.sum(weights)
        total_loss += batch_loss * batch_weight
        total_weight += batch_weight
    
    return total_loss / total_weight if total_weight > 0 else 0
```

## Error Handling

```python
# Handle missing ratings gracefully
def safe_compute_loss(matches_df, ratings_dict):
    # Check for missing ratings
    missing_winners = matches_df.filter(
        ~pl.col("winner_id").is_in(ratings_dict.keys())
    )
    missing_losers = matches_df.filter(
        ~pl.col("loser_id").is_in(ratings_dict.keys())
    )
    
    if len(missing_winners) > 0 or len(missing_losers) > 0:
        logger.warning(
            f"Missing ratings for {len(missing_winners)} winners "
            f"and {len(missing_losers)} losers"
        )
    
    # Filter to matches with complete ratings
    valid_matches = matches_df.filter(
        pl.col("winner_id").is_in(ratings_dict.keys()) &
        pl.col("loser_id").is_in(ratings_dict.keys())
    )
    
    return compute_tournament_loss(valid_matches, ratings_dict, ratings_dict)
```