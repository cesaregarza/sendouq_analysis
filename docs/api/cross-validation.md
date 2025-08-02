# Cross-Validation API Reference

Complete API documentation for the cross-validation module.

## Module: `src.rankings.evaluation.cross_validation`

### Functions

#### `cross_validate_simple`

Fast cross-validation with optimized alpha fitting.

```python
def cross_validate_simple(
    matches_df: pl.DataFrame,
    players_df: Optional[pl.DataFrame] = None,
    teams_df: Optional[pl.DataFrame] = None,
    ranking_entity: str = "player",
    prediction_entity: str = "team",
    agg_func: str = "mean",
    n_splits: int = 5,
    fit_alpha: bool = True,
    alpha_sample_size: int = 10000,
    engine_class: Type[RatingEngine] = RatingEngine,
    engine_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Dict[str, Any]
```

**Parameters:**

- `matches_df` (pl.DataFrame): DataFrame containing match results with columns:
  - `tournament_id`: Tournament identifier
  - `last_game_finished_at`: Match timestamp
  - `winner_[entity]_id`: Winner identifier
  - `loser_[entity]_id`: Loser identifier
  
- `players_df` (Optional[pl.DataFrame]): Player metadata with columns:
  - `player_id`: Player identifier
  - Additional metadata columns

- `teams_df` (Optional[pl.DataFrame]): Team metadata with columns:
  - `team_id`: Team identifier
  - Additional metadata columns

- `ranking_entity` (str): Entity type to rank ("player" or "team")

- `prediction_entity` (str): Entity type for predictions ("player" or "team")

- `agg_func` (str): Aggregation function for team ratings:
  - `"mean"`: Average of player ratings
  - `"max"`: Maximum player rating
  - `"weighted_mean"`: Weighted by player importance

- `n_splits` (int): Number of cross-validation folds

- `fit_alpha` (bool): Whether to optimize alpha parameter

- `alpha_sample_size` (int): Number of matches to sample for alpha optimization

- `engine_class` (Type[RatingEngine]): Rating engine class to use

- `engine_params` (Optional[Dict[str, Any]]): Parameters for rating engine

- `verbose` (bool): Print progress information

**Returns:**

Dictionary containing:
- `avg_loss` (float): Average log loss across folds
- `std_loss` (float): Standard deviation of fold losses
- `fold_losses` (List[float]): Individual fold losses
- `alpha` (float): Optimal alpha parameter (if fit_alpha=True)
- `concordance` (float): Average concordance across folds
- `fold_details` (List[Dict]): Detailed results per fold

**Example:**

```python
results = cross_validate_simple(
    matches_df=matches,
    players_df=players,
    n_splits=5,
    fit_alpha=True,
    verbose=True
)
```

---

#### `cross_validate_ratings`

Comprehensive cross-validation with multiple metrics.

```python
def cross_validate_ratings(
    matches_df: pl.DataFrame,
    players_df: Optional[pl.DataFrame] = None,
    teams_df: Optional[pl.DataFrame] = None,
    ranking_entity: str = "player",
    prediction_entity: str = "team",
    agg_func: str = "mean",
    n_splits: int = 5,
    fit_alpha: bool = True,
    alpha_method: str = "optimized",
    compute_extras: bool = False,
    engine_class: Type[RatingEngine] = RatingEngine,
    engine_params: Optional[Dict[str, Any]] = None,
    min_test_tournaments: int = 1,
    verbose: bool = False
) -> Dict[str, Any]
```

**Parameters:**

All parameters from `cross_validate_simple` plus:

- `alpha_method` (str): Method for alpha optimization:
  - `"optimized"`: Fast closed-form solution
  - `"sampled"`: Sample-based optimization
  - `"full"`: Use all matches (slowest)

- `compute_extras` (bool): Compute additional metrics

- `min_test_tournaments` (int): Minimum tournaments in test set

**Returns:**

Dictionary containing all returns from `cross_validate_simple` plus:
- `skill_score` (float): Improvement over baseline
- `upset_oe` (float): Upset over/expected ratio
- `alpha_std` (float): Alpha parameter stability
- `accuracy_70` (float): Accuracy at 70% confidence
- `placement_spearman` (float): Tournament placement correlation
- `tournament_metrics` (Dict): Per-tournament metrics (if requested)

---

#### `create_time_based_folds`

Create temporal cross-validation splits.

```python
def create_time_based_folds(
    matches_df: pl.DataFrame,
    n_splits: int = 5,
    min_test_tournaments: int = 1,
    min_tournaments_before: int = 10,
    tournament_id_col: str = "tournament_id",
    timestamp_col: str = "last_game_finished_at"
) -> List[Tuple[pl.DataFrame, pl.DataFrame, List[int]]]
```

**Parameters:**

- `matches_df` (pl.DataFrame): Match data
- `n_splits` (int): Number of splits
- `min_test_tournaments` (int): Minimum tournaments per test fold
- `min_tournaments_before` (int): Minimum tournaments before first test
- `tournament_id_col` (str): Column name for tournament ID
- `timestamp_col` (str): Column name for timestamps

**Returns:**

List of tuples containing:
- Train DataFrame
- Test DataFrame
- List of test tournament IDs

---

#### `evaluate_on_split`

Evaluate a single train/test split.

```python
def evaluate_on_split(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    test_tournament_ids: List[int],
    engine_class: Type[RatingEngine],
    engine_params: Dict[str, Any],
    players_df: Optional[pl.DataFrame] = None,
    teams_df: Optional[pl.DataFrame] = None,
    ranking_entity: str = "player",
    prediction_entity: str = "team",
    agg_func: str = "mean",
    fit_alpha: bool = True,
    compute_extras: bool = False
) -> Dict[str, Any]
```

**Parameters:**

- `train_df` (pl.DataFrame): Training matches
- `test_df` (pl.DataFrame): Test matches
- `test_tournament_ids` (List[int]): Test tournament IDs
- Other parameters as in `cross_validate_ratings`

**Returns:**

Dictionary with evaluation metrics for the split

---

### Simple Splits Module

#### `create_simple_time_splits`

Create simple temporal splits optimized for speed.

```python
def create_simple_time_splits(
    matches_df: pl.DataFrame,
    n_splits: int = 5,
    min_tournaments_before_cv: int = 10
) -> List[Tuple[pl.DataFrame, pl.DataFrame]]
```

**Parameters:**

- `matches_df` (pl.DataFrame): Match data
- `n_splits` (int): Number of splits
- `min_tournaments_before_cv` (int): Minimum tournaments before CV

**Returns:**

List of (train, test) DataFrame tuples

---

#### `visualize_splits`

Visualize cross-validation splits.

```python
def visualize_splits(
    matches_df: pl.DataFrame,
    n_splits: int = 5,
    min_tournaments_before_cv: int = 10,
    return_fig: bool = True
) -> Optional[plt.Figure]
```

**Parameters:**

- `matches_df` (pl.DataFrame): Match data
- `n_splits` (int): Number of splits
- `min_tournaments_before_cv` (int): Minimum tournaments before CV
- `return_fig` (bool): Return figure object

**Returns:**

Matplotlib figure object (if return_fig=True)

---

#### `get_split_info`

Get information about cross-validation splits.

```python
def get_split_info(
    matches_df: pl.DataFrame,
    n_splits: int = 5,
    min_tournaments_before_cv: int = 10
) -> pl.DataFrame
```

**Parameters:**

- `matches_df` (pl.DataFrame): Match data
- `n_splits` (int): Number of splits
- `min_tournaments_before_cv` (int): Minimum tournaments before CV

**Returns:**

DataFrame with split information:
- `split`: Split index
- `train_matches`: Number of training matches
- `test_matches`: Number of test matches
- `train_tournaments`: Number of training tournaments
- `test_tournaments`: Number of test tournaments
- `train_start_date`: Training start date
- `train_end_date`: Training end date
- `test_start_date`: Test start date
- `test_end_date`: Test end date

## Error Handling

All functions handle common errors gracefully:

```python
try:
    results = cross_validate_simple(matches_df, n_splits=10)
except ValueError as e:
    if "Insufficient tournaments" in str(e):
        # Not enough data for requested splits
        results = cross_validate_simple(matches_df, n_splits=3)
    else:
        raise
```

## Performance Tips

### 1. Use Simple CV for Large Datasets

```python
# For datasets > 100k matches
results = cross_validate_simple(
    large_matches_df,
    alpha_sample_size=50000,  # Larger sample for stability
    verbose=True
)
```

### 2. Reduce Splits for Quick Iteration

```python
# During development
dev_results = cross_validate_simple(matches_df, n_splits=3)

# For final evaluation
final_results = cross_validate_ratings(
    matches_df, 
    n_splits=10,
    compute_extras=True
)
```

### 3. Cache Rating Computations

```python
# Reuse engine between CV runs
engine = RatingEngine(**params)
engine.fit(train_matches)

# Use pre-computed ratings
results = evaluate_on_split(
    train_df, test_df, test_tournaments,
    engine_class=RatingEngine,
    engine_params=params,
    pre_computed_engine=engine  # Avoid recomputation
)
```