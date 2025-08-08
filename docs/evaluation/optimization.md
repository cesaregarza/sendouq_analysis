# Optimization Module Documentation

The optimization module provides automated hyperparameter tuning for tournament rating engines, supporting both exhaustive grid search and intelligent Bayesian optimization approaches.

## Overview

The optimization module (`src.rankings.evaluation.optimizer`) includes:

1. **Grid Search** - Exhaustive search over parameter combinations
2. **Bayesian Optimization** - Intelligent search using Gaussian Processes
3. **Convenience Functions** - High-level API for common optimization tasks

## Grid Search Optimization

### Basic Usage

```python
from rankings.evaluation.optimizer import GridSearchOptimizer

# Define parameter grid
param_grid = {
    "decay_half_life_days": [7, 14, 30, 60],
    "damping_factor": [0.8, 0.85, 0.9],
    "beta": [0.0, 0.5, 1.0],
    "influence_agg_method": ["top_10_sum", "top_20_sum", "max"]
}

# Create optimizer
optimizer = GridSearchOptimizer(
    param_grid=param_grid,
    n_splits=5,
    fit_alpha=True,
    verbose=True
)

# Run optimization
optimizer.fit(
    matches_df=matches,
    players_df=players,
    ranking_entity="player",
    prediction_entity="team"
)

# Get results
print(f"Best parameters: {optimizer.best_params_}")
print(f"Best score: {optimizer.best_score_:.4f}")
```

### Analyzing Results

```python
# Get summary DataFrame
results_df = optimizer.get_results_summary()
print(results_df.head(10))  # Top 10 parameter combinations

# Visualize parameter importance
import matplotlib.pyplot as plt

for param in param_grid.keys():
    plt.figure(figsize=(10, 6))
    
    # Group by parameter value
    grouped = results_df.group_by(param).agg([
        pl.col("mean_loss").mean().alias("avg_loss"),
        pl.col("mean_loss").std().alias("std_loss")
    ])
    
    plt.errorbar(
        grouped[param], 
        grouped["avg_loss"],
        yerr=grouped["std_loss"],
        marker='o'
    )
    plt.xlabel(param)
    plt.ylabel("CV Loss")
    plt.title(f"Impact of {param} on Performance")
    plt.show()
```

### Custom Engine Parameters

```python
# Optimize custom rating engine
from my_module import CustomRatingEngine

optimizer = GridSearchOptimizer(
    param_grid={
        "custom_param1": [1, 2, 3],
        "custom_param2": ["a", "b", "c"],
        "learning_rate": [0.01, 0.1, 1.0]
    },
    engine_class=CustomRatingEngine,
    n_splits=5
)

optimizer.fit(matches_df=matches)
```

## Bayesian Optimization

### Basic Usage

```python
from rankings.evaluation.optimizer import BayesianOptimizer

# Define parameter bounds
param_bounds = {
    "decay_half_life_days": (7.0, 90.0),      # Continuous
    "damping_factor": (0.7, 0.95),            # Continuous
    "beta": (0.0, 1.0),                       # Continuous
    "n_iterations": (100, 1000)               # Integer
}

# Specify parameter types
param_types = {
    "n_iterations": "int"  # Others default to float
}

# Create optimizer
bayesian_opt = BayesianOptimizer(
    param_bounds=param_bounds,
    param_types=param_types,
    n_initial=10,      # Random exploration points
    n_iterations=50,   # Total evaluations
    verbose=True
)

# Run optimization
bayesian_opt.fit(
    matches_df=matches,
    players_df=players
)
```

### Advanced Bayesian Configuration

```python
# Use different acquisition functions
from skopt import gp_minimize

# Custom Bayesian optimization with more control
def custom_bayesian_optimize(matches_df, param_space):
    
    def objective(params):
        # Unpack parameters
        decay, damping, beta = params
        
        # Run cross-validation
        cv_results = cross_validate_simple(
            matches_df=matches_df,
            engine_params={
                "decay_half_life_days": decay,
                "damping_factor": damping,
                "beta": beta
            }
        )
        
        return cv_results["avg_loss"]
    
    # Run optimization with custom settings
    result = gp_minimize(
        func=objective,
        dimensions=[
            Real(7.0, 90.0, name="decay"),
            Real(0.7, 0.95, name="damping"),
            Real(0.0, 1.0, name="beta")
        ],
        n_calls=100,
        n_initial_points=20,
        acq_func="EI",  # Expected Improvement
        acq_optimizer="lbfgs",
        random_state=42
    )
    
    return result
```

## Convenience Function

### Quick Optimization

```python
from rankings.evaluation.optimizer import optimize_rating_engine

# Grid search with defaults
results = optimize_rating_engine(
    matches_df=matches,
    players_df=players,
    method="grid",
    n_splits=5
)

# Bayesian optimization with custom space
results = optimize_rating_engine(
    matches_df=matches,
    players_df=players,
    method="bayesian",
    param_space={
        "decay_half_life_days": (7.0, 90.0),
        "damping_factor": (0.7, 0.95)
    },
    n_splits=3  # Fewer splits for faster optimization
)
```

## Optimization Strategies

### 1. Coarse-to-Fine Search

```python
# Stage 1: Coarse grid
coarse_results = optimize_rating_engine(
    matches_df=matches,
    method="grid",
    param_space={
        "decay_half_life_days": [7, 30, 90],
        "damping_factor": [0.7, 0.85, 0.95]
    },
    n_splits=3  # Quick evaluation
)

# Stage 2: Fine grid around best parameters
best_decay = coarse_results["best_params"]["decay_half_life_days"]
best_damping = coarse_results["best_params"]["damping_factor"]

fine_results = optimize_rating_engine(
    matches_df=matches,
    method="grid",
    param_space={
        "decay_half_life_days": [
            best_decay - 10, best_decay, best_decay + 10
        ],
        "damping_factor": [
            best_damping - 0.05, best_damping, best_damping + 0.05
        ]
    },
    n_splits=5  # More thorough evaluation
)
```

### 2. Multi-Objective Optimization

```python
def multi_objective_optimize(matches_df, param_space):
    """Optimize for multiple objectives simultaneously."""
    
    results = []
    
    for params in generate_param_combinations(param_space):
        cv_results = cross_validate_ratings(
            matches_df=matches_df,
            engine_params=params,
            compute_extras=True
        )
        
        # Multiple objectives
        objectives = {
            "loss": cv_results["avg_loss"],
            "concordance": -cv_results["concordance"],  # Negative because we minimize
            "stability": cv_results["std_loss"],
            "computation_time": cv_results["avg_fit_time"]
        }
        
        results.append({
            "params": params,
            **objectives
        })
    
    # Find Pareto-optimal solutions
    pareto_front = find_pareto_frontier(results)
    
    return pareto_front
```

### 3. Constraint-Based Optimization

```python
# Optimize with constraints
def constrained_optimize(matches_df, constraints):
    """Optimize with practical constraints."""
    
    param_grid = {
        "decay_half_life_days": [7, 14, 30, 60, 90],
        "damping_factor": [0.7, 0.8, 0.85, 0.9, 0.95],
        "beta": [0.0, 0.25, 0.5, 0.75, 1.0]
    }
    
    # Filter parameter combinations based on constraints
    valid_combinations = []
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        
        # Check constraints
        if constraints.get("max_computation_time"):
            # Estimate computation time
            est_time = estimate_computation_time(param_dict)
            if est_time > constraints["max_computation_time"]:
                continue
        
        if constraints.get("min_stability"):
            # Quick stability check
            if param_dict["decay_half_life_days"] < 14:
                continue  # Too unstable
        
        valid_combinations.append(param_dict)
    
    # Optimize over valid combinations only
    optimizer = GridSearchOptimizer(
        param_grid={"combo_id": range(len(valid_combinations))},
        n_splits=5
    )
    
    # Custom evaluation
    def evaluate_combo(combo_id):
        return cross_validate_simple(
            matches_df=matches_df,
            engine_params=valid_combinations[combo_id]
        )
    
    return optimizer.fit_custom(evaluate_combo)
```

## Performance Optimization

### 1. Parallel Grid Search

```python
from joblib import Parallel, delayed

def parallel_grid_search(matches_df, param_grid, n_jobs=-1):
    """Parallelize grid search evaluation."""
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    # Define evaluation function
    def evaluate_params(params):
        param_dict = dict(zip(param_names, params))
        
        try:
            cv_results = cross_validate_simple(
                matches_df=matches_df,
                engine_params=param_dict,
                n_splits=5
            )
            return {
                "params": param_dict,
                "loss": cv_results["avg_loss"],
                "std": cv_results["std_loss"]
            }
        except Exception as e:
            return {
                "params": param_dict,
                "error": str(e)
            }
    
    # Parallel evaluation
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(params) 
        for params in all_combinations
    )
    
    # Find best
    valid_results = [r for r in results if "error" not in r]
    best = min(valid_results, key=lambda x: x["loss"])
    
    return best, results
```

### 2. Early Stopping

```python
class EarlyStoppingOptimizer(GridSearchOptimizer):
    """Grid search with early stopping for poor parameters."""
    
    def __init__(self, *args, early_stop_threshold=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stop_threshold = early_stop_threshold
    
    def _evaluate_params(self, params, matches_df):
        """Evaluate with early stopping."""
        
        # Quick evaluation on first fold
        first_fold_result = cross_validate_simple(
            matches_df=matches_df,
            engine_params=params,
            n_splits=1  # Just first fold
        )
        
        # Early stop if too poor
        if first_fold_result["avg_loss"] > self.early_stop_threshold:
            return {
                "params": params,
                "loss": first_fold_result["avg_loss"],
                "early_stopped": True
            }
        
        # Full evaluation
        full_result = cross_validate_simple(
            matches_df=matches_df,
            engine_params=params,
            n_splits=self.n_splits
        )
        
        return {
            "params": params,
            "loss": full_result["avg_loss"],
            "early_stopped": False
        }
```

### 3. Adaptive Sampling

```python
def adaptive_bayesian_optimization(matches_df, param_bounds, budget=100):
    """Adaptive sampling based on uncertainty."""
    
    from skopt import Optimizer
    
    # Initialize optimizer
    opt = Optimizer(
        dimensions=list(param_bounds.values()),
        base_estimator="GP",
        n_initial_points=10
    )
    
    # Adaptive sampling
    for i in range(budget):
        # Get next point to evaluate
        if i < 10:
            # Random exploration
            x = opt.ask()
        else:
            # Balance exploration and exploitation
            x = opt.ask(strategy="cl_mean")  # Confidence lower bound
        
        # Evaluate
        y = evaluate_params(x, matches_df)
        
        # Tell optimizer about result
        opt.tell(x, y)
        
        # Adaptive behavior
        if i > 20 and i % 10 == 0:
            # Check convergence
            if check_convergence(opt):
                print(f"Converged after {i} iterations")
                break
    
    return opt.get_result()
```

## Visualization

### 1. Parameter Interaction Plots

```python
def plot_parameter_interactions(results_df):
    """Visualize how parameters interact."""
    
    import seaborn as sns
    
    # Pivot for heatmap
    for param1, param2 in combinations(param_grid.keys(), 2):
        pivot = results_df.pivot_table(
            values="mean_loss",
            index=param1,
            columns=param2,
            aggfunc="mean"
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis_r",
            cbar_kws={"label": "CV Loss"}
        )
        plt.title(f"Interaction: {param1} vs {param2}")
        plt.show()
```

### 2. Optimization Progress

```python
def plot_optimization_progress(optimizer):
    """Plot optimization progress over iterations."""
    
    if hasattr(optimizer, "results_"):
        results = optimizer.results_
        
        # Extract losses over time
        losses = [r.get("loss", r.get("mean_loss", np.inf)) 
                 for r in results]
        
        # Cumulative minimum
        cum_min = np.minimum.accumulate(losses)
        
        plt.figure(figsize=(12, 6))
        
        # Loss over iterations
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(losses)), losses, alpha=0.5)
        plt.plot(cum_min, 'r-', linewidth=2, label="Best so far")
        plt.xlabel("Iteration")
        plt.ylabel("CV Loss")
        plt.title("Optimization Progress")
        plt.legend()
        
        # Parameter exploration
        plt.subplot(1, 2, 2)
        param_names = list(optimizer.param_grid.keys())[:2]  # First 2 params
        if len(param_names) == 2:
            x = [r["params"][param_names[0]] for r in results]
            y = [r["params"][param_names[1]] for r in results]
            colors = losses
            
            scatter = plt.scatter(x, y, c=colors, cmap="viridis_r", alpha=0.7)
            plt.colorbar(scatter, label="CV Loss")
            plt.xlabel(param_names[0])
            plt.ylabel(param_names[1])
            plt.title("Parameter Space Exploration")
        
        plt.tight_layout()
        plt.show()
```

## Best Practices

### 1. Start Simple

```python
# Quick initial search
quick_results = optimize_rating_engine(
    matches_df=matches.sample(n=10000),  # Subsample for speed
    method="grid",
    param_space={
        "decay_half_life_days": [7, 30, 90],
        "damping_factor": [0.8, 0.9]
    },
    n_splits=3
)

# Then refine with full data
final_results = optimize_rating_engine(
    matches_df=matches,  # Full dataset
    method="bayesian",
    param_space={
        "decay_half_life_days": (
            quick_results["best_params"]["decay_half_life_days"] - 20,
            quick_results["best_params"]["decay_half_life_days"] + 20
        ),
        "damping_factor": (
            quick_results["best_params"]["damping_factor"] - 0.1,
            quick_results["best_params"]["damping_factor"] + 0.1
        )
    },
    n_splits=5
)
```

### 2. Validate Optimized Parameters

```python
# Don't just trust CV results
best_params = optimizer.best_params_

# Validate on completely held-out data
validation_results = cross_validate_simple(
    matches_df=validation_matches,  # Separate validation set
    engine_params=best_params,
    n_splits=1  # Single evaluation
)

print(f"CV Loss: {optimizer.best_score_:.4f}")
print(f"Validation Loss: {validation_results['avg_loss']:.4f}")

# Check for overfitting
if validation_results['avg_loss'] > optimizer.best_score_ * 1.1:
    print("Warning: Possible overfitting detected")
```

### 3. Document Results

```python
# Save optimization results
optimization_report = {
    "date": datetime.now().isoformat(),
    "dataset": {
        "n_matches": len(matches_df),
        "date_range": (
            matches_df["date"].min(),
            matches_df["date"].max()
        )
    },
    "optimization": {
        "method": "grid",
        "param_space": param_grid,
        "n_evaluations": len(optimizer.results_),
        "cv_splits": optimizer.n_splits
    },
    "results": {
        "best_params": optimizer.best_params_,
        "best_cv_loss": optimizer.best_score_,
        "validation_loss": validation_results['avg_loss']
    },
    "all_results": optimizer.get_results_summary().to_dict()
}

# Save to file
with open("optimization_results.json", "w") as f:
    json.dump(optimization_report, f, indent=2)
```

## Next Steps

- [API Reference](../api/optimization.md) - Complete function documentation
- [Examples](../examples/optimization-examples.md) - Real optimization workflows
- [Performance Tuning](../advanced/performance.md) - Speed up optimization