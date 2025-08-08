# Evaluation Metrics Documentation

Beyond standard loss functions, the evaluation module provides a comprehensive suite of metrics for assessing ranking system performance from multiple perspectives.

## Overview

The metrics module (`src.rankings.evaluation.metrics_extras`) includes:

1. **Ranking Quality Metrics** - Concordance, correlation measures
2. **Prediction Calibration** - Accuracy at confidence thresholds
3. **Performance Metrics** - Skill scores, upset analysis
4. **Stability Metrics** - Consistency and reliability measures

## Core Metrics

### 1. Concordance

Measures how well rankings preserve pairwise orderings:

```python
from rankings.evaluation.metrics_extras import concordance

# Basic concordance
c_index = concordance(
    matches_df=matches,
    ratings_dict=player_ratings,
    entity_col="player_id"
)
print(f"Concordance: {c_index:.3f}")  # 0.5 = random, 1.0 = perfect
```

#### Weighted Concordance

Weight by match importance or confidence:

```python
# Weight by rating difference magnitude
def weight_by_rating_diff(winner_rating, loser_rating):
    return abs(winner_rating - loser_rating)

weighted_c = concordance(
    matches_df=matches,
    ratings_dict=ratings,
    weight_func=weight_by_rating_diff
)
```

### 2. Skill Score

Improvement over baseline predictors:

```python
from rankings.evaluation.metrics_extras import skill_score

# Compare to 50-50 baseline
ss = skill_score(
    y_true=actual_outcomes,
    y_pred=predictions,
    baseline="uniform"  # Always predict 50%
)
print(f"Skill Score: {ss:.3f}")  # 0 = no skill, 1 = perfect

# Compare to favorite-wins baseline
ss_fav = skill_score(
    y_true=actual_outcomes,
    y_pred=predictions,
    baseline=favorite_baseline_predictions
)
```

### 3. Accuracy at Thresholds

Evaluate prediction accuracy at different confidence levels:

```python
from rankings.evaluation.metrics_extras import accuracy_threshold

# Check if 70% confidence predictions are 70% accurate
acc_70 = accuracy_threshold(
    y_true=actual_outcomes,
    y_pred=predictions,
    threshold=0.7
)
print(f"Accuracy at 70% confidence: {acc_70:.3f}")

# Full calibration curve
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
calibration_curve = [
    accuracy_threshold(actual_outcomes, predictions, t) 
    for t in thresholds
]
```

### 4. Upset Analysis

Analyze model performance on upsets:

```python
from rankings.evaluation.metrics_extras import upset_oe

# Observed/Expected upset ratio
upset_ratio = upset_oe(
    y_true=actual_outcomes,
    y_pred=predictions,
    upset_threshold=0.3  # Prediction < 30% is considered upset
)
print(f"Upset O/E: {upset_ratio:.3f}")  # 1.0 = perfectly calibrated

# Detailed upset analysis
upset_stats = analyze_upsets(
    matches_df=matches,
    predictions=predictions,
    ratings=ratings
)
```

### 5. Placement Correlation

For tournament placement prediction:

```python
from rankings.evaluation.metrics_extras import placement_spearman

# Spearman correlation between predicted and actual placements
placement_corr = placement_spearman(
    predicted_rankings=predicted_placements,
    actual_rankings=actual_placements
)
print(f"Placement correlation: {placement_corr:.3f}")
```

### 6. Alpha Stability

Measure consistency of alpha parameter:

```python
from rankings.evaluation.metrics_extras import alpha_std

# Standard deviation of alpha across CV folds
alpha_stability = alpha_std(
    alpha_values=[1.02, 1.05, 0.98, 1.01, 1.03]
)
print(f"Alpha stability (std): {alpha_stability:.3f}")
```

## Advanced Metrics

### 1. Time-Aware Metrics

Evaluate how performance changes over time:

```python
def compute_temporal_metrics(matches_df, predictions):
    """Compute metrics over rolling time windows."""
    
    # Sort by date
    matches_df = matches_df.sort("date")
    
    # Rolling windows
    window_size = 1000
    metrics_over_time = []
    
    for i in range(window_size, len(matches_df), 100):
        window = matches_df[i-window_size:i]
        window_preds = predictions[i-window_size:i]
        
        metrics = {
            "date": window["date"].max(),
            "concordance": concordance(window, ratings),
            "skill_score": skill_score(window["actual"], window_preds),
            "calibration": accuracy_threshold(window["actual"], window_preds, 0.7)
        }
        metrics_over_time.append(metrics)
    
    return pd.DataFrame(metrics_over_time)
```

### 2. Stratified Metrics

Analyze performance across different segments:

```python
def compute_stratified_metrics(matches_df, predictions, ratings):
    """Compute metrics stratified by various factors."""
    
    results = {}
    
    # By rating difference buckets
    rating_buckets = [(0, 50), (50, 100), (100, 200), (200, float('inf'))]
    for min_diff, max_diff in rating_buckets:
        mask = (
            (matches_df["rating_diff"] >= min_diff) & 
            (matches_df["rating_diff"] < max_diff)
        )
        
        results[f"rating_diff_{min_diff}_{max_diff}"] = {
            "concordance": concordance(matches_df[mask], ratings),
            "accuracy": np.mean(
                (predictions[mask] > 0.5) == matches_df[mask]["actual"]
            ),
            "calibration": accuracy_threshold(
                matches_df[mask]["actual"], 
                predictions[mask], 
                0.7
            )
        }
    
    # By tournament stage
    for stage in ["pools", "bracket", "finals"]:
        mask = matches_df["stage"] == stage
        results[f"stage_{stage}"] = {
            "concordance": concordance(matches_df[mask], ratings),
            "upset_oe": upset_oe(matches_df[mask]["actual"], predictions[mask])
        }
    
    return results
```

### 3. Confidence Interval Metrics

Add uncertainty quantification:

```python
def bootstrap_metric(metric_func, matches_df, predictions, n_bootstrap=1000):
    """Compute metric with bootstrap confidence intervals."""
    
    n = len(matches_df)
    metric_values = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n, size=n, replace=True)
        boot_matches = matches_df[idx]
        boot_preds = predictions[idx]
        
        # Compute metric
        value = metric_func(boot_matches, boot_preds)
        metric_values.append(value)
    
    # Confidence intervals
    return {
        "mean": np.mean(metric_values),
        "std": np.std(metric_values),
        "ci_lower": np.percentile(metric_values, 2.5),
        "ci_upper": np.percentile(metric_values, 97.5)
    }

# Example usage
concordance_ci = bootstrap_metric(
    lambda m, p: concordance(m, ratings),
    matches_df,
    predictions
)
print(f"Concordance: {concordance_ci['mean']:.3f} [{concordance_ci['ci_lower']:.3f}, {concordance_ci['ci_upper']:.3f}]")
```

## Composite Metrics

### 1. Overall Performance Score

Combine multiple metrics into a single score:

```python
def compute_overall_score(matches_df, predictions, ratings):
    """Compute weighted overall performance score."""
    
    # Individual metrics
    metrics = {
        "concordance": concordance(matches_df, ratings),
        "skill_score": skill_score(matches_df["actual"], predictions),
        "calibration": accuracy_threshold(matches_df["actual"], predictions, 0.7),
        "upset_oe": upset_oe(matches_df["actual"], predictions)
    }
    
    # Normalize upset_oe (closer to 1 is better)
    metrics["upset_normalized"] = 1 - abs(metrics["upset_oe"] - 1)
    
    # Weighted combination
    weights = {
        "concordance": 0.3,
        "skill_score": 0.3,
        "calibration": 0.2,
        "upset_normalized": 0.2
    }
    
    overall = sum(
        metrics[key] * weights.get(key, 0) 
        for key in metrics
    )
    
    return overall, metrics
```

### 2. Pareto Frontier Analysis

Find models that excel in multiple dimensions:

```python
def pareto_frontier(results_list):
    """Find non-dominated solutions in multi-objective space."""
    
    # Extract metrics
    points = np.array([
        [r["concordance"], r["skill_score"], -r["loss"]] 
        for r in results_list
    ])
    
    # Find Pareto frontier
    is_pareto = np.ones(len(points), dtype=bool)
    
    for i in range(len(points)):
        if is_pareto[i]:
            # Check if any other point dominates this one
            dominated = np.all(points >= points[i], axis=1)
            dominated[i] = False
            is_pareto[dominated] = False
    
    return [results_list[i] for i in range(len(results_list)) if is_pareto[i]]
```

## Visualization

### 1. Metric Dashboard

```python
import matplotlib.pyplot as plt

def plot_metric_dashboard(cv_results):
    """Create comprehensive metric visualization."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss over folds
    axes[0, 0].plot(cv_results["fold_losses"])
    axes[0, 0].set_title("Loss by Fold")
    axes[0, 0].set_xlabel("Fold")
    axes[0, 0].set_ylabel("Log Loss")
    
    # Concordance distribution
    axes[0, 1].hist(cv_results["fold_concordances"], bins=20)
    axes[0, 1].set_title("Concordance Distribution")
    axes[0, 1].axvline(np.mean(cv_results["fold_concordances"]), color='r', linestyle='--')
    
    # Calibration curve
    thresholds = np.linspace(0.5, 0.95, 10)
    accuracies = [cv_results[f"accuracy_{int(t*100)}"] for t in thresholds]
    axes[0, 2].plot(thresholds, accuracies, 'o-')
    axes[0, 2].plot([0.5, 1], [0.5, 1], 'k--', alpha=0.5)
    axes[0, 2].set_title("Calibration Curve")
    axes[0, 2].set_xlabel("Predicted Probability")
    axes[0, 2].set_ylabel("Actual Win Rate")
    
    # Skill score components
    axes[1, 0].bar(["Model", "Baseline"], 
                   [cv_results["model_loss"], cv_results["baseline_loss"]])
    axes[1, 0].set_title("Model vs Baseline Loss")
    
    # Upset analysis
    axes[1, 1].scatter(cv_results["expected_upsets"], cv_results["observed_upsets"])
    axes[1, 1].plot([0, max(cv_results["expected_upsets"])], 
                    [0, max(cv_results["expected_upsets"])], 'k--')
    axes[1, 1].set_title("Upset Calibration")
    axes[1, 1].set_xlabel("Expected Upsets")
    axes[1, 1].set_ylabel("Observed Upsets")
    
    # Metric correlation heatmap
    metric_names = ["Loss", "Concordance", "Skill", "Calibration"]
    metric_corr = np.corrcoef([
        cv_results["fold_losses"],
        cv_results["fold_concordances"],
        cv_results["fold_skill_scores"],
        cv_results["fold_calibrations"]
    ])
    im = axes[1, 2].imshow(metric_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 2].set_xticks(range(len(metric_names)))
    axes[1, 2].set_yticks(range(len(metric_names)))
    axes[1, 2].set_xticklabels(metric_names)
    axes[1, 2].set_yticklabels(metric_names)
    axes[1, 2].set_title("Metric Correlations")
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    return fig
```

### 2. Temporal Evolution

```python
def plot_temporal_evolution(temporal_metrics):
    """Visualize how metrics evolve over time."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each metric
    metrics_to_plot = ["concordance", "skill_score", "calibration"]
    for metric in metrics_to_plot:
        ax.plot(temporal_metrics["date"], 
                temporal_metrics[metric], 
                label=metric.replace("_", " ").title())
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metric Evolution Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add rolling average
    for metric in metrics_to_plot:
        rolling_avg = temporal_metrics[metric].rolling(window=10).mean()
        ax.plot(temporal_metrics["date"], rolling_avg, 
                '--', alpha=0.5, linewidth=2)
    
    return fig
```

## Best Practices

### 1. Metric Selection

Choose metrics based on your use case:

- **Ranking Systems**: Prioritize concordance and placement correlation
- **Betting/Prediction**: Focus on log loss and calibration
- **Game Balance**: Emphasize upset ratios and skill scores

### 2. Statistical Significance

Always include uncertainty estimates:

```python
# Don't just report point estimates
print(f"Concordance: {c_index:.3f}")  # Bad

# Include confidence intervals
print(f"Concordance: {c_index:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")  # Good
```

### 3. Multiple Testing Correction

When comparing many models:

```python
from statsmodels.stats.multitest import multipletests

# Correct p-values for multiple comparisons
p_values = [compare_models(baseline, model) for model in models]
corrected_p_values = multipletests(p_values, method='bonferroni')[1]
```

## Next Steps

- [Optimization Guide](optimization.md) - Using metrics for model selection
- [API Reference](../api/metrics.md) - Complete function documentation
- [Examples](../examples/metric-analysis.md) - Real-world metric analysis