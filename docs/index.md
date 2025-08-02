# Sendouq Analysis Documentation

Welcome to the Sendouq Analysis documentation. This system provides comprehensive tools for analyzing and evaluating tournament ranking systems, with a focus on the Splatoon competitive scene.

## Overview

The Sendouq Analysis system is designed to:

- **Evaluate ranking algorithms** using rigorous cross-validation techniques
- **Optimize hyperparameters** for rating systems like Bradley-Terry models
- **Measure prediction accuracy** using various loss functions and metrics
- **Analyze tournament data** to understand player and team performance patterns

## Documentation Structure

### Core Modules

- [Evaluation Module](evaluation/index.md) - Cross-validation, loss functions, and metrics
- [Analysis Module](analysis/index.md) - Rating engines and data processing
- [Core Module](core/index.md) - Constants, logging, and shared utilities

### Getting Started

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Example Workflows](getting-started/examples.md)

### API Reference

- [Cross-Validation API](api/cross-validation.md)
- [Loss Functions API](api/loss-functions.md)
- [Metrics API](api/metrics.md)
- [Optimization API](api/optimization.md)

### Advanced Topics

- [Custom Rating Engines](advanced/custom-engines.md)
- [Performance Tuning](advanced/performance.md)
- [Best Practices](advanced/best-practices.md)

## Key Features

### 1. Cross-Validation Framework

The system provides both simple and advanced cross-validation approaches:

- **Simple CV**: Fast temporal splitting with optimized alpha parameter fitting
- **Advanced CV**: Multiple splitting strategies with comprehensive metric evaluation

### 2. Loss Functions

Multiple loss functions for different evaluation scenarios:

- **Log Loss**: Standard probabilistic loss with confidence weighting
- **Tournament-based Loss**: Evaluates predictions within tournament contexts
- **Weighted Loss**: Incorporates match importance and confidence levels

### 3. Optimization Tools

Automated hyperparameter optimization:

- **Grid Search**: Exhaustive search over parameter combinations
- **Bayesian Optimization**: Intelligent search using Gaussian Processes

### 4. Comprehensive Metrics

Beyond standard accuracy:

- **Concordance**: Ranking correlation metrics
- **Skill Score**: Improvement over baseline predictors
- **Upset Analysis**: Over/under performance evaluation
- **Placement Correlation**: Tournament placement accuracy

## System Architecture

```
sendouq_analysis/
├── src/
│   └── rankings/
│       ├── evaluation/        # Main evaluation module
│       │   ├── cross_validation/
│       │   ├── loss.py
│       │   ├── metrics_extras.py
│       │   └── optimizer.py
│       ├── analysis/          # Rating engines and utilities
│       └── core/             # Shared components
├── tests/                    # Comprehensive test suite
└── docs/                     # This documentation
```

## Quick Example

```python
from src.rankings.evaluation.cross_validation import cross_validate_simple
from src.rankings.evaluation.optimizer import optimize_rating_engine

# Load your tournament data
matches_df = load_matches()
players_df = load_players()

# Run simple cross-validation
results = cross_validate_simple(
    matches_df=matches_df,
    players_df=players_df,
    n_splits=5,
    fit_alpha=True
)

# Optimize hyperparameters
best_params = optimize_rating_engine(
    matches_df=matches_df,
    players_df=players_df,
    method="grid",
    n_splits=5
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on:

- Code style and standards
- Testing requirements
- Documentation guidelines
- Pull request process

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Support

For questions, issues, or feature requests:

- GitHub Issues: [Report an issue](https://github.com/sendouq/analysis/issues)
- Documentation: You're reading it!
- Examples: See the [examples directory](../examples/)