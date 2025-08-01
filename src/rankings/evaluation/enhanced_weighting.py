"""
Enhanced weighting schemes for match loss computation.

These schemes adjust the influence of matches based on prediction confidence,
reducing the impact of coin-flip matches and increasing the impact of
matches with clear favorites.
"""

from typing import Literal

import numpy as np


def compute_confidence_weights(
    predictions: np.ndarray,
    scheme: Literal[
        "none",
        "entropy",
        "var_inv",
        "entropy_squared",
        "entropy_exp",
        "threshold",
    ] = "entropy",
    threshold: float = 0.1,
    exp_base: float = 2.0,
) -> np.ndarray:
    """
    Compute confidence-based weights for match predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities (0 to 1)
    scheme : str
        Weighting scheme:
        - "none": Uniform weights (no confidence weighting)
        - "entropy": Linear distance from 0.5: w = 2|p - 0.5|
        - "entropy_squared": Squared distance: w = 4(p - 0.5)²
        - "entropy_exp": Exponential: w = exp_base^(2|p - 0.5|) - 1
        - "var_inv": Inverse variance: w = 1 / (p(1-p))
        - "threshold": Binary threshold: w = 1 if |p-0.5| > threshold, else 0
    threshold : float
        For threshold scheme, minimum distance from 0.5 to include
    exp_base : float
        Base for exponential scheme

    Returns
    -------
    np.ndarray
        Weight for each prediction
    """
    if scheme == "none":
        return np.ones_like(predictions)

    elif scheme == "entropy":
        # Linear distance from 0.5
        # w = 0 at p=0.5, w = 1 at p=0 or p=1
        w = 2.0 * np.abs(predictions - 0.5)

    elif scheme == "entropy_squared":
        # Squared distance from 0.5
        # More aggressive down-weighting of near-0.5 matches
        # w = 0 at p=0.5, w = 1 at p=0 or p=1
        w = 4.0 * (predictions - 0.5) ** 2

    elif scheme == "entropy_exp":
        # Exponential scaling
        # Even more aggressive down-weighting
        distance = 2.0 * np.abs(predictions - 0.5)  # 0 to 1
        w = exp_base**distance - 1  # 0 at p=0.5, exp_base-1 at p=0/1
        w = w / (exp_base - 1)  # Normalize to 0-1 range

    elif scheme == "var_inv":
        # Inverse variance weighting
        # Based on binomial variance p(1-p)
        # Highest weight at p=0 or p=1, lowest at p=0.5
        variance = predictions * (1 - predictions)
        w = 1.0 / np.maximum(variance, 1e-10)
        # Normalize to reasonable range
        w = w / 4.0  # Max variance is 0.25 at p=0.5

    elif scheme == "threshold":
        # Binary threshold
        # Only include matches sufficiently far from 0.5
        w = (np.abs(predictions - 0.5) > threshold).astype(float)

    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    # Ensure positive weights
    w = np.maximum(w, 1e-10)

    return w


def analyze_weight_distribution(
    predictions: np.ndarray, weights: np.ndarray, n_bins: int = 10
) -> dict:
    """
    Analyze how weights are distributed across prediction ranges.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities
    weights : np.ndarray
        Corresponding weights
    n_bins : int
        Number of bins for analysis

    Returns
    -------
    dict
        Analysis results with bins and statistics
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    results = {"bins": [], "total_weight_fraction": []}

    total_weight = np.sum(weights)

    for i in range(n_bins):
        mask = bin_indices == i
        count = np.sum(mask)

        if count > 0:
            bin_weights = weights[mask]
            bin_preds = predictions[mask]

            results["bins"].append(
                {
                    "range": f"[{bins[i]:.2f}-{bins[i+1]:.2f})",
                    "count": int(count),
                    "mean_pred": float(np.mean(bin_preds)),
                    "mean_weight": float(np.mean(bin_weights)),
                    "total_weight": float(np.sum(bin_weights)),
                    "weight_fraction": float(
                        np.sum(bin_weights) / total_weight
                    ),
                }
            )

    # Add summary stats
    results["summary"] = {
        "total_matches": len(predictions),
        "effective_matches": float(np.sum(weights) / np.mean(weights)),
        "weight_concentration": float(np.std(weights) / np.mean(weights)),  # CV
        "near_coinflip_fraction": float(
            np.mean(np.abs(predictions - 0.5) < 0.1)
        ),
    }

    return results


def visualize_weighting_schemes(
    p_range: np.ndarray = None, schemes: list[str] = None
) -> dict:
    """
    Generate data for visualizing different weighting schemes.

    Parameters
    ----------
    p_range : np.ndarray, optional
        Probability range to visualize (default: 0 to 1)
    schemes : list[str], optional
        Schemes to compare

    Returns
    -------
    dict
        Visualization data
    """
    if p_range is None:
        p_range = np.linspace(0, 1, 101)

    if schemes is None:
        schemes = ["entropy", "entropy_squared", "entropy_exp", "var_inv"]

    results = {"p": p_range}

    for scheme in schemes:
        if scheme == "threshold":
            # Use default threshold
            weights = compute_confidence_weights(
                p_range, scheme=scheme, threshold=0.1
            )
        else:
            weights = compute_confidence_weights(p_range, scheme=scheme)

        results[scheme] = weights

    return results


# Demo function
if __name__ == "__main__":
    # Example predictions
    test_predictions = np.array([0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95])

    print("Confidence Weighting Schemes Demo")
    print("=" * 50)
    print(f"Test predictions: {test_predictions}")
    print()

    schemes = [
        "entropy",
        "entropy_squared",
        "entropy_exp",
        "var_inv",
        "threshold",
    ]

    for scheme in schemes:
        weights = compute_confidence_weights(test_predictions, scheme=scheme)
        print(f"\n{scheme.upper()}:")
        for p, w in zip(test_predictions, weights):
            print(f"  p={p:.2f} → weight={w:.3f}")

    # Visualization data
    print("\n\nWeighting Functions (for plotting):")
    print("-" * 50)
    viz_data = visualize_weighting_schemes()

    # Show a few sample points
    indices = [0, 10, 25, 40, 50, 60, 75, 90, 100]
    print("p     entropy  squared  exp      var_inv")
    for i in indices:
        p = viz_data["p"][i]
        print(
            f"{p:.2f}  {viz_data['entropy'][i]:7.3f}  "
            f"{viz_data['entropy_squared'][i]:7.3f}  "
            f"{viz_data['entropy_exp'][i]:7.3f}  "
            f"{viz_data['var_inv'][i]:7.3f}"
        )
