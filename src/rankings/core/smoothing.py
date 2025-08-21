"""Smoothing strategies for denominator calculations in ranking algorithms."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SmoothingStrategy(Protocol):
    """Protocol for denominator smoothing strategies."""

    def denom(self, W_loss: np.ndarray, W_win: np.ndarray) -> np.ndarray:
        """
        Compute smoothed denominators.

        Args:
            W_loss: Loss weights
            W_win: Win weights

        Returns:
            Smoothed denominators
        """
        ...


@dataclass(frozen=True)
class NoSmoothing:
    """No smoothing - use raw loss weights as denominators."""

    def denom(self, W_loss: np.ndarray, W_win: np.ndarray) -> np.ndarray:
        """Return unsmoothed loss weights."""
        return W_loss


@dataclass(frozen=True)
class WinsProportional:
    """
    Smoothing proportional to win weights.

    Denominators are computed as: W_loss + gamma * W_win
    with optional capping relative to W_loss.
    """

    gamma: float = 0.02
    cap_ratio: float = 1.0

    def denom(self, W_loss: np.ndarray, W_win: np.ndarray) -> np.ndarray:
        """
        Compute denominators with wins-proportional smoothing.

        The smoothing term (gamma * W_win) can be capped at cap_ratio * W_loss
        to prevent excessive smoothing for players with many wins but few losses.
        """
        raw = W_loss + self.gamma * W_win

        if np.isfinite(self.cap_ratio):
            # denom = W_loss + min(lambda, cap_ratio * W_loss)
            lambda_term = raw - W_loss
            return W_loss + np.minimum(lambda_term, self.cap_ratio * W_loss)

        return raw


@dataclass(frozen=True)
class ConstantSmoothing:
    """
    Add a constant smoothing term to denominators.

    Denominators are computed as: W_loss + epsilon
    """

    epsilon: float = 1e-6

    def denom(self, W_loss: np.ndarray, W_win: np.ndarray) -> np.ndarray:
        """Add constant epsilon to loss weights."""
        return W_loss + self.epsilon


@dataclass(frozen=True)
class AdaptiveSmoothing:
    """
    Adaptive smoothing based on total volume.

    Players with more total games get less smoothing,
    while players with fewer games get more smoothing.
    """

    base_smooth: float = 0.1
    volume_scale: float = 100.0

    def denom(self, W_loss: np.ndarray, W_win: np.ndarray) -> np.ndarray:
        """
        Compute adaptive smoothing based on total game volume.

        Smoothing decreases as total games increase.
        """
        total_volume = W_loss + W_win

        # Smoothing factor decreases with volume
        smooth_factor = self.base_smooth / (
            1 + total_volume / self.volume_scale
        )

        return W_loss + smooth_factor * W_win


@dataclass(frozen=True)
class HybridSmoothing:
    """
    Hybrid smoothing combining constant and proportional terms.

    Denominators are computed as: W_loss + epsilon + gamma * W_win
    """

    epsilon: float = 1e-6
    gamma: float = 0.01

    def denom(self, W_loss: np.ndarray, W_win: np.ndarray) -> np.ndarray:
        """Apply both constant and proportional smoothing."""
        return W_loss + self.epsilon + self.gamma * W_win


def get_smoothing_strategy(mode: str, **kwargs) -> SmoothingStrategy:
    """
    Factory function to get smoothing strategy by name.

    Args:
        mode: Name of smoothing mode
        **kwargs: Additional parameters for the strategy

    Returns:
        SmoothingStrategy instance
    """
    strategies = {
        "none": NoSmoothing,
        "wins_proportional": WinsProportional,
        "constant": ConstantSmoothing,
        "adaptive": AdaptiveSmoothing,
        "hybrid": HybridSmoothing,
    }

    strategy_class = strategies.get(mode)
    if strategy_class is None:
        raise ValueError(f"Unknown smoothing mode: {mode}")

    # Filter kwargs to only those accepted by the strategy
    import inspect

    if hasattr(strategy_class, "__dataclass_fields__"):
        valid_fields = strategy_class.__dataclass_fields__.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        return strategy_class(**filtered_kwargs)
    else:
        return strategy_class(**kwargs)
