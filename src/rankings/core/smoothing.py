"""Smoothing strategies for denominator calculations in ranking algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from typing import Any


@runtime_checkable
class SmoothingStrategy(Protocol):
    """Protocol for denominator smoothing strategies."""

    def denom(
        self, loss_weights: np.ndarray, win_weights: np.ndarray
    ) -> np.ndarray:
        """Compute smoothed denominators.

        Args:
            loss_weights: Loss weights.
            win_weights: Win weights.

        Returns:
            Smoothed denominators.
        """
        ...


@dataclass(frozen=True)
class NoSmoothing:
    """No smoothing - use raw loss weights as denominators."""

    def denom(
        self, loss_weights: np.ndarray, win_weights: np.ndarray
    ) -> np.ndarray:
        """Return unsmoothed loss weights.

        Args:
            loss_weights: Loss weights.
            win_weights: Win weights (unused).

        Returns:
            Original loss weights without smoothing.
        """
        return loss_weights


@dataclass(frozen=True)
class WinsProportional:
    """Smoothing proportional to win weights.

    Denominators are computed as: loss_weights + gamma * win_weights
    with optional capping relative to loss_weights.
    """

    gamma: float = 0.02
    cap_ratio: float = 1.0

    def denom(
        self, loss_weights: np.ndarray, win_weights: np.ndarray
    ) -> np.ndarray:
        """Compute denominators with wins-proportional smoothing.

        The smoothing term (gamma * win_weights) can be capped at cap_ratio * loss_weights
        to prevent excessive smoothing for players with many wins but few losses.

        Args:
            loss_weights: Loss weights.
            win_weights: Win weights.

        Returns:
            Smoothed denominators with wins-proportional adjustment.
        """
        raw_denominator = loss_weights + self.gamma * win_weights

        if np.isfinite(self.cap_ratio):
            lambda_smoothing = raw_denominator - loss_weights
            return loss_weights + np.minimum(
                lambda_smoothing, self.cap_ratio * loss_weights
            )

        return raw_denominator


@dataclass(frozen=True)
class ConstantSmoothing:
    """Add a constant smoothing term to denominators.

    Denominators are computed as: loss_weights + epsilon.
    """

    epsilon: float = 1e-6

    def denom(
        self, loss_weights: np.ndarray, win_weights: np.ndarray
    ) -> np.ndarray:
        """Add constant epsilon to loss weights.

        Args:
            loss_weights: Loss weights.
            win_weights: Win weights (unused).

        Returns:
            Loss weights with constant smoothing term added.
        """
        return loss_weights + self.epsilon


@dataclass(frozen=True)
class AdaptiveSmoothing:
    """Adaptive smoothing based on total volume.

    Players with more total games get less smoothing,
    while players with fewer games get more smoothing.
    """

    base_smooth: float = 0.1
    volume_scale: float = 100.0

    def denom(
        self, loss_weights: np.ndarray, win_weights: np.ndarray
    ) -> np.ndarray:
        """Compute adaptive smoothing based on total game volume.

        Smoothing decreases as total games increase.

        Args:
            loss_weights: Loss weights.
            win_weights: Win weights.

        Returns:
            Denominators with volume-adaptive smoothing.
        """
        total_volume = loss_weights + win_weights

        smoothing_factor = self.base_smooth / (
            1 + total_volume / self.volume_scale
        )

        return loss_weights + smoothing_factor * win_weights


@dataclass(frozen=True)
class HybridSmoothing:
    """Hybrid smoothing combining constant and proportional terms.

    Denominators are computed as: loss_weights + epsilon + gamma * win_weights.
    """

    epsilon: float = 1e-6
    gamma: float = 0.01

    def denom(
        self, loss_weights: np.ndarray, win_weights: np.ndarray
    ) -> np.ndarray:
        """Apply both constant and proportional smoothing.

        Args:
            loss_weights: Loss weights.
            win_weights: Win weights.

        Returns:
            Denominators with both constant and proportional smoothing.
        """
        return loss_weights + self.epsilon + self.gamma * win_weights


def get_smoothing_strategy(mode: str, **kwargs: Any) -> SmoothingStrategy:
    """Factory function to get smoothing strategy by name.

    Args:
        mode: Name of smoothing mode.
        **kwargs: Additional parameters for the strategy.

    Returns:
        SmoothingStrategy instance.

    Raises:
        ValueError: If the smoothing mode is unknown.
    """
    strategy_classes = {
        "none": NoSmoothing,
        "wins_proportional": WinsProportional,
        "constant": ConstantSmoothing,
        "adaptive": AdaptiveSmoothing,
        "hybrid": HybridSmoothing,
    }

    strategy_class = strategy_classes.get(mode)
    if strategy_class is None:
        raise ValueError(f"Unknown smoothing mode: {mode}")

    import inspect

    if hasattr(strategy_class, "__dataclass_fields__"):
        valid_fields = strategy_class.__dataclass_fields__.keys()
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in valid_fields
        }
        return strategy_class(**filtered_kwargs)
    else:
        return strategy_class(**kwargs)
