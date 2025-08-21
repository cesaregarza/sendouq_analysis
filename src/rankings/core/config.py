"""Configuration dataclasses for ranking algorithms."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DecayConfig:
    """Configuration for time decay calculations."""

    half_life_days: float = 30.0

    @property
    def decay_rate(self) -> float:
        """Calculate decay rate from half-life."""
        import numpy as np

        return (
            np.log(2) / self.half_life_days if self.half_life_days > 0 else 0.0
        )


@dataclass
class PageRankConfig:
    """Configuration for PageRank algorithm."""

    alpha: float = 0.85
    tol: float = 1e-8
    max_iter: int = 200
    orientation: str = "row"  # "row" or "col" stochastic
    redistribute_dangling: bool = True


@dataclass
class EngineConfig:
    """Common configuration for ranking engines."""

    # Beta parameter for tournament influence weighting
    # 0.0 = no influence effect, 1.0 = full influence effect
    beta: float = 1.0

    # Minimum exposure threshold
    min_exposure: Optional[float] = None

    # Score decay parameters
    score_decay_delay_days: float = 30.0
    score_decay_rate: float = 0.01

    # Smoothing parameter
    lambda_smooth: Optional[float] = None

    # Gamma for wins-proportional smoothing
    gamma: float = 0.02

    # Cap ratio for smoothing
    cap_ratio: float = 1.0

    # Verbose output
    verbose: bool = False


@dataclass
class TickTockConfig:
    """Configuration specific to Tick-Tock algorithm."""

    # Core configurations
    engine: EngineConfig = field(default_factory=EngineConfig)
    pagerank: PageRankConfig = field(default_factory=PageRankConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)

    # Tick-tock specific
    max_ticks: int = 5  # Match old default of max_tick_tock=5
    convergence_tol: float = (
        1e-4  # Match old default of tick_tock_stabilize_tol=1e-4
    )

    # Teleport mode
    teleport_mode: str = "volume_inverse"  # Match old default, options: "uniform" or "volume_inverse"

    # Smoothing mode
    smoothing_mode: str = "wins_proportional"

    # Influence aggregation
    influence_method: str = "arithmetic"  # "arithmetic" (mean), "geometric", "top_20_sum", "top_20_geom"

    # Retrospective strength
    compute_retrospective: bool = True
    retrospective_decay_rate: float = 0.005


@dataclass
class ExposureLogOddsConfig:
    """Configuration specific to Exposure Log-Odds algorithm."""

    # Core configurations
    engine: EngineConfig = field(default_factory=EngineConfig)
    pagerank: PageRankConfig = field(default_factory=PageRankConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)

    # Tick-tock settings for TTL
    tick_tock: TickTockConfig = field(default_factory=TickTockConfig)

    # Exposure specific
    use_surprisal: bool = False
    surprisal_mode: str = "basic"  # "basic", "weighted", "adaptive"

    # Active players determination
    use_tick_tock_active: bool = True
    active_threshold_days: float = 90.0

    # Lambda selection
    lambda_mode: str = "auto"  # "auto", "fixed", "adaptive"
    fixed_lambda: Optional[float] = None

    # Score transformation
    apply_log_transform: bool = True
    score_scale: float = 1.0


@dataclass
class PipelineConfig:
    """Configuration for the full ranking pipeline."""

    # Algorithm selection
    algorithm: str = "tick_tock"  # "tick_tock" or "exposure_log_odds"

    # Algorithm-specific configs
    tick_tock: TickTockConfig = field(default_factory=TickTockConfig)
    exposure_log_odds: ExposureLogOddsConfig = field(
        default_factory=ExposureLogOddsConfig
    )

    # Post-processing
    apply_grades: bool = True
    grade_thresholds: list = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9]
    )

    # Output options
    include_diagnostics: bool = False
    include_intermediate: bool = False

    # Performance options
    use_sparse: bool = True
    parallel: bool = False
    cache_results: bool = False


def merge_configs(*configs: dict) -> dict:
    """
    Merge multiple configuration dictionaries.

    Later configs override earlier ones.
    """
    result = {}
    for config in configs:
        result.update(config)
    return result
