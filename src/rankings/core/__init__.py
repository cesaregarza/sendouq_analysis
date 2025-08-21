"""Core components for ranking algorithms."""

# Keep existing parser import for backward compatibility
from rankings.core.config import (
    DecayConfig,
    EngineConfig,
    ExposureLogOddsConfig,
    PageRankConfig,
    PipelineConfig,
    TickTockConfig,
)
from rankings.core.convert import (
    build_node_mapping,
    convert_matches_dataframe,
    convert_matches_format,
    convert_team_matches,
    factorize_ids,
)
from rankings.core.edges import (
    build_exposure_triplets,
    build_player_edges,
    build_team_edges,
    compute_denominators,
    edges_to_triplets,
    normalize_edges,
)
from rankings.core.influence import (
    aggregate_multi_round_influence,
    compute_retrospective_strength,
    compute_tournament_influence,
    normalize_influence,
)
from rankings.core.pagerank import pagerank_dense, pagerank_sparse
from rankings.core.parser import parse_tournaments_data
from rankings.core.protocols import RatingBackend
from rankings.core.results import (
    BenchmarkResult,
    ExposureLogOddsResult,
    PipelineResult,
    RankResult,
    TickTockResult,
    ValidationResult,
)
from rankings.core.smoothing import (
    AdaptiveSmoothing,
    ConstantSmoothing,
    HybridSmoothing,
    NoSmoothing,
    SmoothingStrategy,
    WinsProportional,
    get_smoothing_strategy,
)
from rankings.core.teleport import (
    ActivePlayersTeleport,
    CustomTeleport,
    TeleportStrategy,
    UniformTeleport,
    VolumeInverseTeleport,
    uniform,
    volume_inverse,
)
from rankings.core.time import (
    Clock,
    add_time_features,
    apply_inactivity_decay,
    compute_decay_factor,
    create_time_windows,
    decay_expr,
    event_ts_expr,
    filter_by_recency,
)

__all__ = [
    # Parser (backward compatibility)
    "parse_tournaments_data",
    # Config
    "DecayConfig",
    "EngineConfig",
    "ExposureLogOddsConfig",
    "PageRankConfig",
    "PipelineConfig",
    "TickTockConfig",
    # Convert
    "build_node_mapping",
    "convert_matches_dataframe",
    "convert_matches_format",
    "convert_team_matches",
    "factorize_ids",
    # Edges
    "build_exposure_triplets",
    "build_player_edges",
    "build_team_edges",
    "compute_denominators",
    "edges_to_triplets",
    "normalize_edges",
    # Influence
    "aggregate_multi_round_influence",
    "compute_retrospective_strength",
    "compute_tournament_influence",
    "normalize_influence",
    # PageRank
    "pagerank_dense",
    "pagerank_sparse",
    # Protocols
    "RatingBackend",
    # Results
    "BenchmarkResult",
    "ExposureLogOddsResult",
    "PipelineResult",
    "RankResult",
    "TickTockResult",
    "ValidationResult",
    # Smoothing
    "AdaptiveSmoothing",
    "ConstantSmoothing",
    "HybridSmoothing",
    "NoSmoothing",
    "SmoothingStrategy",
    "WinsProportional",
    "get_smoothing_strategy",
    # Teleport
    "ActivePlayersTeleport",
    "CustomTeleport",
    "TeleportStrategy",
    "UniformTeleport",
    "VolumeInverseTeleport",
    "uniform",
    "volume_inverse",
    # Time
    "Clock",
    "add_time_features",
    "apply_inactivity_decay",
    "compute_decay_factor",
    "create_time_windows",
    "decay_expr",
    "event_ts_expr",
    "filter_by_recency",
]
