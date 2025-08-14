# Rankings Core

The core module provides the foundational building blocks for the rankings system:

- Parsing Sendou.ink tournament JSON into structured Polars DataFrames
- Centralized configuration constants and default parameters
- Consistent logging utilities for engines and utilities

It is intentionally engine-agnostic and used by both the recommended Exposure Log-Odds engine and the core tick-tock engine.

---

## Modules

- `parser.py`: Normalizes Sendou.ink JSON exports into typed tables used by ranking engines
- `constants.py`: Canonical defaults for decay, PageRank, tournament strength, teleport vectors, thresholds, and scraping
- `logging.py`: Structured logging, timing, progress reporting, and helpers
- `__init__.py`: Public exports for `parse_tournaments_data` and constants

---

## Data model (parser output)

`parse_tournaments_data(tournaments: list[dict]) -> dict[str, polars.DataFrame | None]`

Returns up to seven tables. Empty tables are returned as `None` for simplicity.

- `tournaments`: One row per tournament with rich metadata
  - Keys: `tournament_id`, `event_id`, `name`, `description`, `start_time`, `is_finalized`, `parent_tournament_id`, `discord_url`, `logo_url`, `logo_validated_at`, `logo_src`, `map_picking_style`, `rules`, `tags`, `cast_twitch_accounts`, `org_*`, `author_*`, `staff`, `staff_count`, `settings`, `settings_*`, `sub_count_plus_one`, `sub_count_plus_two`, `casted_matches_count`, `casted_matches_info`, `tie_breaker_map_pool`, `to_set_map_pool`, `bracket_progression_overrides`, `participated_users_count`, `team_count`, `match_count`, `stage_count`, `group_count`, `round_count`
- `stages`: Stage metadata
  - Keys: `tournament_id`, `stage_id`, `stage_name`, `stage_number`, `stage_type`, and flattened `setting_*`
- `groups`: Group membership per stage
  - Keys: `tournament_id`, `stage_id`, `group_id`, `group_number`
- `rounds`: Round-level info
  - Keys: `tournament_id`, `stage_id`, `group_id`, `round_id`, `round_number`, `maps_count`, `maps_type`
- `teams`: Team roster and attributes
  - Keys: `tournament_id`, `team_id`, `team_name`, `seed`, flags like `prefers_not_to_host`, `no_screen`, `dropped_out`, `invite_code`, `created_at`
- `players`: Team members per tournament
  - Keys: `tournament_id`, `team_id`, `user_id`, `username`, `discord_id`, `in_game_name`, `country`, `twitch`, `is_owner`, `roster_created_at`
- `matches`: Normalized match outcomes
  - Keys: `tournament_id`, `stage_id`, `group_id`, `round_id`, `match_id`, `match_number`, `status`, `last_game_finished_at`, `match_created_at`,
    opponent fields `team1_*`, `team2_*`, derived `winner_team_id`, `loser_team_id`, `score_diff`, `total_games`, and `is_bye`

Notes:
- `winner_team_id`/`loser_team_id` are `None` when no recorded win; `is_bye` marks byes/forfeits.
- Timestamps are passed through; engines decide which to prioritize (`last_game_finished_at` or `match_created_at`).

Example:
```python
import json
from rankings.core import parse_tournaments_data

with open("tournament_data.json") as f:
    tournaments = json.load(f)

tables = parse_tournaments_data(tournaments)
matches_df = tables["matches"]
players_df = tables["players"]
```

---

## Configuration defaults (constants)

Time and decay:
- `DEFAULT_REFERENCE_DATE`: timezone-aware baseline for decay
- `DEFAULT_DECAY_HALF_LIFE_DAYS`: default half-life in days (30)
- `DEFAULT_DECAY_RATE`: derived as ln(2)/half-life

PageRank core:
- `DEFAULT_DAMPING_FACTOR` (0.85), `DEFAULT_PAGERANK_TOLERANCE`, `DEFAULT_MAX_ITERATIONS`

Tick-tock engine controls (used by the core engine):
- `DEFAULT_TICK_TOCK_TOLERANCE`, `DEFAULT_MAX_TICK_TOCK`, `DEFAULT_MAX_PAGERANK_ITER`

Tournament strength:
- `DEFAULT_BETA` (strength exponent), `DEFAULT_INFLUENCE_AGG_METHOD` (e.g., `top_20_sum`), `DEFAULT_STRENGTH_AGG`, `DEFAULT_STRENGTH_K`

Teleport vectors:
- `TELEPORT_UNIFORM`, `TELEPORT_VOLUME_INVERSE`, `TELEPORT_VOLUME_MIX`
- `DEFAULT_VOLUME_MIX_ETA`, `DEFAULT_VOLUME_MIX_GAMMA`, `DEFAULT_VOLUME_EPSILON`

Activity thresholds:
- `MIN_TOURNAMENTS_FOR_RANKING`, `MIN_TOURNAMENTS_BEFORE_CV`, `MIN_MATCHES_FOR_EDGE`

Scraping defaults:
- `SENDOU_BASE_URL`, `SENDOU_DATA_SUFFIX`, `CALENDAR_URL`, timeouts/retries/backoff, and batch sizes

Engines read these defaults on initialization; you can override per-instance.

---

## How engines use core

- Both engines take `matches` and (for player mode) `players` tables from the parser.
- Time decay is computed relative to `DEFAULT_REFERENCE_DATE` unless overridden.
- Teleport vectors can be uniform or volume-aware; defaults bias against extreme volume grind.
- Tournament strength is computed from current ratings via aggregation methods (e.g., `top_20_sum`) and fed back into edge weights (core engine) or used to weight exposure (log-odds engine).

Example with recommended engine:
```python
from rankings.core import parse_tournaments_data
from rankings.analysis.engine.exposure_logodds import ExposureLogOddsEngine

tables = parse_tournaments_data(tournaments)
engine = ExposureLogOddsEngine(beta=1.0)
player_rankings = engine.rank_players(tables["matches"], tables["players"])
```

Alternative (core tick-tock engine):
```python
from rankings import RatingEngine

engine = RatingEngine(beta=1.0, influence_agg_method="top_20_sum")
player_rankings = engine.rank_players(tables["matches"], tables["players"])
```

---

## Logging utilities

The logging helpers unify diagnostics across engines and utilities.

```python
from rankings.core.logging import setup_logging, get_logger, log_timing

setup_logging(level="INFO")
logger = get_logger(__name__)

with log_timing(logger, "rank players"):
    rankings = engine.rank_players(matches_df, players_df)
```

Also available: `log_dataframe_stats`, `log_algorithm_convergence`, and the `ProgressLogger` context manager for long operations.

---

## Extending the core

- Parser:
  - Add new fields by extending row dictionaries in `parser.py` for the relevant tables.
  - Keep schemas stable; only add columns, avoid renaming existing ones.
  - Return `None` for empty tables to simplify downstream checks.
- Constants:
  - Add new defaults in `constants.py` and export from `core.__init__`.
  - Choose safe values and document ranges/semantics.
- Logging:
  - Reuse `get_logger(__name__)` within new modules; wrap expensive sections with `log_timing`.

---

## Performance notes

- Polars DataFrames are used for speed and memory efficiency.
- Parsing flattens nested JSON only as needed to retain performance and clarity.
- Engines vectorize critical math (NumPy) and limit iterations with strict tolerances.

---

## Common pitfalls

- Matches with byes/forfeits are flagged via `is_bye` and should be filtered in analysis.
- Some tournaments lack decisive winner/loser info; `winner_team_id`/`loser_team_id` may be `None`.
- User IDs can be strings in synthetic datasets; avoid unsafe casts in engine code.



