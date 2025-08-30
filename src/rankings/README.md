# Sendou.ink Tournament Rankings

Modern ranking engines and CLI for Sendou.ink tournament data.

## Engines

- ExposureLogOddsEngine (recommended): Exposure‑weighted dual PageRank with log‑odds scoring. Robust to volume/grinding.
- TickTockEngine: Iterative PageRank with tournament influence feedback (“tick‑tock”).
- RatingEngine (compat): Backwards‑compatible facade that delegates to modern engines.

## Quick Start

```python
import json
import polars as pl

from rankings.core import parse_tournaments_data
from rankings.algorithms import ExposureLogOddsEngine, TickTockEngine

# Parse one or more tournament payloads (list[dict])
with open("tournament_data.json") as f:
    payload = json.load(f)

tables = parse_tournaments_data([payload])
matches = tables["matches"]
players = tables["players"]

# Recommended production engine
elo = ExposureLogOddsEngine()
elo_rankings = elo.rank_players(matches, players)  # columns: id, player_rank, win_pr, loss_pr, exposure

# Tick‑tock alternative
tt = TickTockEngine()
tt_rankings = tt.rank_players(matches, players)    # columns: player_id, rating

# Legacy‑style API (compat facade)
from rankings import RatingEngine
re = RatingEngine(beta=1.0, influence_agg_method="top_20_sum")
legacy_rankings = re.rank_players(matches=matches, players=players)  # columns: id, player_rank
```

## CLI

- `rankings_update`: end‑to‑end discovery/scrape/import/rank pipeline (used by CI). See `src/rankings/cli/update.py`.
- `ranked`, `rankings_pull`, `rankings_compile`: helpers for scraping/compilation.

## Configuration

- See `src/rankings/core/config.py` for:
  - `ExposureLogOddsConfig`, `TickTockConfig`
  - Shared `EngineConfig`, `PageRankConfig`, `DecayConfig`
- Engines accept config objects or sensible defaults.

## Notes

- The `rankings.analysis` package is legacy and kept for compatibility; new code should import from `rankings.algorithms`.
- Examples and sampled data should live under `examples/`; large notebooks and artifacts are intentionally excluded.

