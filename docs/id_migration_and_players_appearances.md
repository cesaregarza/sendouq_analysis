**ID Strategy + Players Route Integration**

This document summarizes the recent changes that (a) integrate Sendou’s new players/appearances route into the ranked scraping/engine, and (b) introduce canonical UUID identifiers across core entities while preserving backwards compatibility with existing integer IDs.

**Goals**
- Decouple internal identity from Sendou’s IDs so we can ingest from multiple providers.
- Use actual per‑match player appearances when building exposure/log‑odds edges (no more full-roster edges).
- Keep the system working for existing flows and data.

**Highlights**
- Scraping now enriches tournament payloads with per‑match player appearances from a public API route.
- Conversion and the Exposure Log‑Odds engine prefer appearances over team rosters when available.
- Canonical UUID columns were added to all core entities; imports populate deterministic UUIDs and cross‑reference them in `external_ids`.

---

**Old vs New: Schema (Core Entities)**

Old (selected columns; primary integer IDs only)
- `tournaments`:
  - `tournament_id` (PK, bigint)
  - metadata fields (name, description, start_time_ms, is_finalized, is_ranked, …)
- `players`:
  - `player_id` (PK, bigint)
  - `display_name`, `discord_id`, `country`, …
- `tournament_teams`:
  - `team_id` (PK, bigint)
  - `tournament_id` (FK), `name`, `seed`, …
- `matches`:
  - `match_id` (PK, bigint)
  - `tournament_id`, `stage_id`, `group_id`, `round_id`, `winner_team_id`, `loser_team_id`, …
- `stages` / `groups` / `rounds`:
  - `stage_id` / `group_id` / `round_id` (PK, bigint) + FKs + metadata
- `external_ids`:
  - `alias_id` (PK), `entity_type`, `internal_id`, `provider`, `external_id`

New (adds canonical UUIDs, keeps old integer IDs)
- `tournaments`:
  - `tournament_id` (PK, bigint)
  - `tournament_uuid` (UUID, unique, nullable initially)
- `players`:
  - `player_id` (PK, bigint)
  - `player_uuid` (UUID, unique, nullable initially)
- `tournament_teams`:
  - `team_id` (PK, bigint)
  - `team_uuid` (UUID, unique, nullable initially)
- `matches`:
  - `match_id` (PK, bigint)
  - `match_uuid` (UUID, unique, nullable initially)
- `stages` / `groups` / `rounds`:
  - `stage_uuid` / `group_uuid` / `round_uuid` (UUID, unique, nullable initially)
- `external_ids`:
  - Added `internal_uuid` (UUID, nullable) and index on `(entity_type, internal_uuid)`

Reasoning
- UUIDs provide provider‑agnostic identity. We can later make these the primary keys or add foreign keys on UUIDs without breaking existing flows.
- We keep integer IDs for compatibility during the transition.
- Deterministic UUIDs (see below) ensure reproducible backfills and allow safe re‑ingestion.

Population Strategy
- Importer generates deterministic UUIDs using `uuid5(NAMESPACE_URL, "sendou:{entity}:{id}")` for all entities where a Sendou ID exists.
- Example: `sendou:tournament:12345` → UUIDv5.
- Writes both the integer `internal_id` and the new `internal_uuid` to `external_ids` for cross‑reference.

Backfill Procedure
- Re-run JSON import on existing files; the importer adds missing columns and populates UUIDs automatically:
  - `poetry run rankings_import --data-dir data/tournaments`
  - or via compile pipeline with `--import`.

---

**Players/Appearances: Scraper → Converter → Engine**

Old Behavior
- Player edges used full team rosters for every match.
- Exposure/Log‑Odds would create cross‑team edges between all roster members, including non‑participants.

New Behavior
- Scraping:
  - Added `fetch_tournament_players(tournament_id)` to call `GET {SENDOU_PUBLIC_API_BASE_URL}/tournament/{id}/players` with `SENDOU_KEY` Bearer auth.
  - Batch scraper enriches each tournament payload by attaching the players route JSON under `player_matches`.
  - Improved logging includes the attempted URL for easier diagnosis.
- Storage/Loader:
  - `load_match_appearances(data_dir)` parses `player_matches` out of saved JSON into a DF: `(tournament_id, match_id, team_id, user_id)`.
- Converter:
  - `convert_matches_dataframe(..., appearances=None)` now accepts an appearances DF and uses it to set `winners`/`losers` per match.
  - Falls back to rosters if an appearance list is missing for a side.
  - Downstream columns unchanged (winners, losers, weight, share, ts).
- Engine:
  - `ExposureLogOddsEngine.rank_players(..., appearances=None)` passes appearances to the converter.
  - `cli/compile.py` loads appearances from scraped JSON and supplies them when ranking.

Reasoning
- Edges now connect only players who actually played. This reduces noise and teammate inflation in exposure/log‑odds.
- Optional parameter keeps older data usable (no appearances → roster fallback).

Operational Notes
- Requires `SENDOU_KEY` to enrich with the public players route; otherwise the pipeline continues with roster fallback.
- Appearance payloads are parsed best‑effort (common shapes supported). If upstream shape changes, the parser can be extended.

---

**New Scripts & Helpers**

Scrape Existing Tournaments with Players Data
- File: `codex_scripts/scrape_existing_tournaments.py`
- Purpose: Backfill/enrich local JSON by re‑scraping tournaments known to the DB, attaching `player_matches`.
- Behavior:
  - Loads DB config from `.env` (non‑destructive) and queries `{SCHEMA}.tournaments` for IDs, with filters.
  - Writes to a dated subfolder under `--data-dir` by default (e.g., `players_refresh_YYYYMMDD_HHMMSS`).
  - Flags: `--only-ranked`, `--include-unfinalized`, `--since-days`, `--until-days`, `--limit`, `--shuffle`, `--dry-run`, `--batch-size`.

Notebook DB Helper
- File: `codex_scripts/notebook_db.py`
- Purpose: Load tables directly in notebooks without psql.
- Helpers: `load_table`, `load_tournaments`, `load_matches`, `load_players`, `load_roster_entries`, `load_teams`, `load_player_rankings`, `load_core_tables`, `list_tables`, `head`.

---

**Touched Code (Traceability)**
- Scraping
  - `src/rankings/scraping/calendar_api.py`: `fetch_tournament_players()`
  - `src/rankings/scraping/batch.py`: attach `player_matches`; improved URL logging
  - `src/rankings/scraping/storage.py`: `load_match_appearances()` parser
- Conversion & Engine
  - `src/rankings/core/convert.py`: appearances support in `convert_matches_dataframe`
  - `src/rankings/algorithms/exposure_log_odds.py`: `appearances` kwarg in `rank_players`
  - `src/rankings/cli/compile.py`: load and pass appearances into engine
- Scripts
  - `codex_scripts/scrape_existing_tournaments.py`: DB‑driven scrape/enrich
  - `codex_scripts/notebook_db.py`: notebook loaders
- Schema & Import
  - `src/rankings/sql/models.py`: add `*_uuid` columns, `external_ids.internal_uuid`
  - `src/rankings/cli/db_import.py`: `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, deterministic UUID population, `external_ids` with `internal_uuid`

---

**Migration Guidance**
- Backfill UUIDs:
  - Re‑run the JSON import. It will add columns and populate UUIDs deterministically.
- Keep using integer IDs where existing code expects them; adopt UUIDs in new code paths and joins when convenient.
- Cross‑reference provider IDs via `external_ids` using either `internal_id` (int) or `internal_uuid` (UUID) as you transition.

**Risks & Limitations**
- Players route auth (SENDOU_KEY) is required to enrich; otherwise the pipeline gracefully falls back to rosters.
- The current appearances parser supports common payload shapes; if upstream changes, it may need minor updates.
- UUID columns are nullable initially to avoid breaking existing inserts; once adoption is complete, we can enforce NOT NULL/foreign keys on UUIDs.

**Next Steps (Optional)**
- Make UUIDs first‑class join keys in loaders and engines; add UUID foreign keys.
- Persist appearances in DB (e.g., `match_appearances`) for durability instead of rereading JSON at compile time.
- Add provider‑agnostic UUID generation for additional ingest sources (start.gg, battlefy, challonge) using the same uuid5 strategy.

