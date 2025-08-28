# sendouq-analysis & sendouq-dashboard

Comprehensive data analysis and interactive dashboard tools for tournament and player statistics, built with Python, Dash, and modern data science libraries. Includes advanced ranking algorithms for Sendou.ink tournament data.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Poetry](#poetry)
  - [Docker](#docker)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Command Line Tools](#command-line-tools)
  - [Running the Dashboard](#running-the-dashboard)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Tournament Rankings](#tournament-rankings)
- [Project Structure](#project-structure)
- [Development & Contribution](#development--contribution)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [License](#license)
- [Authors & Acknowledgments](#authors--acknowledgments)

---

## Project Overview

**sendouq-analysis**: Python package for scraping, parsing, and aggregating tournament and player data. Includes utilities for data ingestion, transformation, and statistical analysis.

**sendouq-dashboard**: Interactive dashboard (Dash/Plotly) for visualizing player and tournament statistics, powered by a PostgreSQL backend.

**rankings**: Advanced tournament ranking module. Recommended default is the Exposure Log-Odds Engine which removes volume bias; the core tick-tock engine remains available for comparison.

---

## Architecture

- **Data Pipeline**: Scraping → Ingestion → Aggregation → Dashboard
- **Technologies**: Python 3.10+, pandas, SQLAlchemy, Dash, Plotly, Poetry, Docker
- **Database**: PostgreSQL (cloud or local)

---

## Installation

### Poetry
1. Install [Poetry](https://python-poetry.org/docs/#installation).
2. Clone the repository:
   ```bash
   git clone <repo-url>
   cd sendouq_analysis
   ```
3. Install dependencies:
   ```bash
   poetry install
   ```

### Docker
1. Build the Docker image:
   ```bash
   docker build -t sendouq-dashboard .
   ```
2. (Optional) Run with environment variables (see below):
   ```bash
   docker run --env-file .env -p 8050:8050 sendouq-dashboard
   ```

---

## Environment Setup

Set the following environment variables (in your shell, `.env`, or via Docker):

```
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_DB=your_db_name
POSTGRES_HOST=your_db_host
POSTGRES_PORT=your_db_port
```

For scraping/aggregation on DigitalOcean, set:
```
DO_API_TOKEN=your_digitalocean_token
```

---

## Usage

### Command Line Tools

- **Scrape tournament/match data:**
  ```bash
  poetry run scrape
  # or
  poetry run python src/sendouq_analysis/endpoints/scrape.py
  ```
- **Aggregate player/tournament stats:**
  ```bash
  poetry run aggregate
  # or
  poetry run python src/sendouq_analysis/endpoints/aggregate.py

- **Compile core tables from Postgres (optional scrape + import):**

  - Import local JSONs and write Parquet core tables under `data/compiled`:

    poetry run rankings_compile --import --only-ranked

  - Scrape recent tournaments, import to DB, and compile last 120 days:

    poetry run rankings_compile --scrape ranked --recent-count 200 --import --since-days 120 --only-ranked

  - Add `--run-engine` to compute rankings and save a compact model state for influence queries:

    poetry run rankings_compile --import --only-ranked --run-engine --sslmode disable

  - By default, Parquet snapshots are not written. Use `--write-parquet` to produce `matches.parquet`, `players.parquet`, and with `--run-engine`, `rankings.parquet`. Engine runs always write `engine_state.json`.

Docker
- Weekly scrape:
  - `docker compose -f dockerfiles/docker-compose.weekly.yml up --build`
- Ranked (recent weeks) scrape:
  - `docker compose -f dockerfiles/docker-compose.ranked.yml up --build`
- Update + import + compile + rank + save to DB:
  - `docker compose -f dockerfiles/docker-compose.update.yml up --build`
  - Configure DB via `.env` or env vars: set `RANKINGS_DATABASE_URL` (or `DATABASE_URL`) and `RANKINGS_DB_SCHEMA`.
  - To explicitly tag persisted rankings, set `BUILD_VERSION=my_tag` or pass `--build-version my_tag` via `command`.
  - To emit Parquet snapshots from the updater container, set `WRITE_PARQUET=true`.
  ```

### Running the Dashboard

- **With Poetry:**
  ```bash
  poetry run debug_dashboard
  # or
  poetry run python src/sendouq_dashboard/app.py
  ```
- **With Docker:**
  ```bash
  ./run_dashapp.sh
  # or
  docker run --env-file .env -p 8050:8050 sendouq-dashboard
  ```
- **Directly:**
  ```bash
  python src/sendouq_dashboard/app.py
  # or
  python -m sendouq_dashboard
  ```

Dashboard will be available at [http://localhost:8050](http://localhost:8050).

### Jupyter Notebooks

Explore and analyze data interactively using the provided notebooks (e.g., `test_*.ipynb`).

### Tournament Rankings

The rankings module provides comprehensive tournament ranking capabilities. Recommended engine: Exposure Log-Odds (volume-bias free).

#### Quick Start
```python
import json
from rankings import parse_tournaments_data
from rankings.analysis.engine.exposure_logodds import ExposureLogOddsEngine

# Load and parse tournament data
with open("tournament_data.json") as f:
    raw_data = json.load(f)

tables = parse_tournaments_data(raw_data)
matches_df = tables["matches"]
players_df = tables["players"]

# Recommended: Exposure Log-Odds player rankings
engine = ExposureLogOddsEngine(beta=1.0)
player_rankings = engine.rank_players(matches_df, players_df)

# Access tournament influence/strength
tournament_influence = engine.tournament_influence
tournament_strength = engine.tournament_strength
```

#### Scraping Tournament Data
```python
from rankings import scrape_tournament, scrape_latest_tournaments, scrape_tournaments_from_calendar

# Scrape a specific tournament
tournament_data = scrape_tournament(1955)

# Scrape latest tournaments
results = scrape_latest_tournaments(count=50)

# Discover and scrape tournaments from calendar
calendar_results = scrape_tournaments_from_calendar()
```

#### Key Features
- **Tournament Data Parsing**: Parse Sendou.ink JSON exports into structured DataFrames
- **Team & Player Rankings**: Rank both teams and individual players
- **Time Decay**: Exponential decay weighting for match age (configurable half-life)
- **Tournament Strength**: Dynamic tournament importance calculation
- **Engines**: Exposure Log-Odds (recommended) and core tick-tock engine
- **Scraping Support**: Built-in tournament discovery and batch scraping from Sendou.ink

---

## Project Structure

```
├── src/
│   ├── sendouq_analysis/      # Data scraping, ingestion, aggregation
│   │   ├── scrape_tournament.py
│   │   ├── endpoints/
│   │   ├── compute/
│   │   ├── ingest/
│   │   ├── transforms/
│   │   └── utils.py
│   ├── sendouq_dashboard/     # Dashboard app and data loaders
│   │   ├── app.py
│   │   ├── load/
│   │   ├── constants/
│   │   └── __main__.py
│   └── rankings/              # Tournament ranking algorithms
│       ├── core/              # Core parsing and configuration
│       │   ├── parser.py      # Tournament JSON parsing
│       │   └── constants.py   # Configuration constants
│       ├── scraping/          # Tournament data acquisition
│       │   ├── api.py         # Sendou.ink API interface
│       │   ├── batch.py       # Batch scraping operations
│       │   ├── discovery.py   # Tournament discovery via calendar
│       │   └── storage.py     # Data persistence utilities
│       └── analysis/          # Ranking algorithms
│           ├── engine.py      # Advanced RatingEngine implementation
│           └── utils.py       # Analysis utilities
├── requirements.txt / pyproject.toml / poetry.lock
├── run_dashapp.sh / build_scraper.sh
├── dockerfile
├── data/ / tournament_data.json / tournament_schema.json
├── test_*.ipynb
```

---

## Development & Contribution

1. Fork and clone the repo.
2. Install dependencies with Poetry.
3. Use `black` and `isort` for code formatting:
   ```bash
   poetry run black .
   poetry run isort .
   ```
4. Add tests and notebooks as needed.
5. Submit a pull request with a clear description.

---

## Troubleshooting & FAQ

- **Database connection errors:**
  - Check your environment variables and database status.
- **Dashboard not loading:**
  - Ensure the database is accessible and populated.
- **Docker build issues:**
  - Make sure Docker is running and you have network access.
- **Where is my data stored?**
  - All data is stored in the configured PostgreSQL database.
- **How do I add new visualizations?**
  - Edit or extend `src/sendouq_dashboard/app.py` and related modules.

---

## License

MIT License. See the `LICENSE` file for details.

---

## Authors & Acknowledgments

- **Cesar E Garza** - Initial work
- [Contributors](https://github.com/your-repo/sendouq-analysis/contributors)
- Thanks to the open-source community and all inspiration sources.
