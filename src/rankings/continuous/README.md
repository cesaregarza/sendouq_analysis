# Continuous Tournament Scraping System

A robust, intelligent system for continuously scraping tournament data from Sendou.ink with automatic state tracking, lifecycle management, and adaptive scheduling.

## Overview

The continuous scraping system is designed to run 24/7, automatically discovering new tournaments and tracking them through their complete lifecycle. It handles various edge cases including deleted tournaments, stale scheduled tournaments, and differentiates between "not yet created" and "deleted" 404 responses.

## Features

### 🎯 Intelligent State Tracking
- **Tournament Lifecycle**: Tracks tournaments through states: `SCHEDULED` → `IN_PROGRESS` → `COMPLETED`
- **Special States**: Handles `DELETED` (404s), `STALE` (never started), and `UNKNOWN` (not discovered)
- **Persistent Storage**: Maintains state between restarts using JSON persistence
- **Metadata Tracking**: Records first seen, last modified, error counts, and more

### ⚡ Adaptive Scraping Strategy
- **Dynamic Intervals**: Different check frequencies based on tournament state
  - In-progress: 15 minutes (actively updating)
  - Scheduled: 1 hour (checking if started)
  - Completed: 24 hours (rarely changes)
- **Smart Discovery**: Continuously probes for new tournament IDs
- **404 Intelligence**: Distinguishes between "not yet created" and "deleted" tournaments

### 🛡️ Robust Error Handling
- **Rate Limiting**: Respects API limits (configurable, default 500/hour)
- **Retry Logic**: Smart retries with exponential backoff
- **Graceful Degradation**: Continues operation even with partial failures
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## Installation

The continuous scraping module is part of the `rankings` package:

```bash
# Ensure you have the rankings package installed
pip install -e .

# Or with poetry
poetry install
```

## Quick Start

### Command Line

```bash
# Run continuously with default settings (hourly cycles)
python -m rankings.continuous.cli

# Run a single scraping cycle
python -m rankings.continuous.cli --once

# Check current status
python -m rankings.continuous.cli --status
```

### Python API

```python
from rankings.continuous import ContinuousScraper

# Create and run scraper with defaults
scraper = ContinuousScraper()
scraper.run_continuous(interval_minutes=60)
```

## Usage Examples

### Basic Continuous Scraping

```python
from rankings.continuous import ContinuousScraper

scraper = ContinuousScraper(
    output_dir="data/tournaments",
    state_file="data/tournament_state.json"
)

# Run forever with hourly cycles
scraper.run_continuous(interval_minutes=60)
```

### Custom Strategy Configuration

```python
from rankings.continuous import ContinuousScraper, ScrapingStrategy

# Configure custom intervals and limits
strategy = ScrapingStrategy(
    in_progress_interval=10,      # Check active tournaments every 10 min
    scheduled_interval=30,         # Check scheduled every 30 min
    completed_interval=2880,       # Check completed every 2 days
    lookahead_ids=50,             # Look 50 IDs ahead for discovery
    burst_size=30,                # Scrape up to 30 per cycle
    max_requests_per_hour=1000,   # Higher rate limit
    stale_scheduled_days=5        # Mark as stale after 5 days
)

scraper = ContinuousScraper(strategy=strategy)
scraper.run_continuous()
```

### Single Cycle Operation

```python
from rankings.continuous import ContinuousScraper

scraper = ContinuousScraper()

# Run one cycle and get results
results = scraper.run_once()
print(f"Scraped: {results['scraped']}")
print(f"Failed: {results['failed']}")
print(f"Discovered: {results['discovered']}")

# Check status
status = scraper.get_status()
print(f"Total tournaments tracked: {status['total_tournaments']}")
print(f"Active tournaments: {status['active_tournaments']}")
print(f"States: {status['states']}")
```

### Monitoring and Status

```python
from rankings.continuous import TournamentStateTracker, TournamentState

# Load existing state
tracker = TournamentStateTracker("data/tournament_state.json")

# Get tournaments by state
in_progress = tracker.get_tournaments_by_state(TournamentState.IN_PROGRESS)
print(f"Tournaments in progress: {len(in_progress)}")

# Check specific tournament
meta = tracker.get_tournament(1955)
if meta:
    print(f"Tournament 1955 state: {meta.state.value}")
    print(f"Last checked: {meta.last_checked}")
    print(f"Scrape count: {meta.scrape_count}")

# Get active tournaments
active = tracker.get_active_tournaments()
print(f"Active tournaments to monitor: {len(active)}")
```

## Command Line Interface

### Basic Commands

```bash
# Run continuously (default: hourly)
python -m rankings.continuous.cli

# Run with custom interval (30 minutes)
python -m rankings.continuous.cli --interval 30

# Run single cycle
python -m rankings.continuous.cli --once

# Show status
python -m rankings.continuous.cli --status

# Run for limited cycles
python -m rankings.continuous.cli --max-cycles 10
```

### Advanced Configuration

```bash
# Full configuration example
python -m rankings.continuous.cli \
    --interval 45 \
    --in-progress-interval 10 \
    --scheduled-interval 30 \
    --completed-interval 1440 \
    --lookahead 30 \
    --burst-size 25 \
    --max-requests-hour 800 \
    --output-dir data/tournaments \
    --state-file data/state.json \
    --start-id 2000 \
    --verbose
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--once` | False | Run single cycle and exit |
| `--status` | False | Show status and exit |
| `--interval` | 60 | Minutes between cycles |
| `--max-cycles` | None | Max cycles to run (None = forever) |
| `--output-dir` | data/tournaments | Tournament data directory |
| `--state-file` | data/tournament_state.json | State persistence file |
| `--in-progress-interval` | 15 | Minutes between in-progress checks |
| `--scheduled-interval` | 60 | Minutes between scheduled checks |
| `--completed-interval` | 1440 | Minutes between completed checks |
| `--lookahead` | 20 | IDs to look ahead for discovery |
| `--burst-size` | 20 | Max tournaments per cycle |
| `--max-requests-hour` | 500 | API rate limit per hour |
| `--start-id` | None | Start discovery from specific ID |
| `--verbose` | False | Enable debug logging |

## Architecture

### Module Structure

```
rankings/continuous/
├── __init__.py           # Package exports
├── README.md            # This file
├── cli.py               # Command-line interface
├── manager.py           # Main ContinuousScraper class
├── state.py             # Tournament state tracking
└── strategies.py        # Scraping strategies and prioritization
```

### Key Components

#### TournamentState (Enum)
- `SCHEDULED`: Tournament created but not started
- `IN_PROGRESS`: Tournament currently running
- `COMPLETED`: Tournament finished
- `DELETED`: Tournament was deleted (404)
- `STALE`: Scheduled tournament that never started
- `UNKNOWN`: Not yet discovered

#### TournamentStateTracker
- Maintains persistent record of all tournaments
- Tracks metadata: dates, errors, 404 counts
- Provides methods for querying by state
- Handles state transitions

#### ScrapingStrategy
- Configures intervals for different states
- Sets retry and discovery parameters
- Defines cleanup policies
- Controls rate limiting

#### ScrapingPrioritizer
- Determines which tournaments to scrape
- Orders by priority (in-progress > scheduled > completed)
- Manages discovery of new IDs
- Applies time-based filtering

#### ContinuousScraper
- Main orchestrator class
- Manages scraping cycles
- Handles HTTP requests and responses
- Saves tournament data
- Applies rate limiting

## How It Works

### Tournament Discovery
1. System continuously probes IDs ahead of the highest known ID
2. New tournaments are added to tracking with `UNKNOWN` state
3. First successful scrape determines actual state

### State Determination
The system automatically determines tournament state from API responses:
- Checks `status` field for "COMPLETED"
- Examines brackets and matches for in-progress detection
- Uses start time to identify scheduled tournaments
- Falls back to `SCHEDULED` if uncertain

### 404 Handling Strategy
- Tracks consecutive 404 responses
- Different thresholds for known (10) vs unknown (5) tournaments
- Tournament marked as `DELETED` only after threshold
- Prevents false positives from temporary API issues

### Stale Tournament Cleanup
- Scheduled tournaments are monitored for 7 days after scheduled date
- If not started within this period, marked as `STALE`
- Stale tournaments are ignored in future cycles
- Configurable via `stale_scheduled_days` parameter

### Data Storage
- Tournament data saved to timestamped JSON files
- Format: `tournaments_continuous_YYYYMMDD_HHMMSS.json`
- State tracking persisted separately in `tournament_state.json`
- Both are human-readable JSON for easy inspection

## Monitoring

### Logs
The system produces detailed logs to both console and file:
- Info level: Normal operations, cycle summaries
- Debug level: Detailed scraping actions (use `--verbose`)
- Error level: Failures and exceptions
- Log file: `continuous_scraper.log`

### Status Checking
```bash
# Command line
python -m rankings.continuous.cli --status

# Python
scraper = ContinuousScraper()
status = scraper.get_status()
```

Status includes:
- Total tournaments tracked
- Breakdown by state
- Highest tournament ID seen
- Current request count
- Active tournament count

## Best Practices

### Production Deployment

1. **Use systemd or supervisor** for process management
2. **Set up log rotation** to prevent disk fill
3. **Monitor disk space** for output directory
4. **Configure alerts** for error rates
5. **Regular backups** of state file

### Example systemd Service

```ini
[Unit]
Description=Sendou.ink Tournament Scraper
After=network.target

[Service]
Type=simple
User=scraper
WorkingDirectory=/opt/sendouq_analysis
ExecStart=/usr/bin/python -m rankings.continuous.cli --interval 60
Restart=always
RestartSec=60
StandardOutput=append:/var/log/tournament_scraper.log
StandardError=append:/var/log/tournament_scraper.error.log

[Install]
WantedBy=multi-user.target
```

### Rate Limiting Considerations

- Default: 500 requests/hour (safe for most APIs)
- Burst protection: Max 20 requests in quick succession
- Automatic throttling when limits approached
- Configure based on API documentation

### Storage Management

- Completed tournaments removed from tracking after 30 days
- Consider external archival for long-term storage
- Monitor output directory size
- Implement log rotation for production

## Troubleshooting

### Common Issues

#### High 404 Rate
- **Cause**: Probing too far ahead or API issues
- **Solution**: Reduce `lookahead_ids` parameter

#### Stale Data
- **Cause**: Intervals too long for active tournaments
- **Solution**: Decrease `in_progress_interval`

#### Rate Limiting
- **Cause**: Too many requests per hour
- **Solution**: Increase intervals or reduce `burst_size`

#### Memory Usage
- **Cause**: Too many tournaments in state tracking
- **Solution**: Reduce `cleanup_completed_days` to remove old tournaments sooner

### Debug Mode

Enable verbose logging to troubleshoot issues:
```bash
python -m rankings.continuous.cli --verbose
```

This shows:
- Individual scraping attempts
- State transitions
- Rate limiting decisions
- Discovery process

## Contributing

When contributing to the continuous scraping system:

1. **Maintain backwards compatibility** with state file format
2. **Add tests** for new state transitions
3. **Document** new configuration options
4. **Consider rate limiting** impact of changes
5. **Test long-running stability** before submitting

## License

Part of the sendouq_analysis project. See main project LICENSE file.