#!/usr/bin/env python
"""
Command-line interface for the continuous tournament scraper.

Usage:
    python -m rankings.continuous.cli [options]
    
Examples:
    # Run continuously with default settings (hourly)
    python -m rankings.continuous.cli
    
    # Run once
    python -m rankings.continuous.cli --once
    
    # Custom interval (30 minutes)
    python -m rankings.continuous.cli --interval 30
    
    # With custom directories
    python -m rankings.continuous.cli --output-dir data/tournaments --state-file data/state.json
    
    # Show status only
    python -m rankings.continuous.cli --status
"""

import argparse
import logging
import sys
from pathlib import Path

from rankings.continuous.manager import ContinuousScraper
from rankings.continuous.strategies import ScrapingStrategy


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("continuous_scraper.log"),
        ],
    )


def main() -> None:
    """Main entry point for the continuous scraper CLI."""
    parser = argparse.ArgumentParser(
        description="Continuous tournament scraper for Sendou.ink",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Operation mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--once",
        action="store_true",
        help="Run a single scraping cycle and exit",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show current scraper status and exit",
    )

    # Scraping configuration
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Minutes between scraping cycles (default: 60)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        help="Maximum number of cycles to run (default: unlimited)",
    )

    # File paths
    parser.add_argument(
        "--output-dir",
        default="data/tournaments",
        help="Directory to save tournament data (default: data/tournaments)",
    )
    parser.add_argument(
        "--state-file",
        default="data/tournament_state.json",
        help="File to persist tournament state (default: data/tournament_state.json)",
    )

    # Strategy configuration
    parser.add_argument(
        "--in-progress-interval",
        type=int,
        default=15,
        help="Minutes between checks for in-progress tournaments (default: 15)",
    )
    parser.add_argument(
        "--scheduled-interval",
        type=int,
        default=60,
        help="Minutes between checks for scheduled tournaments (default: 60)",
    )
    parser.add_argument(
        "--completed-interval",
        type=int,
        default=1440,
        help="Minutes between checks for completed tournaments (default: 1440/24h)",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=20,
        help="Number of IDs to look ahead for discovery (default: 20)",
    )
    parser.add_argument(
        "--burst-size",
        type=int,
        default=20,
        help="Maximum tournaments to scrape per cycle (default: 20)",
    )
    parser.add_argument(
        "--max-requests-hour",
        type=int,
        default=500,
        help="Maximum API requests per hour (default: 500)",
    )

    # Other options
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--start-id",
        type=int,
        help="Start discovery from a specific tournament ID",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Create strategy from arguments
    strategy = ScrapingStrategy(
        in_progress_interval=args.in_progress_interval,
        scheduled_interval=args.scheduled_interval,
        completed_interval=args.completed_interval,
        lookahead_ids=args.lookahead,
        burst_size=args.burst_size,
        max_requests_per_hour=args.max_requests_hour,
    )

    # Create scraper
    scraper = ContinuousScraper(
        output_dir=args.output_dir,
        state_file=args.state_file,
        strategy=strategy,
    )

    # Handle start ID if provided
    if args.start_id:
        logger.info(f"Setting starting tournament ID to {args.start_id}")
        # Ensure the ID is in the state tracker
        scraper.state_tracker.update_tournament(args.start_id)

    # Execute based on mode
    if args.status:
        # Show status and exit
        status = scraper.get_status()
        print("\n=== Continuous Scraper Status ===")
        print(f"Total tournaments tracked: {status['total_tournaments']}")
        print(f"Highest tournament ID: {status['highest_id']}")
        print(f"Active tournaments: {status['active_tournaments']}")
        print(f"Requests this hour: {status['requests_this_hour']}")
        print("\nTournaments by state:")
        for state, count in status["states"].items():
            print(f"  {state}: {count}")

    elif args.once:
        # Run single cycle
        logger.info("Running single scraping cycle")
        results = scraper.run_once()
        print(f"\nCycle complete: {results}")

    else:
        # Run continuously
        logger.info(
            f"Starting continuous scraping with {args.interval} minute intervals"
        )
        try:
            scraper.run_continuous(
                interval_minutes=args.interval, max_cycles=args.max_cycles
            )
        except KeyboardInterrupt:
            logger.info("Scraping stopped by user")
            print("\nScraping stopped")
        finally:
            # Save state before exiting
            scraper.state_tracker.save_state()
            status = scraper.get_status()
            print(f"\nFinal status: {status}")


if __name__ == "__main__":
    main()
