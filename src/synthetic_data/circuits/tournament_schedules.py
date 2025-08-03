"""
Tournament schedule generation utilities.

Provides various pre-configured tournament schedules that mimic real-world
tournament densities and structures.
"""

from datetime import datetime
from typing import Any, Optional

import numpy as np

from synthetic_data.circuits.tournament_circuit import (
    TournamentConfig,
    TournamentType,
)
from synthetic_data.core.tournament_generator import TournamentFormat


class TournamentScheduleGenerator:
    """Utility class for generating various tournament schedules."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize schedule generator.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def create_dense_schedule(
        self,
        days: int = 180,
        min_team_size: int = 8,
        tournaments_per_day: tuple = (1, 5),
        size_distribution: Optional[dict[str, float]] = None,
    ) -> list[TournamentConfig]:
        """
        Create a dense tournament schedule with multiple daily tournaments.

        Parameters
        ----------
        days : int
            Number of days to generate tournaments for
        min_team_size : int
            Minimum number of teams per tournament
        tournaments_per_day : tuple
            (min, max) tournaments per day
        size_distribution : dict, optional
            Distribution of tournament sizes. Default mimics real-world distribution.

        Returns
        -------
        List[TournamentConfig]
            Generated tournament configurations
        """
        if size_distribution is None:
            size_distribution = {
                "small": 0.45,  # 20-48 teams
                "medium": 0.35,  # 64-128 teams
                "large": 0.15,  # 192-384 teams
                "mega": 0.05,  # 512+ teams
            }

        configs = []

        for day in range(days):
            is_weekend = (day % 7) in [5, 6]

            # More tournaments on weekends
            if is_weekend:
                n_tournaments = self.rng.integers(
                    max(tournaments_per_day[0], 2), tournaments_per_day[1] + 1
                )
            else:
                n_tournaments = self.rng.integers(
                    tournaments_per_day[0], min(tournaments_per_day[1], 3) + 1
                )

            # Special event days
            if self.rng.random() < 0.14:  # ~once per week
                n_tournaments += self.rng.integers(1, 3)

            for t_idx in range(n_tournaments):
                config = self._generate_tournament_config(
                    day=day,
                    time_slot=t_idx,
                    min_team_size=min_team_size,
                    size_distribution=size_distribution,
                )
                configs.append(config)

        return sorted(configs, key=lambda c: c.start_offset_days)

    def create_weekly_series(
        self,
        weeks: int = 26,
        series_configs: Optional[list[dict[str, Any]]] = None,
    ) -> list[TournamentConfig]:
        """
        Create recurring weekly tournament series.

        Parameters
        ----------
        weeks : int
            Number of weeks to generate
        series_configs : list of dict, optional
            Custom series configurations

        Returns
        -------
        List[TournamentConfig]
            Generated tournament configurations
        """
        if series_configs is None:
            series_configs = [
                {
                    "name": "Elite_Championship",
                    "type": TournamentType.INVITATIONAL,
                    "teams": 32,
                    "format": TournamentFormat.DOUBLE_ELIMINATION,
                },
                {
                    "name": "Open_Qualifier",
                    "type": TournamentType.OPEN,
                    "teams": 256,
                    "format": TournamentFormat.SWISS,
                },
                {
                    "name": "Amateur_League",
                    "type": TournamentType.SKILL_CAPPED,
                    "teams": 128,
                    "format": TournamentFormat.DOUBLE_ELIMINATION,
                    "skill_cap": 10.0,
                },
                {
                    "name": "Intermediate_Cup",
                    "type": TournamentType.SKILL_CAPPED,
                    "teams": 64,
                    "format": TournamentFormat.SINGLE_ELIMINATION,
                    "skill_cap": 14.0,
                },
            ]

        configs = []

        for week in range(weeks):
            for series in series_configs:
                day = week * 7 + self.rng.integers(0, 7)

                config = TournamentConfig(
                    name=f"{series['name']}_W{week+1}",
                    tournament_type=series["type"],
                    format=series["format"],
                    n_teams=series["teams"],
                    start_offset_days=day + 0.5,  # Mid-day
                    selection_bias=0.3
                    if series["type"] == TournamentType.INVITATIONAL
                    else 0.7,
                    skill_cap=series.get("skill_cap"),
                    swiss_rounds=self._calculate_swiss_rounds(series["teams"])
                    if series["format"] == TournamentFormat.SWISS
                    else None,
                )
                configs.append(config)

        return configs

    def create_realistic_schedule(
        self,
        days: int = 180,
        include_weekly_series: bool = True,
        min_team_size: int = 20,
    ) -> list[TournamentConfig]:
        """
        Create a realistic tournament schedule combining dense daily tournaments
        and weekly series.

        Parameters
        ----------
        days : int
            Number of days to generate
        include_weekly_series : bool
            Whether to include weekly recurring tournaments
        min_team_size : int
            Minimum number of teams per tournament

        Returns
        -------
        List[TournamentConfig]
            Complete tournament schedule
        """
        # Generate base dense schedule
        configs = self.create_dense_schedule(
            days=days,
            min_team_size=min_team_size,
            tournaments_per_day=(1, 4),
        )

        # Add weekly series
        if include_weekly_series:
            weeks = days // 7
            weekly_configs = self.create_weekly_series(weeks=weeks)

            # Filter weekly configs to fit within days limit
            weekly_configs = [
                c for c in weekly_configs if c.start_offset_days < days
            ]
            configs.extend(weekly_configs)

        return sorted(configs, key=lambda c: c.start_offset_days)

    def create_sparse_schedule(
        self,
        days: int = 180,
        tournaments_per_week: int = 3,
        min_team_size: int = 32,
    ) -> list[TournamentConfig]:
        """
        Create a sparse tournament schedule with fewer, larger tournaments.

        Parameters
        ----------
        days : int
            Number of days to generate
        tournaments_per_week : int
            Average tournaments per week
        min_team_size : int
            Minimum number of teams

        Returns
        -------
        List[TournamentConfig]
            Sparse tournament schedule
        """
        configs = []
        weeks = days // 7

        for week in range(weeks):
            n_tournaments = self.rng.poisson(tournaments_per_week)

            for _ in range(n_tournaments):
                day = week * 7 + self.rng.integers(0, 7)

                # Sparse schedules tend to have larger tournaments
                size_roll = self.rng.random()
                if size_roll < 0.3:
                    n_teams = self.rng.choice([64, 96, 128])
                elif size_roll < 0.7:
                    n_teams = self.rng.choice([192, 256])
                else:
                    n_teams = self.rng.choice([384, 512])

                config = self._generate_tournament_config(
                    day=day,
                    time_slot=0,
                    n_teams_override=n_teams,
                    min_team_size=min_team_size,
                )
                configs.append(config)

        return sorted(configs, key=lambda c: c.start_offset_days)

    def _generate_tournament_config(
        self,
        day: int,
        time_slot: int,
        min_team_size: int = 8,
        size_distribution: Optional[dict[str, float]] = None,
        n_teams_override: Optional[int] = None,
    ) -> TournamentConfig:
        """Generate a single tournament configuration."""
        # Determine size
        if n_teams_override:
            n_teams = n_teams_override
        else:
            size_roll = self.rng.random()
            cumulative = 0

            for size_cat, prob in size_distribution.items():
                cumulative += prob
                if size_roll < cumulative:
                    n_teams = self._get_team_count_for_category(
                        size_cat, min_team_size
                    )
                    break

        # Determine format based on size
        if n_teams <= 48:
            format_weights = {
                TournamentFormat.SINGLE_ELIMINATION: 0.5,
                TournamentFormat.DOUBLE_ELIMINATION: 0.4,
                TournamentFormat.SWISS: 0.1,
            }
        elif n_teams <= 128:
            format_weights = {
                TournamentFormat.SINGLE_ELIMINATION: 0.2,
                TournamentFormat.DOUBLE_ELIMINATION: 0.5,
                TournamentFormat.SWISS: 0.3,
            }
        else:
            format_weights = {
                TournamentFormat.SINGLE_ELIMINATION: 0.1,
                TournamentFormat.DOUBLE_ELIMINATION: 0.3,
                TournamentFormat.SWISS: 0.6,
            }

        format = self._weighted_choice(format_weights)

        # Determine type
        type_roll = self.rng.random()
        if type_roll < 0.50:
            tournament_type = TournamentType.SKILL_CAPPED
            # Time-based skill caps
            if time_slot == 0:  # Morning
                skill_cap = self.rng.uniform(6.0, 10.0)
            elif time_slot >= 3:  # Evening
                skill_cap = self.rng.uniform(12.0, 18.0)
            else:  # Afternoon
                skill_cap = self.rng.uniform(8.0, 14.0)
        elif type_roll < 0.85:
            tournament_type = TournamentType.OPEN
            skill_cap = None
        else:
            tournament_type = TournamentType.INVITATIONAL
            skill_cap = None
            n_teams = min(n_teams, 64)  # Invitationals are smaller

        return TournamentConfig(
            name=f"Tournament_D{day+1}_T{time_slot+1}_{n_teams}teams",
            tournament_type=tournament_type,
            format=format,
            n_teams=n_teams,
            start_offset_days=day + (time_slot * 0.1),
            selection_bias=0.3
            if tournament_type == TournamentType.INVITATIONAL
            else 0.7,
            skill_cap=skill_cap,
            swiss_rounds=self._calculate_swiss_rounds(n_teams)
            if format == TournamentFormat.SWISS
            else None,
        )

    def _get_team_count_for_category(self, category: str, min_size: int) -> int:
        """Get random team count for a size category."""
        if min_size >= 20:
            size_ranges = {
                "small": [24, 32, 48],
                "medium": [64, 96, 128],
                "large": [192, 256, 384],
                "mega": [512, 768, 1024],
            }
        else:
            size_ranges = {
                "small": [8, 16, 32],
                "medium": [64, 128],
                "large": [256, 384],
                "mega": [512, 768, 1024],
            }

        return self.rng.choice(size_ranges.get(category, [64]))

    def _calculate_swiss_rounds(self, n_teams: int) -> int:
        """Calculate appropriate number of Swiss rounds."""
        # Standard formula: log2(n) + 1, with some adjustments
        base_rounds = int(np.log2(n_teams)) + 1

        # Adjust for very large tournaments
        if n_teams > 256:
            base_rounds = min(base_rounds, 9)  # Cap at 9 rounds

        return base_rounds

    def _weighted_choice(self, weights: dict[Any, float]) -> Any:
        """Make a weighted random choice."""
        items = list(weights.keys())
        probs = list(weights.values())
        return self.rng.choice(items, p=np.array(probs) / sum(probs))


# Convenience functions for common use cases
def create_dense_schedule(
    days: int = 180, seed: Optional[int] = None
) -> list[TournamentConfig]:
    """
    Create a dense tournament schedule with ~500 tournaments over the period.

    Parameters
    ----------
    days : int
        Number of days to generate
    seed : int, optional
        Random seed

    Returns
    -------
    List[TournamentConfig]
        Dense tournament schedule
    """
    generator = TournamentScheduleGenerator(seed=seed)
    return generator.create_realistic_schedule(days=days, min_team_size=20)


def create_sparse_schedule(
    days: int = 180, seed: Optional[int] = None
) -> list[TournamentConfig]:
    """
    Create a sparse tournament schedule with ~100 tournaments over the period.

    Parameters
    ----------
    days : int
        Number of days to generate
    seed : int, optional
        Random seed

    Returns
    -------
    List[TournamentConfig]
        Sparse tournament schedule
    """
    generator = TournamentScheduleGenerator(seed=seed)
    return generator.create_sparse_schedule(days=days, tournaments_per_week=3)


def create_competitive_season(
    months: int = 6,
    seed: Optional[int] = None,
) -> list[TournamentConfig]:
    """
    Create a competitive season with major tournaments and qualifiers.

    Parameters
    ----------
    months : int
        Length of season in months
    seed : int, optional
        Random seed

    Returns
    -------
    List[TournamentConfig]
        Competitive season schedule
    """
    generator = TournamentScheduleGenerator(seed=seed)
    days = months * 30

    # Create base schedule
    configs = generator.create_dense_schedule(
        days=days,
        min_team_size=32,
        tournaments_per_day=(2, 6),
    )

    # Add monthly majors
    for month in range(months):
        day = month * 30 + 15  # Mid-month

        # Major tournament
        major = TournamentConfig(
            name=f"Major_Championship_Month_{month+1}",
            tournament_type=TournamentType.INVITATIONAL,
            format=TournamentFormat.DOUBLE_ELIMINATION,
            n_teams=128,
            start_offset_days=day,
            selection_bias=0.2,  # Very selective
        )
        configs.append(major)

        # Open qualifier for next major
        qualifier = TournamentConfig(
            name=f"Major_Qualifier_Month_{month+1}",
            tournament_type=TournamentType.OPEN,
            format=TournamentFormat.SWISS,
            n_teams=512,
            start_offset_days=day - 7,  # Week before major
            selection_bias=0.8,
            swiss_rounds=9,
        )
        configs.append(qualifier)

    return sorted(configs, key=lambda c: c.start_offset_days)


if __name__ == "__main__":
    # Example usage
    print("=== Dense Schedule Example ===")
    dense = create_dense_schedule(days=30)  # Just 1 month for demo
    print(f"Generated {len(dense)} tournaments in 30 days")
    print(f"Average per day: {len(dense)/30:.2f}")

    print("\n=== Sparse Schedule Example ===")
    sparse = create_sparse_schedule(days=30)
    print(f"Generated {len(sparse)} tournaments in 30 days")
    print(f"Average per day: {len(sparse)/30:.2f}")

    print("\n=== Competitive Season Example ===")
    season = create_competitive_season(months=3)
    print(f"Generated {len(season)} tournaments in 3 months")

    # Show some majors
    majors = [c for c in season if "Major" in c.name]
    print(f"\nMajor tournaments: {len(majors)}")
    for major in majors[:4]:
        print(
            f"  Day {major.start_offset_days}: {major.name} ({major.n_teams} teams)"
        )
