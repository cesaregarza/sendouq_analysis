"""
Utility to add scores to matches in tournament results.
"""
import numpy as np

from synthetic_data.tournament_circuit import CircuitResults
from synthetic_data.tournament_generator import Tournament


def add_scores_to_circuit_results(
    circuit_results: CircuitResults, rng=None
) -> CircuitResults:
    """Add realistic scores to all matches in circuit results."""
    if rng is None:
        rng = np.random.default_rng()

    for tournament in circuit_results.tournaments:
        add_scores_to_tournament(tournament, rng)

    return circuit_results


def add_scores_to_tournament(tournament: Tournament, rng=None):
    """Add realistic scores to all matches in a tournament."""
    if rng is None:
        rng = np.random.default_rng()

    for stage in tournament.stages:
        # Process regular rounds
        for round_matches in stage.rounds.values():
            for match in round_matches:
                add_score_to_match(match, rng)

        # Process winners bracket
        for round_matches in stage.winners_bracket.values():
            for match in round_matches:
                add_score_to_match(match, rng)

        # Process losers bracket
        for round_matches in stage.losers_bracket.values():
            for match in round_matches:
                add_score_to_match(match, rng)

        # Process grand finals
        for match in stage.grand_finals:
            add_score_to_match(match, rng)


def add_score_to_match(match, rng):
    """Add a realistic score to a single match."""
    if match.winner is None:
        return

    # Determine if it's a close match based on skill difference
    skill_diff = abs(match.team_a.avg_skill - match.team_b.avg_skill)
    is_close = skill_diff < 2.0

    if is_close:
        # Close match - could be 3-2 or 3-1
        if rng.random() < 0.6:
            winner_score = 3
            loser_score = 2
        else:
            winner_score = 3
            loser_score = 1
    else:
        # Not close - likely 3-0 or 3-1
        if rng.random() < 0.7:
            winner_score = 3
            loser_score = 0
        else:
            winner_score = 3
            loser_score = 1

    # Assign scores based on winner
    if match.winner == match.team_a:
        match.score_a = winner_score
        match.score_b = loser_score
    else:
        match.score_a = loser_score
        match.score_b = winner_score
