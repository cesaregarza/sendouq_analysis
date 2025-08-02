"""
Match simulation module using Bradley-Terry probability model.

This module simulates match outcomes based on player skills using the
Bradley-Terry model from the rankings analysis.
"""

from typing import List, Optional, Tuple

import numpy as np

from rankings.analysis.transforms import bt_prob
from synthetic_data.player_generator import SyntheticPlayer
from synthetic_data.tournament_generator import Match, Team


class MatchSimulator:
    """Simulates match outcomes using Bradley-Terry probability model."""
    
    def __init__(
        self,
        seed: Optional[int] = None,
        alpha: float = 1.0,
        bo_games: int = 7,
        skill_weight: float = 0.8
    ):
        """
        Initialize the match simulator.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        alpha : float
            Bradley-Terry temperature parameter
        bo_games : int
            Best-of-N games per match (must be odd)
        skill_weight : float
            Weight for skill vs performance (0-1)
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha
        self.bo_games = bo_games
        self.skill_weight = skill_weight
        
        if bo_games % 2 == 0:
            raise ValueError("bo_games must be odd for best-of format")
            
    def simulate_match(self, match: Match) -> Match:
        """
        Simulate a match outcome and update the match object.
        
        Parameters
        ----------
        match : Match
            Match to simulate
            
        Returns
        -------
        Match
            Updated match with winner and scores
        """
        team_a_strength = self._calculate_team_strength(match.team_a)
        team_b_strength = self._calculate_team_strength(match.team_b)
        
        # Simulate best-of-N games
        wins_needed = (self.bo_games + 1) // 2
        wins_a = 0
        wins_b = 0
        
        while wins_a < wins_needed and wins_b < wins_needed:
            # Get performance-adjusted strengths for this game
            perf_a = self._get_team_performance(match.team_a, team_a_strength)
            perf_b = self._get_team_performance(match.team_b, team_b_strength)
            
            # Calculate win probability using Bradley-Terry
            # Convert to positive scores for BT model
            score_a = np.exp(perf_a)
            score_b = np.exp(perf_b)
            
            prob_a_wins = bt_prob(score_a, score_b, alpha=self.alpha)
            
            # Simulate game outcome
            if self.rng.random() < prob_a_wins:
                wins_a += 1
            else:
                wins_b += 1
                
        # Update match with results
        match.score_a = wins_a
        match.score_b = wins_b
        match.winner = match.team_a if wins_a > wins_b else match.team_b
        
        return match
    
    def simulate_matches(self, matches: List[Match]) -> List[Match]:
        """
        Simulate multiple matches.
        
        Parameters
        ----------
        matches : List[Match]
            List of matches to simulate
            
        Returns
        -------
        List[Match]
            Updated matches with results
        """
        return [self.simulate_match(match) for match in matches]
    
    def _calculate_team_strength(self, team: Team) -> float:
        """
        Calculate overall team strength from player skills.
        
        Parameters
        ----------
        team : Team
            Team to evaluate
            
        Returns
        -------
        float
            Team strength value
        """
        if not team.players:
            return 0.0
            
        # Use average skill as base strength
        avg_skill = sum(p.true_skill for p in team.players) / len(team.players)
        
        # Add small bonus for team synergy based on affinity match
        affinities = [p.team_affinity for p in team.players if p.team_affinity]
        if affinities:
            most_common = max(set(affinities), key=affinities.count)
            synergy_bonus = affinities.count(most_common) / len(team.players) * 0.1
            avg_skill += synergy_bonus
            
        return avg_skill
    
    def _get_team_performance(self, team: Team, base_strength: float) -> float:
        """
        Get team performance for a specific game.
        
        Combines base strength with player performance variance.
        
        Parameters
        ----------
        team : Team
            Team playing
        base_strength : float
            Base team strength
            
        Returns
        -------
        float
            Performance value for this game
        """
        # Weight between true skill and variable performance
        skill_component = base_strength * self.skill_weight
        
        # Performance component - average of player performances
        performances = [p.get_performance(self.rng) for p in team.players]
        avg_performance = sum(performances) / len(performances)
        performance_component = avg_performance * (1 - self.skill_weight)
        
        return skill_component + performance_component
    
    def calculate_match_probability(
        self,
        team_a: Team,
        team_b: Team
    ) -> float:
        """
        Calculate probability of team A winning against team B.
        
        Parameters
        ----------
        team_a : Team
            First team
        team_b : Team
            Second team
            
        Returns
        -------
        float
            Probability of team A winning
        """
        strength_a = self._calculate_team_strength(team_a)
        strength_b = self._calculate_team_strength(team_b)
        
        # Convert to positive scores for BT model
        score_a = np.exp(strength_a)
        score_b = np.exp(strength_b)
        
        return bt_prob(score_a, score_b, alpha=self.alpha)
    
    def generate_upset(
        self,
        match: Match,
        upset_probability: float = 0.2
    ) -> Match:
        """
        Generate an upset result with specified probability.
        
        Parameters
        ----------
        match : Match
            Match to potentially upset
        upset_probability : float
            Probability of forcing an upset
            
        Returns
        -------
        Match
            Match with potentially upset result
        """
        # First simulate normally
        match = self.simulate_match(match)
        
        # Check if we should force an upset
        if self.rng.random() < upset_probability:
            # Determine who is favored
            prob_a = self.calculate_match_probability(match.team_a, match.team_b)
            favored = match.team_a if prob_a > 0.5 else match.team_b
            
            # If favored team won, flip the result
            if match.winner == favored:
                match.winner = match.team_b if favored == match.team_a else match.team_a
                match.score_a, match.score_b = match.score_b, match.score_a
                
        return match
    
    def simulate_with_noise(
        self,
        match: Match,
        noise_factor: float = 0.1
    ) -> Match:
        """
        Simulate match with additional randomness.
        
        Parameters
        ----------
        match : Match
            Match to simulate
        noise_factor : float
            Additional randomness factor (0-1)
            
        Returns
        -------
        Match
            Match with noisy result
        """
        # Temporarily reduce skill weight to add more randomness
        original_weight = self.skill_weight
        self.skill_weight = max(0.1, self.skill_weight * (1 - noise_factor))
        
        match = self.simulate_match(match)
        
        # Restore original weight
        self.skill_weight = original_weight
        
        return match