"""
Game environments for testing causal world models.

Othello: Fully deterministic (no chance nodes)
2048: Deterministic afterstate + stochastic chance
"""

from .othello import OthelloEnv, OthelloState, collect_random_game as collect_othello_game
from .game2048 import Game2048Env, Game2048State, collect_random_game as collect_2048_game

__all__ = [
    'OthelloEnv', 'OthelloState', 'collect_othello_game',
    'Game2048Env', 'Game2048State', 'collect_2048_game',
]
