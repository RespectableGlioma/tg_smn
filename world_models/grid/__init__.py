"""
Grid-based game implementations for world model training.

This module provides simple implementations of:
- 2048 (stochastic tile spawning)
- Othello (fully deterministic)
- GameRenderer (convert boards to images)

Usage:
    from world_models.grid import Game2048, OthelloGame, GameRenderer
    
    # 2048
    game = Game2048()
    obs = game.reset()
    obs, reward, done, info = game.step(action)  # action in {0,1,2,3}
    
    # Othello
    game = OthelloGame()
    obs = game.reset()
    valid_moves = game.get_valid_moves()
    obs, reward, done, info = game.step(action)  # action in {0..63}
    
    # Rendering
    renderer = GameRenderer(img_size=64)
    img = renderer.render_2048(game.board)  # [64, 64] grayscale
"""

from .game_2048 import Game2048
from .othello import OthelloGame
from .renderer import GameRenderer

__all__ = ['Game2048', 'OthelloGame', 'GameRenderer']
