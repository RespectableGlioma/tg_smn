"""
Simple 2048 Game Implementation

This provides the core game logic for 2048 that can be used
for training world models.
"""

import numpy as np
from typing import Tuple, Optional


class Game2048:
    """
    2048 game implementation.
    
    Actions:
        0: Up
        1: Right
        2: Down
        3: Left
    """
    
    def __init__(self, size: int = 4):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int32)
        self.score = 0
        self.done = False
        self._add_random_tile()
        self._add_random_tile()
    
    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.done = False
        self._add_random_tile()
        self._add_random_tile()
        return self.board.copy()
    
    def _add_random_tile(self) -> bool:
        """Add a random tile (2 or 4) to an empty cell."""
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return False
        
        r, c = empty[np.random.randint(len(empty))]
        self.board[r, c] = 2 if np.random.random() < 0.9 else 4
        return True
    
    def _slide_row_left(self, row: np.ndarray) -> Tuple[np.ndarray, int]:
        """Slide and merge a single row to the left."""
        # Remove zeros
        non_zero = row[row != 0]
        
        # Merge adjacent equal tiles
        merged = []
        score = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                score += non_zero[i] * 2
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1
        
        # Pad with zeros
        result = np.zeros(len(row), dtype=np.int32)
        result[:len(merged)] = merged
        
        return result, score
    
    def _move(self, direction: int) -> Tuple[bool, int]:
        """
        Execute a move and return (changed, score_gained).
        
        Direction: 0=Up, 1=Right, 2=Down, 3=Left
        """
        board_before = self.board.copy()
        score_gained = 0
        
        # Rotate board so we always slide left
        rotated = np.rot90(self.board, direction)
        
        new_board = np.zeros_like(rotated)
        for i in range(self.size):
            new_board[i], row_score = self._slide_row_left(rotated[i])
            score_gained += row_score
        
        # Rotate back
        self.board = np.rot90(new_board, -direction)
        
        changed = not np.array_equal(board_before, self.board)
        return changed, score_gained
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the game.
        
        Args:
            action: 0=Up, 1=Right, 2=Down, 3=Left
            
        Returns:
            observation: current board state
            reward: score gained this step
            done: whether game is over
            info: additional info
        """
        if self.done:
            return self.board.copy(), 0.0, True, {'valid': False}
        
        changed, score_gained = self._move(action)
        self.score += score_gained
        
        if changed:
            self._add_random_tile()
        
        # Check if game is over
        self.done = self._is_game_over()
        
        return self.board.copy(), float(score_gained), self.done, {'valid': changed}
    
    def _is_game_over(self) -> bool:
        """Check if no moves are possible."""
        # If there are empty cells, game continues
        if np.any(self.board == 0):
            return False
        
        # Check for possible merges
        for i in range(self.size):
            for j in range(self.size):
                val = self.board[i, j]
                # Check right neighbor
                if j + 1 < self.size and self.board[i, j + 1] == val:
                    return False
                # Check down neighbor
                if i + 1 < self.size and self.board[i + 1, j] == val:
                    return False
        
        return True
    
    def get_valid_actions(self) -> list:
        """Return list of valid actions."""
        valid = []
        for action in range(4):
            board_copy = self.board.copy()
            changed, _ = self._move(action)
            if changed:
                valid.append(action)
            self.board = board_copy
        return valid if valid else [0]  # Return at least one action
    
    def copy(self) -> 'Game2048':
        """Create a copy of the game state."""
        game = Game2048(self.size)
        game.board = self.board.copy()
        game.score = self.score
        game.done = self.done
        return game
    
    def __repr__(self):
        return f"Game2048(score={self.score}, done={self.done})\n{self.board}"


# Convenience function
def create_game() -> Game2048:
    """Create a new 2048 game."""
    return Game2048()


if __name__ == '__main__':
    # Quick test
    game = Game2048()
    print("Initial board:")
    print(game.board)
    
    for _ in range(10):
        action = np.random.randint(4)
        obs, reward, done, info = game.step(action)
        print(f"\nAction: {action}, Reward: {reward}, Done: {done}")
        print(obs)
        if done:
            break
