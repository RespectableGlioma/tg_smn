"""
Simple Othello/Reversi Game Implementation

This provides the core game logic for Othello that can be used
for training world models.
"""

import numpy as np
from typing import List, Tuple, Optional


class OthelloGame:
    """
    Othello/Reversi game implementation.
    
    Board representation:
        0: Empty
        1: Black (player 1)
       -1: White (player 2)
    
    Actions are integers 0-63 representing board positions (row * 8 + col).
    """
    
    # Directions: (dr, dc) for 8 directions
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    def __init__(self, size: int = 8):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int32)
        self.current_player = 1  # Black starts
        self.done = False
        self._setup_initial_position()
    
    def _setup_initial_position(self):
        """Set up the standard starting position."""
        mid = self.size // 2
        self.board[mid-1, mid-1] = -1  # White
        self.board[mid-1, mid] = 1     # Black
        self.board[mid, mid-1] = 1     # Black
        self.board[mid, mid] = -1      # White
    
    def reset(self) -> np.ndarray:
        """Reset to initial state."""
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.current_player = 1
        self.done = False
        self._setup_initial_position()
        return self.board.copy()
    
    def _pos_to_rc(self, pos: int) -> Tuple[int, int]:
        """Convert position (0-63) to (row, col)."""
        return pos // self.size, pos % self.size
    
    def _rc_to_pos(self, row: int, col: int) -> int:
        """Convert (row, col) to position (0-63)."""
        return row * self.size + col
    
    def _is_valid_pos(self, row: int, col: int) -> bool:
        """Check if position is on the board."""
        return 0 <= row < self.size and 0 <= col < self.size
    
    def _get_flips(self, row: int, col: int, player: int) -> List[Tuple[int, int]]:
        """Get all pieces that would be flipped by playing at (row, col)."""
        if self.board[row, col] != 0:
            return []
        
        flips = []
        opponent = -player
        
        for dr, dc in self.DIRECTIONS:
            r, c = row + dr, col + dc
            line = []
            
            # Follow the direction while we see opponent pieces
            while self._is_valid_pos(r, c) and self.board[r, c] == opponent:
                line.append((r, c))
                r, c = r + dr, c + dc
            
            # If we ended on our own piece, all pieces in line are flipped
            if line and self._is_valid_pos(r, c) and self.board[r, c] == player:
                flips.extend(line)
        
        return flips
    
    def is_valid_move(self, pos: int, player: Optional[int] = None) -> bool:
        """Check if a move is valid."""
        if player is None:
            player = self.current_player
        
        row, col = self._pos_to_rc(pos)
        
        if not self._is_valid_pos(row, col):
            return False
        if self.board[row, col] != 0:
            return False
        
        return len(self._get_flips(row, col, player)) > 0
    
    def get_valid_moves(self, player: Optional[int] = None) -> List[int]:
        """Get list of valid moves for player."""
        if player is None:
            player = self.current_player
        
        valid = []
        for pos in range(self.size * self.size):
            if self.is_valid_move(pos, player):
                valid.append(pos)
        return valid
    
    def make_move(self, pos: int) -> bool:
        """
        Make a move at position.
        
        Returns True if move was valid and made, False otherwise.
        """
        if self.done:
            return False
        
        row, col = self._pos_to_rc(pos)
        flips = self._get_flips(row, col, self.current_player)
        
        if not flips:
            return False
        
        # Place piece
        self.board[row, col] = self.current_player
        
        # Flip captured pieces
        for r, c in flips:
            self.board[r, c] = self.current_player
        
        # Switch player
        self.current_player = -self.current_player
        
        # Check if game is over
        self._check_game_over()
        
        return True
    
    def _check_game_over(self):
        """Check if the game is over."""
        # If current player has moves, game continues
        if self.get_valid_moves():
            return
        
        # If opponent has moves, skip current player's turn
        if self.get_valid_moves(-self.current_player):
            self.current_player = -self.current_player
            return
        
        # Neither player has moves, game is over
        self.done = True
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the game.
        
        Args:
            action: Position 0-63 to play
            
        Returns:
            observation: current board state
            reward: 0 (reward is determined at game end)
            done: whether game is over
            info: additional info
        """
        valid = self.make_move(action)
        
        reward = 0.0
        if self.done:
            black = np.sum(self.board == 1)
            white = np.sum(self.board == -1)
            if black > white:
                reward = 1.0
            elif white > black:
                reward = -1.0
        
        return self.board.copy(), reward, self.done, {'valid': valid}
    
    def get_score(self) -> Tuple[int, int]:
        """Return (black_count, white_count)."""
        black = np.sum(self.board == 1)
        white = np.sum(self.board == -1)
        return int(black), int(white)
    
    def copy(self) -> 'OthelloGame':
        """Create a copy of the game state."""
        game = OthelloGame(self.size)
        game.board = self.board.copy()
        game.current_player = self.current_player
        game.done = self.done
        return game
    
    def __repr__(self):
        symbols = {0: '.', 1: 'B', -1: 'W'}
        lines = []
        for row in self.board:
            lines.append(' '.join(symbols[v] for v in row))
        score = self.get_score()
        return f"Othello(Black={score[0]}, White={score[1]}, Turn={'B' if self.current_player == 1 else 'W'})\n" + '\n'.join(lines)


# Convenience function
def create_game() -> OthelloGame:
    """Create a new Othello game."""
    return OthelloGame()


if __name__ == '__main__':
    # Quick test
    game = OthelloGame()
    print("Initial board:")
    print(game)
    
    for _ in range(10):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break
        action = valid_moves[np.random.randint(len(valid_moves))]
        obs, reward, done, info = game.step(action)
        print(f"\nAction: {action}, Done: {done}")
        print(game)
        if done:
            break
