"""
Othello (Reversi) game environment.

Othello is a perfect example of a fully deterministic game:
- No chance nodes
- All transitions are deterministic given (state, action)
- The "afterstate" equals the "next state"

This makes it ideal for testing whether the model learns:
- Deterministic causal rules (flipping mechanics)
- Zero chance entropy (no stochastic transitions)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


# Board encoding
EMPTY = 0
BLACK = 1
WHITE = 2

# Directions for flipping: (row_delta, col_delta)
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


@dataclass
class OthelloState:
    """Immutable Othello state."""
    board: np.ndarray  # (8, 8) int array: 0=empty, 1=black, 2=white
    current_player: int  # 1=black, 2=white
    
    def copy(self) -> "OthelloState":
        return OthelloState(
            board=self.board.copy(),
            current_player=self.current_player
        )
    
    def to_tensor(self) -> np.ndarray:
        """Convert to 3-channel representation: (3, 8, 8) float."""
        out = np.zeros((3, 8, 8), dtype=np.float32)
        out[0] = (self.board == BLACK).astype(np.float32)  # Black pieces
        out[1] = (self.board == WHITE).astype(np.float32)  # White pieces
        out[2] = float(self.current_player == BLACK)       # Current player indicator
        return out
    
    def to_flat_board(self) -> np.ndarray:
        """Flatten board to (64,) for classification targets."""
        return self.board.flatten()


def initial_state() -> OthelloState:
    """Standard Othello starting position."""
    board = np.zeros((8, 8), dtype=np.int32)
    board[3, 3] = WHITE
    board[3, 4] = BLACK
    board[4, 3] = BLACK
    board[4, 4] = WHITE
    return OthelloState(board=board, current_player=BLACK)


def _opponent(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def _get_flips(board: np.ndarray, player: int, r: int, c: int) -> List[Tuple[int, int]]:
    """Get all cells that would be flipped if player places at (r, c)."""
    if board[r, c] != EMPTY:
        return []
    
    opponent = _opponent(player)
    all_flips = []
    
    for dr, dc in DIRECTIONS:
        flips = []
        nr, nc = r + dr, c + dc
        
        # Walk in this direction, collecting opponent pieces
        while _in_bounds(nr, nc) and board[nr, nc] == opponent:
            flips.append((nr, nc))
            nr += dr
            nc += dc
        
        # Valid if we hit our own piece (not empty, not out of bounds)
        if _in_bounds(nr, nc) and board[nr, nc] == player and len(flips) > 0:
            all_flips.extend(flips)
    
    return all_flips


def is_valid_move(state: OthelloState, action: int) -> bool:
    """Check if action (0-63) is valid for current player."""
    if action < 0 or action >= 64:
        return False
    r, c = action // 8, action % 8
    flips = _get_flips(state.board, state.current_player, r, c)
    return len(flips) > 0


def get_valid_moves(state: OthelloState) -> List[int]:
    """Return list of valid action indices (0-63)."""
    valid = []
    for action in range(64):
        if is_valid_move(state, action):
            valid.append(action)
    return valid


def apply_move(state: OthelloState, action: int) -> OthelloState:
    """Apply action and return new state. Assumes action is valid."""
    r, c = action // 8, action % 8
    flips = _get_flips(state.board, state.current_player, r, c)
    
    new_board = state.board.copy()
    new_board[r, c] = state.current_player
    for fr, fc in flips:
        new_board[fr, fc] = state.current_player
    
    # Switch player
    next_player = _opponent(state.current_player)
    new_state = OthelloState(board=new_board, current_player=next_player)
    
    # If opponent has no valid moves, switch back (or game might end)
    if len(get_valid_moves(new_state)) == 0:
        new_state = OthelloState(board=new_board, current_player=state.current_player)
    
    return new_state


def is_terminal(state: OthelloState) -> bool:
    """Game ends when neither player can move."""
    if len(get_valid_moves(state)) > 0:
        return False
    # Check if opponent can move
    opp_state = OthelloState(board=state.board, current_player=_opponent(state.current_player))
    return len(get_valid_moves(opp_state)) == 0


def get_winner(state: OthelloState) -> Optional[int]:
    """Return winner (BLACK, WHITE) or None if tie. Only valid if terminal."""
    black_count = np.sum(state.board == BLACK)
    white_count = np.sum(state.board == WHITE)
    if black_count > white_count:
        return BLACK
    elif white_count > black_count:
        return WHITE
    return None


def get_score(state: OthelloState) -> Tuple[int, int]:
    """Return (black_count, white_count)."""
    return int(np.sum(state.board == BLACK)), int(np.sum(state.board == WHITE))


def render_board(state: OthelloState) -> str:
    """ASCII rendering of the board."""
    symbols = {EMPTY: '.', BLACK: 'X', WHITE: 'O'}
    lines = []
    lines.append("  0 1 2 3 4 5 6 7")
    for r in range(8):
        row = [symbols[state.board[r, c]] for c in range(8)]
        lines.append(f"{r} " + " ".join(row))
    lines.append(f"Player: {'Black' if state.current_player == BLACK else 'White'}")
    return "\n".join(lines)


class OthelloEnv:
    """
    Gym-like environment wrapper for Othello.
    
    Key property: This game is FULLY DETERMINISTIC.
    - afterstate == next_state (no chance nodes)
    - chance entropy should be 0
    """
    
    def __init__(self, render_style: str = "simple"):
        self.render_style = render_style
        self.state: OthelloState = initial_state()
        self.num_actions = 64  # 8x8 grid positions
        self.board_shape = (8, 8)
        self.num_classes = 3  # empty, black, white
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset to initial state."""
        if seed is not None:
            np.random.seed(seed)
        self.state = initial_state()
        return self.state.to_tensor(), {"valid_actions": get_valid_moves(self.state)}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute action.
        
        Returns:
            obs: (3, 8, 8) tensor
            reward: +1 for win, -1 for loss, 0 otherwise
            terminated: True if game over
            truncated: False (no truncation in board games)
            info: dict with valid_actions, etc.
        """
        valid_actions = get_valid_moves(self.state)
        
        if action not in valid_actions:
            # Invalid move: either pass or penalize
            if len(valid_actions) == 0:
                # No valid moves - check if game ends
                self.state = OthelloState(
                    board=self.state.board,
                    current_player=_opponent(self.state.current_player)
                )
            else:
                # Invalid move penalty (could also just pick random valid)
                return self.state.to_tensor(), -0.1, False, False, {
                    "valid_actions": valid_actions,
                    "invalid_move": True
                }
        else:
            self.state = apply_move(self.state, action)
        
        terminated = is_terminal(self.state)
        reward = 0.0
        
        if terminated:
            winner = get_winner(self.state)
            if winner == BLACK:
                reward = 1.0  # Could be from Black's perspective
            elif winner == WHITE:
                reward = -1.0
        
        return self.state.to_tensor(), reward, terminated, False, {
            "valid_actions": get_valid_moves(self.state),
            "board": self.state.board.copy(),
            "player": self.state.current_player,
        }
    
    def get_afterstate(self, action: int) -> np.ndarray:
        """
        In Othello, afterstate == next_state (no stochasticity).
        This is the key insight: chance entropy should be 0.
        """
        if not is_valid_move(self.state, action):
            return self.state.to_tensor()  # No change for invalid
        next_state = apply_move(self.state, action)
        return next_state.to_tensor()
    
    def get_board_target(self) -> np.ndarray:
        """Get flat board array for auxiliary prediction targets."""
        return self.state.to_flat_board()
    
    def render(self) -> str:
        return render_board(self.state)
    
    @property
    def has_chance(self) -> bool:
        """Othello has no chance nodes."""
        return False
    
    def sample_random_action(self) -> int:
        """Sample a random valid action."""
        valid = get_valid_moves(self.state)
        if len(valid) == 0:
            return 0  # Pass
        return np.random.choice(valid)


def collect_random_game(max_moves: int = 100) -> List[dict]:
    """
    Collect a trajectory of random play.
    
    Returns list of transitions:
        {
            'obs': (3, 8, 8),
            'action': int,
            'next_obs': (3, 8, 8),
            'board_before': (64,),
            'board_after': (64,),
            'reward': float,
            'done': bool,
            'valid_actions': list,
        }
    """
    env = OthelloEnv()
    obs, info = env.reset()
    transitions = []
    
    for _ in range(max_moves):
        valid_actions = info.get("valid_actions", [])
        if len(valid_actions) == 0 or is_terminal(env.state):
            break
        
        action = np.random.choice(valid_actions)
        board_before = env.state.to_flat_board()
        
        next_obs, reward, done, _, info = env.step(action)
        board_after = env.state.to_flat_board()
        
        transitions.append({
            'obs': obs,
            'action': action,
            'next_obs': next_obs,
            'board_before': board_before,
            'board_after': board_after,
            'reward': reward,
            'done': done,
            'valid_actions': valid_actions,
        })
        
        obs = next_obs
        if done:
            break
    
    return transitions
