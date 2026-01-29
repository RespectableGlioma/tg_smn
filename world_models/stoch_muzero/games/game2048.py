"""
2048 game environment.

2048 is the PERFECT example of the causal core + stochastic envelope separation:

1. DETERMINISTIC AFTERSTATE:
   - Given (state, action), the "slide + merge" result is fully deterministic
   - This is the "rule core" - pure causal mechanism
   
2. STOCHASTIC CHANCE:
   - After the deterministic afterstate, a tile spawns randomly:
   - Position: uniform over empty cells
   - Value: 90% chance of 2, 10% chance of 4
   - This is the irreducible stochasticity

The model should learn:
- afterstate: deterministic, low entropy
- chance: high entropy (log(#empties) + entropy({2,4}))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


# Tile values are powers of 2: 0=empty, 1=2, 2=4, 3=8, ..., 16=65536
# We use log2 encoding for classification
MAX_TILE_POWER = 17  # 2^17 = 131072 (theoretical max)

# Actions
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
ACTION_NAMES = ["LEFT", "RIGHT", "UP", "DOWN"]


@dataclass
class Game2048State:
    """
    2048 game state.
    
    Board encoding: 4x4 int array where value k means tile 2^k (k=0 means empty)
    This log2 encoding is better for neural nets than raw values.
    """
    board: np.ndarray  # (4, 4) int array: 0=empty, k=2^k tile
    score: int
    
    def copy(self) -> "Game2048State":
        return Game2048State(board=self.board.copy(), score=self.score)
    
    def to_tensor(self, num_classes: int = 17) -> np.ndarray:
        """
        Convert to one-hot representation: (num_classes, 4, 4) float.
        Each cell is one-hot encoded by its tile power.
        """
        out = np.zeros((num_classes, 4, 4), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                v = int(self.board[r, c])
                if v < num_classes:
                    out[v, r, c] = 1.0
        return out
    
    def to_flat_board(self) -> np.ndarray:
        """Flatten board to (16,) for classification targets."""
        return self.board.flatten().astype(np.int64)
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Return list of (row, col) for empty cells."""
        return [(r, c) for r in range(4) for c in range(4) if self.board[r, c] == 0]
    
    def get_max_tile(self) -> int:
        """Return the maximum tile value (as power of 2)."""
        max_power = int(np.max(self.board))
        return 2 ** max_power if max_power > 0 else 0


def _slide_and_merge_row(row: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Slide and merge a single row to the LEFT.
    Returns (new_row, score_gained).
    
    This is the DETERMINISTIC rule core.
    """
    # Filter out zeros
    non_zero = row[row != 0]
    if len(non_zero) == 0:
        return np.zeros(4, dtype=row.dtype), 0
    
    # Merge adjacent equal tiles
    merged = []
    score = 0
    i = 0
    while i < len(non_zero):
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            new_val = non_zero[i] + 1  # log2 encoding: merge doubles the value
            merged.append(new_val)
            score += 2 ** new_val  # Actual score is the merged tile value
            i += 2
        else:
            merged.append(non_zero[i])
            i += 1
    
    # Pad with zeros
    result = np.zeros(4, dtype=row.dtype)
    result[:len(merged)] = merged
    return result, score


def _apply_action_deterministic(board: np.ndarray, action: int) -> Tuple[np.ndarray, int, bool]:
    """
    Apply action DETERMINISTICALLY (no random spawn).
    Returns (afterstate_board, score_gained, board_changed).
    
    This is the CAUSAL RULE CORE - pure deterministic transition.
    """
    new_board = np.zeros_like(board)
    total_score = 0
    
    if action == LEFT:
        for r in range(4):
            new_board[r], s = _slide_and_merge_row(board[r])
            total_score += s
    
    elif action == RIGHT:
        for r in range(4):
            new_board[r], s = _slide_and_merge_row(board[r, ::-1])
            new_board[r] = new_board[r, ::-1]
            total_score += s
    
    elif action == UP:
        for c in range(4):
            col, s = _slide_and_merge_row(board[:, c])
            new_board[:, c] = col
            total_score += s
    
    elif action == DOWN:
        for c in range(4):
            col, s = _slide_and_merge_row(board[::-1, c])
            new_board[:, c] = col[::-1]
            total_score += s
    
    changed = not np.array_equal(board, new_board)
    return new_board, total_score, changed


def _spawn_random_tile(board: np.ndarray, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, int, int]:
    """
    Spawn a random tile on empty cell.
    Returns (new_board, spawn_position, spawn_value).
    
    This is the STOCHASTIC CHANCE NODE:
    - Position: uniform over empty cells
    - Value: 90% for 2 (encoded as 1), 10% for 4 (encoded as 2)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    empty = [(r, c) for r in range(4) for c in range(4) if board[r, c] == 0]
    if len(empty) == 0:
        return board, -1, 0
    
    # Random position (uniform)
    idx = rng.integers(len(empty))
    r, c = empty[idx]
    position = r * 4 + c  # Flatten to 0-15
    
    # Random value: 90% for 2, 10% for 4
    value = 1 if rng.random() < 0.9 else 2
    
    new_board = board.copy()
    new_board[r, c] = value
    
    return new_board, position, value


def initial_state(seed: Optional[int] = None) -> Game2048State:
    """Create initial state with two random tiles."""
    rng = np.random.default_rng(seed)
    board = np.zeros((4, 4), dtype=np.int32)
    board, _, _ = _spawn_random_tile(board, rng)
    board, _, _ = _spawn_random_tile(board, rng)
    return Game2048State(board=board, score=0)


def is_game_over(board: np.ndarray) -> bool:
    """Check if no valid moves remain."""
    # Check for empty cells
    if np.any(board == 0):
        return False
    
    # Check for adjacent equal tiles (can merge)
    for r in range(4):
        for c in range(4):
            if c + 1 < 4 and board[r, c] == board[r, c + 1]:
                return False
            if r + 1 < 4 and board[r, c] == board[r + 1, c]:
                return False
    
    return True


def render_board(state: Game2048State) -> str:
    """ASCII rendering of the board."""
    lines = [f"Score: {state.score}"]
    lines.append("+" + "------+" * 4)
    for r in range(4):
        row_vals = []
        for c in range(4):
            v = state.board[r, c]
            if v == 0:
                row_vals.append("     ")
            else:
                row_vals.append(f"{2**v:5d}")
        lines.append("|" + "|".join(row_vals) + "|")
        lines.append("+" + "------+" * 4)
    return "\n".join(lines)


class Game2048Env:
    """
    Gym-like environment wrapper for 2048.
    
    This is the ideal test case for causal rule learning:
    - Deterministic afterstate (slide+merge)
    - Stochastic chance (tile spawn)
    
    The model should learn to separate these cleanly.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.state: Game2048State = initial_state(seed)
        self.num_actions = 4  # LEFT, RIGHT, UP, DOWN
        self.board_shape = (4, 4)
        self.num_classes = 17  # 0=empty, 1-16 = tile powers
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = initial_state(seed)
        return self.state.to_tensor(), {
            "score": self.state.score,
            "max_tile": self.state.get_max_tile(),
        }
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute action. This does:
        1. Apply deterministic slide+merge (afterstate)
        2. Apply stochastic tile spawn (chance)
        
        Returns:
            obs: (num_classes, 4, 4) tensor
            reward: score gained from merges
            terminated: True if game over
            truncated: False
            info: dict with details
        """
        # 1. DETERMINISTIC: Compute afterstate
        afterstate_board, score_gained, changed = _apply_action_deterministic(
            self.state.board, action
        )
        
        if not changed:
            # Invalid move (board unchanged) - still valid to take, just no effect
            return self.state.to_tensor(), 0.0, False, False, {
                "score": self.state.score,
                "max_tile": self.state.get_max_tile(),
                "valid_move": False,
                "afterstate_board": afterstate_board.copy(),
            }
        
        # 2. STOCHASTIC: Spawn random tile
        final_board, spawn_pos, spawn_val = _spawn_random_tile(afterstate_board, self.rng)
        
        new_score = self.state.score + score_gained
        self.state = Game2048State(board=final_board, score=new_score)
        
        terminated = is_game_over(final_board)
        
        return self.state.to_tensor(), float(score_gained), terminated, False, {
            "score": new_score,
            "max_tile": self.state.get_max_tile(),
            "valid_move": True,
            "afterstate_board": afterstate_board.copy(),
            "spawn_position": spawn_pos,
            "spawn_value": spawn_val,
            "changed_cells": (self.state.board != afterstate_board).astype(np.int32),
        }
    
    def get_afterstate(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Compute DETERMINISTIC afterstate without applying chance.
        
        Returns:
            afterstate_tensor: (num_classes, 4, 4)
            afterstate_board: (4, 4) raw board
            changed: whether the action changed the board
            
        This is the key method for separating causal rules from chance.
        """
        afterstate_board, _, changed = _apply_action_deterministic(
            self.state.board, action
        )
        
        # Convert to tensor
        afterstate = Game2048State(board=afterstate_board, score=0)
        return afterstate.to_tensor(), afterstate_board, changed
    
    def get_chance_distribution(self, afterstate_board: np.ndarray) -> dict:
        """
        Get the TRUE chance distribution for a given afterstate.
        
        This is what the model should learn to predict.
        
        Returns:
            {
                'empty_mask': (16,) bool - which cells are empty
                'position_probs': (16,) float - uniform over empties
                'value_probs': (2,) float - [0.9, 0.1] for 2 vs 4
                'num_empties': int
            }
        """
        empty_mask = (afterstate_board.flatten() == 0)
        num_empties = int(np.sum(empty_mask))
        
        if num_empties == 0:
            position_probs = np.zeros(16, dtype=np.float32)
        else:
            position_probs = empty_mask.astype(np.float32) / num_empties
        
        value_probs = np.array([0.9, 0.1], dtype=np.float32)  # P(2), P(4)
        
        return {
            'empty_mask': empty_mask,
            'position_probs': position_probs,
            'value_probs': value_probs,
            'num_empties': num_empties,
        }
    
    def get_board_target(self) -> np.ndarray:
        """Get flat board array for auxiliary prediction targets."""
        return self.state.to_flat_board()
    
    def render(self) -> str:
        return render_board(self.state)
    
    @property
    def has_chance(self) -> bool:
        """2048 has chance nodes."""
        return True
    
    def get_valid_actions(self) -> List[int]:
        """
        Return actions that actually change the board.
        """
        valid = []
        for action in range(4):
            _, _, changed = _apply_action_deterministic(self.state.board, action)
            if changed:
                valid.append(action)
        return valid
    
    def sample_random_action(self, valid_only: bool = True) -> int:
        """Sample a random action (optionally only valid ones)."""
        if valid_only:
            valid = self.get_valid_actions()
            if len(valid) == 0:
                return self.rng.integers(4)  # Game over anyway
            return int(self.rng.choice(valid))
        return int(self.rng.integers(4))


def collect_random_game(max_moves: int = 1000, seed: Optional[int] = None, 
                        valid_only: bool = True) -> List[dict]:
    """
    Collect a trajectory of random play.
    
    Args:
        max_moves: Maximum moves before stopping
        seed: Random seed
        valid_only: If True, only sample valid (board-changing) actions
    
    Returns list of transitions with explicit separation of:
        - board_before: state before action
        - afterstate: deterministic result of action (RULE CORE)
        - board_after: final state after chance spawn (STOCHASTIC)
        - chance_info: spawn position and value
    """
    env = Game2048Env(seed=seed)
    obs, info = env.reset()
    transitions = []
    
    for _ in range(max_moves):
        valid_actions = env.get_valid_actions()
        if len(valid_actions) == 0:
            break
        
        # Sample action
        if valid_only:
            action = int(env.rng.choice(valid_actions))
        else:
            action = int(env.rng.integers(4))
        
        # Record state BEFORE
        board_before = env.state.to_flat_board()
        obs_before = obs.copy()
        
        # Compute DETERMINISTIC afterstate
        afterstate_tensor, afterstate_board, changed = env.get_afterstate(action)
        
        # Get TRUE chance distribution
        chance_dist = env.get_chance_distribution(afterstate_board)
        
        # Execute action (applies afterstate + chance)
        next_obs, reward, done, _, step_info = env.step(action)
        
        transitions.append({
            # Observations
            'obs': obs_before,
            'action': action,
            'next_obs': next_obs,
            
            # Board states (for auxiliary supervision)
            'board_before': board_before,
            'afterstate_board': afterstate_board.flatten().astype(np.int64),
            'board_after': env.state.to_flat_board(),
            
            # Chance information (ground truth for chance head)
            'spawn_position': step_info.get('spawn_position', -1),
            'spawn_value': step_info.get('spawn_value', 0),
            'empty_mask': chance_dist['empty_mask'],
            'num_empties': chance_dist['num_empties'],
            
            # Meta
            'reward': reward,
            'done': done,
            'score': step_info.get('score', 0),
            'max_tile': step_info.get('max_tile', 0),
            'valid_move': changed,
        })
        
        obs = next_obs
        if done:
            break
    
    return transitions


def compute_oracle_chance_entropy(afterstate_board: np.ndarray) -> float:
    """
    Compute the TRUE entropy of the chance distribution.
    
    H = log(#empties) + H({0.9, 0.1})
    
    This is what a perfect model should achieve.
    """
    num_empties = int(np.sum(afterstate_board == 0))
    if num_empties == 0:
        return 0.0
    
    # Entropy over positions (uniform)
    H_pos = np.log(num_empties)
    
    # Entropy over values
    p = np.array([0.9, 0.1])
    H_val = -np.sum(p * np.log(p + 1e-10))
    
    return float(H_pos + H_val)
