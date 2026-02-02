"""2048 game environment with afterstate separation.

2048 is a perfect testbed for stochastic MuZero because:
1. Slide/merge is DETERMINISTIC (the "rule")
2. Tile spawn is STOCHASTIC (90% probability of 2, 10% of 4)
3. Simple state representation (4x4 grid of powers of 2)

State representation follows MuZero conventions:
- Binary encoding: 31 bits per tile × 16 tiles = 496 features
- This can represent tiles up to 2^31 (far beyond what's achievable)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from .base import Game, ChanceOutcome


@dataclass
class State2048:
    """State of a 2048 game."""

    grid: np.ndarray  # 4x4 array of tile values (0 = empty, 2, 4, 8, ...)
    score: int = 0
    done: bool = False

    def copy(self) -> "State2048":
        return State2048(
            grid=self.grid.copy(),
            score=self.score,
            done=self.done,
        )


# Actions: 0=up, 1=right, 2=down, 3=left
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_NAMES = ["up", "right", "down", "left"]


class Game2048(Game):
    """
    2048 game with explicit afterstate separation.

    Chance outcomes:
    - 0: no spawn (invalid move)
    - 1-16: spawn tile value 2 at positions 0-15
    - 17-32: spawn tile value 4 at positions 0-15
    """

    BITS_PER_TILE = 31  # Can represent up to 2^31
    GRID_SIZE = 4

    def __init__(self):
        self._rng = np.random.default_rng()

    @property
    def action_space_size(self) -> int:
        return 4  # up, right, down, left

    @property
    def chance_space_size(self) -> int:
        # 0 = no spawn (invalid move)
        # 1-16 = spawn 2 at positions 0-15
        # 17-32 = spawn 4 at positions 0-15
        return 33

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        # 31 bits per tile × 16 tiles
        return (self.BITS_PER_TILE * self.GRID_SIZE * self.GRID_SIZE,)

    def reset(self) -> State2048:
        """Reset to initial state with two random tiles."""
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int64)

        # Spawn two initial tiles
        empty = list(zip(*np.where(grid == 0)))
        pos1 = empty[self._rng.integers(len(empty))]
        grid[pos1] = 2 if self._rng.random() < 0.9 else 4

        empty = list(zip(*np.where(grid == 0)))
        pos2 = empty[self._rng.integers(len(empty))]
        grid[pos2] = 2 if self._rng.random() < 0.9 else 4

        return State2048(grid=grid, score=0, done=False)

    def clone_state(self, state: State2048) -> State2048:
        return state.copy()

    def legal_actions(self, state: State2048) -> List[int]:
        """Return list of actions that would change the grid."""
        legal = []
        for action in range(4):
            afterstate, _, _ = self.apply_action(state, action)
            if not np.array_equal(afterstate.grid, state.grid):
                legal.append(action)
        return legal

    def _slide_and_merge_line(self, line: np.ndarray) -> Tuple[np.ndarray, int]:
        """Slide and merge a single line to the left, return new line and score gained."""
        # Remove zeros
        non_zero = line[line != 0]
        if len(non_zero) == 0:
            return np.zeros_like(line), 0

        # Merge adjacent equal tiles
        merged = []
        score = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_val = non_zero[i] * 2
                merged.append(merged_val)
                score += merged_val
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1

        # Pad with zeros
        result = np.zeros_like(line)
        result[: len(merged)] = merged
        return result, score

    def _apply_action_to_grid(
        self, grid: np.ndarray, action: int
    ) -> Tuple[np.ndarray, int]:
        """Apply action to grid, return new grid and score gained."""
        new_grid = grid.copy()
        total_score = 0

        if action == ACTION_UP:
            for col in range(self.GRID_SIZE):
                new_grid[:, col], score = self._slide_and_merge_line(grid[:, col])
                total_score += score
        elif action == ACTION_DOWN:
            for col in range(self.GRID_SIZE):
                line = grid[:, col][::-1]
                merged, score = self._slide_and_merge_line(line)
                new_grid[:, col] = merged[::-1]
                total_score += score
        elif action == ACTION_LEFT:
            for row in range(self.GRID_SIZE):
                new_grid[row, :], score = self._slide_and_merge_line(grid[row, :])
                total_score += score
        elif action == ACTION_RIGHT:
            for row in range(self.GRID_SIZE):
                line = grid[row, :][::-1]
                merged, score = self._slide_and_merge_line(line)
                new_grid[row, :] = merged[::-1]
                total_score += score

        return new_grid, total_score

    def apply_action(
        self, state: State2048, action: int
    ) -> Tuple[State2048, float, Dict[str, Any]]:
        """
        Apply action to get afterstate (DETERMINISTIC).

        The afterstate is the grid after sliding and merging, but BEFORE
        a new tile spawns. This is the deterministic "rule" part.

        Returns:
            afterstate: Grid after slide/merge, before spawn.
            reward: Score gained from merging.
            info: Contains 'empty_positions' for chance sampling.
        """
        new_grid, score_gained = self._apply_action_to_grid(state.grid, action)

        # Find empty positions for potential spawn
        empty_positions = list(zip(*np.where(new_grid == 0)))

        afterstate = State2048(
            grid=new_grid,
            score=state.score + score_gained,
            done=False,  # Will be updated after chance
        )

        info = {
            "empty_positions": empty_positions,
            "grid_changed": not np.array_equal(new_grid, state.grid),
        }

        # Reward is the score gained from merging
        return afterstate, float(score_gained), info

    def sample_chance(
        self, afterstate: State2048, info: Dict[str, Any]
    ) -> ChanceOutcome:
        """Sample a tile spawn location and value."""
        empty_positions = info.get("empty_positions", [])
        grid_changed = info.get("grid_changed", True)

        # If grid didn't change, no spawn (chance outcome 0)
        if not grid_changed or len(empty_positions) == 0:
            return 0

        # Sample position and value
        pos_idx = self._rng.integers(len(empty_positions))
        row, col = empty_positions[pos_idx]
        flat_pos = row * self.GRID_SIZE + col  # 0-15

        # 90% chance of 2, 10% chance of 4
        if self._rng.random() < 0.9:
            return flat_pos + 1  # 1-16 for value 2
        else:
            return flat_pos + 17  # 17-32 for value 4

    def get_chance_distribution(
        self, afterstate: State2048, info: Dict[str, Any]
    ) -> np.ndarray:
        """Get probability distribution over chance outcomes."""
        dist = np.zeros(self.chance_space_size, dtype=np.float32)

        empty_positions = info.get("empty_positions", [])
        grid_changed = info.get("grid_changed", True)

        if not grid_changed or len(empty_positions) == 0:
            dist[0] = 1.0  # No spawn
            return dist

        # Uniform over positions, 90/10 split for value
        prob_per_pos = 1.0 / len(empty_positions)
        for row, col in empty_positions:
            flat_pos = row * self.GRID_SIZE + col
            dist[flat_pos + 1] = prob_per_pos * 0.9  # Value 2
            dist[flat_pos + 17] = prob_per_pos * 0.1  # Value 4

        return dist

    def apply_chance(self, afterstate: State2048, chance: ChanceOutcome) -> State2048:
        """Apply chance outcome (spawn tile) to get next state."""
        if chance == 0:
            # No spawn (invalid move or no space)
            next_state = afterstate.copy()
            # Check if game is over
            if len(self.legal_actions(next_state)) == 0:
                next_state.done = True
            return next_state

        # Decode chance outcome
        if chance <= 16:
            # Spawn value 2 at position chance-1
            flat_pos = chance - 1
            value = 2
        else:
            # Spawn value 4 at position chance-17
            flat_pos = chance - 17
            value = 4

        row, col = flat_pos // self.GRID_SIZE, flat_pos % self.GRID_SIZE

        next_grid = afterstate.grid.copy()
        next_grid[row, col] = value

        next_state = State2048(
            grid=next_grid,
            score=afterstate.score,
            done=False,
        )

        # Check if game is over (no legal moves)
        if len(self.legal_actions(next_state)) == 0:
            next_state.done = True

        return next_state

    def is_terminal(self, state: State2048) -> bool:
        return state.done

    def encode_state(self, state: State2048) -> torch.Tensor:
        """Encode state as binary representation."""
        return self._encode_grid(state.grid)

    def encode_afterstate(self, afterstate: State2048) -> torch.Tensor:
        """Encode afterstate as binary representation."""
        return self._encode_grid(afterstate.grid)

    def _encode_grid(self, grid: np.ndarray) -> torch.Tensor:
        """
        Encode grid as binary representation.

        Each tile is encoded as 31 bits representing the exponent
        (since tiles are powers of 2).
        """
        features = []
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                val = grid[row, col]
                if val == 0:
                    # Empty tile: all zeros
                    bits = [0] * self.BITS_PER_TILE
                else:
                    # Encode log2(val) in binary
                    exp = int(np.log2(val))
                    bits = [(exp >> i) & 1 for i in range(self.BITS_PER_TILE)]
                features.extend(bits)

        return torch.tensor(features, dtype=torch.float32)

    def get_max_tile(self, state: State2048) -> int:
        """Get the maximum tile value in the grid."""
        return int(state.grid.max())

    def render(self, state: State2048) -> str:
        """Render the game state as a string."""
        lines = []
        lines.append(f"Score: {state.score}")
        lines.append("-" * 25)
        for row in range(self.GRID_SIZE):
            cells = []
            for col in range(self.GRID_SIZE):
                val = state.grid[row, col]
                if val == 0:
                    cells.append("    .")
                else:
                    cells.append(f"{val:5d}")
            lines.append(" ".join(cells))
        lines.append("-" * 25)
        if state.done:
            lines.append("GAME OVER")
        return "\n".join(lines)


# Convenience function for testing
def play_random_game(max_moves: int = 10000) -> Tuple[int, int]:
    """Play a random game and return (max_tile, score)."""
    game = Game2048()
    state = game.reset()

    for _ in range(max_moves):
        legal = game.legal_actions(state)
        if not legal:
            break
        action = np.random.choice(legal)
        result = game.step(state, action)
        state = result.next_state
        if result.done:
            break

    return game.get_max_tile(state), state.score


if __name__ == "__main__":
    # Quick test
    game = Game2048()
    state = game.reset()
    print("Initial state:")
    print(game.render(state))

    # Play a few moves
    for i in range(5):
        legal = game.legal_actions(state)
        if not legal:
            break
        action = legal[0]
        print(f"\nAction: {ACTION_NAMES[action]}")

        # Show afterstate (before spawn)
        afterstate, reward, info = game.apply_action(state, action)
        print(f"Afterstate (reward={reward}):")
        print(game.render(afterstate))

        # Sample and apply chance
        chance = game.sample_chance(afterstate, info)
        state = game.apply_chance(afterstate, chance)
        print(f"After chance (outcome={chance}):")
        print(game.render(state))
