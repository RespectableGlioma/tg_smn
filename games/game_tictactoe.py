"""Tic-Tac-Toe: minimal fully deterministic two-player game.

Used as a testbed for validating:
1. Two-player self-play with value negation
2. Macro discovery in deterministic games (all entropies = 0)
3. The complete training pipeline for zero-sum games
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from .base import Game, ChanceOutcome


@dataclass
class TicTacToeState:
    """State of a Tic-Tac-Toe game."""

    board: np.ndarray  # 3x3, values: 0=empty, 1=player1(X), 2=player2(O)
    current_player: int = 1  # 1 or 2
    done: bool = False
    winner: Optional[int] = None  # None, 1, 2, or 0 for draw

    def copy(self) -> "TicTacToeState":
        return TicTacToeState(
            board=self.board.copy(),
            current_player=self.current_player,
            done=self.done,
            winner=self.winner,
        )


class GameTicTacToe(Game):
    """
    Tic-Tac-Toe with afterstate separation for Stochastic MuZero.

    Fully deterministic (chance_space_size=1), so every transition
    has entropy 0 and all trajectory segments are macro candidates.
    """

    def __init__(self):
        self._rng = np.random.default_rng()

    @property
    def action_space_size(self) -> int:
        return 9  # 3x3 grid positions

    @property
    def chance_space_size(self) -> int:
        return 1  # Fully deterministic

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (27,)  # 3 planes of 3x3 flattened

    @property
    def is_two_player(self) -> bool:
        return True

    def current_player(self, state: TicTacToeState) -> int:
        return 0 if state.current_player == 1 else 1

    def reset(self) -> TicTacToeState:
        return TicTacToeState(
            board=np.zeros((3, 3), dtype=np.int32),
            current_player=1,
            done=False,
            winner=None,
        )

    def clone_state(self, state: TicTacToeState) -> TicTacToeState:
        return state.copy()

    def legal_actions(self, state: TicTacToeState) -> List[int]:
        if state.done:
            return []
        return [i for i in range(9) if state.board[i // 3, i % 3] == 0]

    def _check_winner(self, board: np.ndarray) -> Optional[int]:
        """Check if there's a winner. Returns 1, 2, or None."""
        for player in [1, 2]:
            # Rows
            for r in range(3):
                if all(board[r, c] == player for c in range(3)):
                    return player
            # Columns
            for c in range(3):
                if all(board[r, c] == player for r in range(3)):
                    return player
            # Diagonals
            if all(board[i, i] == player for i in range(3)):
                return player
            if all(board[i, 2 - i] == player for i in range(3)):
                return player
        return None

    def apply_action(
        self, state: TicTacToeState, action: int
    ) -> Tuple[TicTacToeState, float, Dict[str, Any]]:
        """Apply action (place piece). Fully deterministic."""
        row, col = action // 3, action % 3

        new_board = state.board.copy()
        new_board[row, col] = state.current_player

        # Check for winner
        winner = self._check_winner(new_board)
        is_draw = winner is None and np.all(new_board != 0)

        if winner is not None:
            done = True
            # Reward from perspective of the player who just moved
            reward = 1.0
        elif is_draw:
            done = True
            winner = 0  # Draw
            reward = 0.0
        else:
            done = False
            reward = 0.0

        # Switch player
        next_player = 2 if state.current_player == 1 else 1

        afterstate = TicTacToeState(
            board=new_board,
            current_player=next_player,
            done=done,
            winner=winner,
        )

        return afterstate, reward, {}

    def sample_chance(
        self, afterstate: TicTacToeState, info: Dict[str, Any]
    ) -> ChanceOutcome:
        return 0  # Deterministic

    def get_chance_distribution(
        self, afterstate: TicTacToeState, info: Dict[str, Any]
    ) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)

    def apply_chance(
        self, afterstate: TicTacToeState, chance: ChanceOutcome
    ) -> TicTacToeState:
        return afterstate  # Identity for deterministic games

    def is_terminal(self, state: TicTacToeState) -> bool:
        return state.done

    def encode_state(self, state: TicTacToeState) -> torch.Tensor:
        """
        Encode from current player's perspective.
        3 planes: my pieces, opponent pieces, empty squares.
        """
        me = state.current_player
        opp = 2 if me == 1 else 1

        my_pieces = (state.board == me).astype(np.float32).flatten()
        opp_pieces = (state.board == opp).astype(np.float32).flatten()
        empty = (state.board == 0).astype(np.float32).flatten()

        return torch.tensor(np.concatenate([my_pieces, opp_pieces, empty]))

    def encode_afterstate(self, afterstate: TicTacToeState) -> torch.Tensor:
        return self.encode_state(afterstate)

    def render(self, state: TicTacToeState) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        lines = []
        for r in range(3):
            line = " ".join(symbols[state.board[r, c]] for c in range(3))
            lines.append(line)
        lines.append(f"Player: {'X' if state.current_player == 1 else 'O'}")
        if state.done:
            if state.winner == 0:
                lines.append("Result: Draw")
            elif state.winner is not None:
                lines.append(f"Result: {symbols[state.winner]} wins!")
        return "\n".join(lines)
