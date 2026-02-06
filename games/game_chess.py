"""Chess: fully deterministic two-player game with rich rule structure.

Uses python-chess as the rule engine. The agent must discover legal moves
through play — they are NOT hardcoded into the policy. The model learns
which actions are legal by experiencing rejection of illegal moves.

AlphaZero-style action encoding:
- 4672 actions = 64 from-squares × 73 move types
- 56 queen-like moves (8 directions × 7 distances)
- 8 knight moves
- 9 underpromotions (3 directions × 3 piece types)
- Queen promotions encoded as queen-like moves

State encoding:
- 22 planes × 64 squares = 1408 features (flattened for MLP)
- Board always from current player's perspective (flipped for black)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

try:
    import chess
except ImportError:
    raise ImportError("python-chess is required: pip install python-chess")

from .base import Game, ChanceOutcome

# Move encoding constants

# 8 directions for queen-like moves: (file_delta, rank_delta) per unit step
QUEEN_DIRECTIONS = [
    (0, 1),    # 0: N
    (1, 1),    # 1: NE
    (1, 0),    # 2: E
    (1, -1),   # 3: SE
    (0, -1),   # 4: S
    (-1, -1),  # 5: SW
    (-1, 0),   # 6: W
    (-1, 1),   # 7: NW
]

# 8 knight move offsets: (file_delta, rank_delta)
KNIGHT_MOVES = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# Underpromotion: 3 forward directions × 3 piece types
# Directions from current player's perspective (always moving "north")
UNDERPROMO_DIRECTIONS = [(-1, 1), (0, 1), (1, 1)]  # Left capture, straight, right capture
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


class GameChess(Game):
    """
    Chess with AlphaZero-style encoding for Stochastic MuZero.

    Fully deterministic (chance_space_size=1), so every transition has
    entropy ≈ 0. All trajectory segments are macro candidates, enabling
    discovery of opening sequences, tactical motifs, etc.
    """

    def __init__(self):
        pass

    @property
    def action_space_size(self) -> int:
        return 4672  # 64 from-squares × 73 move types

    @property
    def chance_space_size(self) -> int:
        return 1  # Fully deterministic

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (1408,)  # 22 planes × 64 squares

    @property
    def is_two_player(self) -> bool:
        return True

    def current_player(self, state: chess.Board) -> int:
        return 0 if state.turn == chess.WHITE else 1

    def reset(self) -> chess.Board:
        return chess.Board()

    def clone_state(self, state: chess.Board) -> chess.Board:
        return state.copy()

    def legal_actions(self, state: chess.Board) -> List[int]:
        """Get all legal actions as AlphaZero-style action indices."""
        if state.is_game_over():
            return []
        actions = set()
        is_black = state.turn == chess.BLACK
        for move in state.legal_moves:
            action = self._move_to_action(move, is_black)
            actions.add(action)
        return sorted(actions)

    def apply_action(
        self, state: chess.Board, action: int
    ) -> Tuple[chess.Board, float, Dict[str, Any]]:
        """Apply action. Returns (afterstate, reward, info)."""
        board = state.copy()
        is_black = board.turn == chess.BLACK
        move = self._action_to_move(action, is_black, board)

        if move is None or move not in board.legal_moves:
            # Invalid action — return unchanged state with zero reward.
            # The MCTS should only select legal actions, so this is a fallback.
            return board, 0.0, {"invalid": True}

        board.push(move)

        # Reward: +1 for checkmate (from perspective of player who just moved)
        reward = 0.0
        if board.is_checkmate():
            reward = 1.0

        return board, reward, {}

    def sample_chance(
        self, afterstate: chess.Board, info: Dict[str, Any]
    ) -> ChanceOutcome:
        return 0  # Deterministic

    def get_chance_distribution(
        self, afterstate: chess.Board, info: Dict[str, Any]
    ) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)

    def apply_chance(
        self, afterstate: chess.Board, chance: ChanceOutcome
    ) -> chess.Board:
        return afterstate  # Identity for deterministic games

    def is_terminal(self, state: chess.Board) -> bool:
        return state.is_game_over()

    def encode_state(self, state: chess.Board) -> torch.Tensor:
        """
        Encode board as 22 planes × 64 squares = 1408 features.

        Always from current player's perspective (board flipped for black).

        Planes:
         0-5:  Current player pieces (P, N, B, R, Q, K)
         6-11: Opponent pieces (P, N, B, R, Q, K)
         12:   My kingside castling
         13:   My queenside castling
         14:   Opponent kingside castling
         15:   Opponent queenside castling
         16:   En passant square
         17:   Halfmove clock (normalized)
         18:   Fullmove number (normalized)
         19:   Color to move (always 1 — we see from own perspective)
         20:   Twofold repetition
         21:   Threefold repetition
        """
        planes = np.zeros((22, 8, 8), dtype=np.float32)
        is_black = state.turn == chess.BLACK

        # Piece planes (0-5: current player, 6-11: opponent)
        me = state.turn
        opp = not state.turn
        for pt in range(1, 7):  # PAWN=1 .. KING=6
            for sq in state.pieces(pt, me):
                r, f = chess.square_rank(sq), chess.square_file(sq)
                if is_black:
                    r = 7 - r
                planes[pt - 1, r, f] = 1.0

            for sq in state.pieces(pt, opp):
                r, f = chess.square_rank(sq), chess.square_file(sq)
                if is_black:
                    r = 7 - r
                planes[pt + 5, r, f] = 1.0

        # Castling rights (planes 12-15): my KS, my QS, opp KS, opp QS
        if is_black:
            planes[12] = float(state.has_kingside_castling_rights(chess.BLACK))
            planes[13] = float(state.has_queenside_castling_rights(chess.BLACK))
            planes[14] = float(state.has_kingside_castling_rights(chess.WHITE))
            planes[15] = float(state.has_queenside_castling_rights(chess.WHITE))
        else:
            planes[12] = float(state.has_kingside_castling_rights(chess.WHITE))
            planes[13] = float(state.has_queenside_castling_rights(chess.WHITE))
            planes[14] = float(state.has_kingside_castling_rights(chess.BLACK))
            planes[15] = float(state.has_queenside_castling_rights(chess.BLACK))

        # En passant (plane 16)
        if state.ep_square is not None:
            r = chess.square_rank(state.ep_square)
            f = chess.square_file(state.ep_square)
            if is_black:
                r = 7 - r
            planes[16, r, f] = 1.0

        # Halfmove clock (plane 17, normalized to [0, 1])
        planes[17] = state.halfmove_clock / 100.0

        # Fullmove number (plane 18, normalized)
        planes[18] = min(state.fullmove_number / 200.0, 1.0)

        # Color to move (plane 19): always 1 from own perspective
        planes[19] = 1.0

        # Repetition planes (20-21)
        if state.is_repetition(2):
            planes[20] = 1.0
        if state.is_repetition(3):
            planes[21] = 1.0

        return torch.tensor(planes.reshape(-1), dtype=torch.float32)

    def encode_afterstate(self, afterstate: chess.Board) -> torch.Tensor:
        return self.encode_state(afterstate)

    # ------------------------------------------------------------------
    # Action encoding / decoding
    # ------------------------------------------------------------------

    def _move_to_action(self, move: chess.Move, is_black: bool) -> int:
        """Convert a chess.Move to an action index in [0, 4671]."""
        from_sq = move.from_square
        to_sq = move.to_square

        # Mirror squares for black so encoding is always from own perspective
        if is_black:
            from_sq = chess.square_mirror(from_sq)
            to_sq = chess.square_mirror(to_sq)

        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        to_file = chess.square_file(to_sq)
        to_rank = chess.square_rank(to_sq)

        df = to_file - from_file
        dr = to_rank - from_rank

        # 1) Underpromotion (knight, bishop, rook)
        if move.promotion is not None and move.promotion != chess.QUEEN:
            try:
                dir_idx = UNDERPROMO_DIRECTIONS.index((df, dr))
            except ValueError:
                dir_idx = 1  # fallback to straight
            piece_idx = UNDERPROMO_PIECES.index(move.promotion)
            move_type = 64 + dir_idx * 3 + piece_idx

        # 2) Knight move
        elif (df, dr) in KNIGHT_MOVES:
            knight_idx = KNIGHT_MOVES.index((df, dr))
            move_type = 56 + knight_idx

        # 3) Queen-like move (straight / diagonal slides, pawn pushes, queen promos)
        else:
            direction = self._delta_to_direction(df, dr)
            distance = max(abs(df), abs(dr))
            move_type = direction * 7 + (distance - 1)

        return from_sq * 73 + move_type

    def _action_to_move(
        self, action: int, is_black: bool, board: chess.Board
    ) -> Optional[chess.Move]:
        """Convert an action index back to a chess.Move (or None if invalid)."""
        from_sq = action // 73
        move_type = action % 73

        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)

        promotion = None

        if move_type < 56:
            # Queen-like move
            direction = move_type // 7
            distance = move_type % 7 + 1
            df, dr = QUEEN_DIRECTIONS[direction]
            to_file = from_file + df * distance
            to_rank = from_rank + dr * distance

        elif move_type < 64:
            # Knight move
            knight_idx = move_type - 56
            df, dr = KNIGHT_MOVES[knight_idx]
            to_file = from_file + df
            to_rank = from_rank + dr

        else:
            # Underpromotion
            under_idx = move_type - 64
            dir_idx = under_idx // 3
            piece_idx = under_idx % 3
            df, dr = UNDERPROMO_DIRECTIONS[dir_idx]
            to_file = from_file + df
            to_rank = from_rank + dr
            promotion = UNDERPROMO_PIECES[piece_idx]

        # Bounds check
        if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
            return None

        to_sq = chess.square(to_file, to_rank)

        # Detect queen promotion: pawn reaching last rank via queen-like move
        if promotion is None and to_rank == 7:
            actual_from = chess.square_mirror(from_sq) if is_black else from_sq
            piece = board.piece_at(actual_from)
            if piece is not None and piece.piece_type == chess.PAWN:
                promotion = chess.QUEEN

        # Unmirror for black
        if is_black:
            from_sq = chess.square_mirror(from_sq)
            to_sq = chess.square_mirror(to_sq)

        return chess.Move(from_sq, to_sq, promotion=promotion)

    @staticmethod
    def _delta_to_direction(df: int, dr: int) -> int:
        """Map (file_delta, rank_delta) to one of 8 compass directions."""
        if df == 0 and dr > 0:
            return 0   # N
        if df > 0 and dr > 0:
            return 1   # NE
        if df > 0 and dr == 0:
            return 2   # E
        if df > 0 and dr < 0:
            return 3   # SE
        if df == 0 and dr < 0:
            return 4   # S
        if df < 0 and dr < 0:
            return 5   # SW
        if df < 0 and dr == 0:
            return 6   # W
        if df < 0 and dr > 0:
            return 7   # NW
        return 0  # fallback (shouldn't happen for valid moves)

    def render(self, state: chess.Board) -> str:
        """Human-readable board display."""
        turn_str = "White" if state.turn == chess.WHITE else "Black"
        status = ""
        if state.is_checkmate():
            winner = "Black" if state.turn == chess.WHITE else "White"
            status = f"\nCheckmate! {winner} wins."
        elif state.is_stalemate():
            status = "\nStalemate — draw."
        elif state.is_check():
            status = "\nCheck!"
        return f"{state}\nTurn: {turn_str} | Move: {state.fullmove_number}{status}"

    def action_to_algebraic(self, action: int, from_white_perspective: bool = True) -> str:
        """
        Convert action index to human-readable algebraic notation.

        Args:
            action: Action index (0-4671)
            from_white_perspective: If True, decode as if white is moving

        Returns:
            String like "e2e4", "Ng1f3", "e7e8=Q", or "???" if invalid
        """
        from_sq = action // 73
        move_type = action % 73

        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)

        promotion = None
        promotion_str = ""

        if move_type < 56:
            # Queen-like move
            direction = move_type // 7
            distance = move_type % 7 + 1
            df, dr = QUEEN_DIRECTIONS[direction]
            to_file = from_file + df * distance
            to_rank = from_rank + dr * distance

        elif move_type < 64:
            # Knight move
            knight_idx = move_type - 56
            df, dr = KNIGHT_MOVES[knight_idx]
            to_file = from_file + df
            to_rank = from_rank + dr

        else:
            # Underpromotion
            under_idx = move_type - 64
            dir_idx = under_idx // 3
            piece_idx = under_idx % 3
            df, dr = UNDERPROMO_DIRECTIONS[dir_idx]
            to_file = from_file + df
            to_rank = from_rank + dr
            promotion = UNDERPROMO_PIECES[piece_idx]
            promotion_str = "=" + chess.piece_symbol(promotion).upper()

        # Bounds check
        if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
            return "???"

        # Queen promotion detection
        if promotion is None and to_rank == 7:
            promotion_str = "=Q"

        # Build square names
        from_name = chess.FILE_NAMES[from_file] + chess.RANK_NAMES[from_rank]
        to_name = chess.FILE_NAMES[to_file] + chess.RANK_NAMES[to_rank]

        return f"{from_name}{to_name}{promotion_str}"


# Module-level helper for macro decoding
def chess_action_decoder(action: int) -> str:
    """Decode a chess action index to algebraic notation (module-level helper)."""
    game = GameChess()
    return game.action_to_algebraic(action)
