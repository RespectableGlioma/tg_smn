from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

from .common import Game


DIRS = [(-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)]


@dataclass
class OthelloState:
    board: np.ndarray  # int8 (8,8): -1 white, 0 empty, +1 black
    player: int        # +1 black to move, -1 white to move


def _init_board() -> np.ndarray:
    b = np.zeros((8, 8), dtype=np.int8)
    b[3, 3] = -1
    b[4, 4] = -1
    b[3, 4] = 1
    b[4, 3] = 1
    return b


def _on_board(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def _captures_in_dir(board: np.ndarray, player: int, r: int, c: int, dr: int, dc: int) -> int:
    rr, cc = r + dr, c + dc
    cnt = 0
    while _on_board(rr, cc) and board[rr, cc] == -player:
        cnt += 1
        rr += dr
        cc += dc
    if cnt > 0 and _on_board(rr, cc) and board[rr, cc] == player:
        return cnt
    return 0


def _legal_moves(board: np.ndarray, player: int) -> List[int]:
    moves: List[int] = []
    empties = np.argwhere(board == 0)
    for r, c in empties:
        ok = False
        for dr, dc in DIRS:
            if _captures_in_dir(board, player, int(r), int(c), dr, dc) > 0:
                ok = True
                break
        if ok:
            moves.append(int(r) * 8 + int(c))
    return moves


def _apply_move(board: np.ndarray, player: int, move_id: int) -> Tuple[np.ndarray, bool]:
    # move_id: 0..63
    r, c = divmod(move_id, 8)
    if board[r, c] != 0:
        return board.copy(), False
    newb = board.copy()
    total_flips = 0
    for dr, dc in DIRS:
        cnt = _captures_in_dir(board, player, r, c, dr, dc)
        if cnt > 0:
            total_flips += cnt
            rr, cc = r + dr, c + dc
            for _ in range(cnt):
                newb[rr, cc] = player
                rr += dr
                cc += dc
    if total_flips == 0:
        return board.copy(), False
    newb[r, c] = player
    return newb, True


def _winner(board: np.ndarray) -> int:
    # return +1 if black wins, -1 if white wins, 0 if draw
    black = int(np.sum(board == 1))
    white = int(np.sum(board == -1))
    if black > white:
        return 1
    if white > black:
        return -1
    return 0


class Othello(Game):
    name = "othello"

    def __init__(self, img_size: int = 64, num_styles: int = 16):
        assert img_size % 8 == 0, "img_size should be divisible by 8"
        self.img_size = img_size
        self.obs_shape = (img_size, img_size)
        self.num_styles = num_styles
        self.action_size = 65  # 0..63 place, 64 pass
        self.chance_size = 1   # deterministic

    def reset(self, rng: np.random.RandomState) -> OthelloState:
        return OthelloState(board=_init_board(), player=1)

    def legal_actions(self, state: OthelloState) -> np.ndarray:
        mask = np.zeros((self.action_size,), dtype=np.bool_)
        moves = _legal_moves(state.board, state.player)
        if len(moves) == 0:
            mask[64] = True  # must pass
        else:
            mask[moves] = True
        return mask

    def apply_action(self, state: OthelloState, action: int) -> Tuple[OthelloState, float, bool, Dict[str, Any]]:
        board = state.board
        player = state.player
        legal = self.legal_actions(state)
        valid = bool(legal[action])

        info: Dict[str, Any] = {}

        if action == 64:
            # pass
            newb = board.copy()
            next_player = -player
            next_state = OthelloState(newb, next_player)
        else:
            newb, ok = _apply_move(board, player, action)
            valid = valid and ok
            # determine next player under pass rules
            opp = -player
            opp_moves = _legal_moves(newb, opp)
            if len(opp_moves) > 0:
                next_player = opp
            else:
                cur_moves = _legal_moves(newb, player)
                if len(cur_moves) > 0:
                    next_player = player
                else:
                    next_player = opp  # terminal; arbitrary
                    info["terminal"] = True
            next_state = OthelloState(newb, next_player)

        done = self.is_terminal(next_state)
        # keep reward 0 for all steps; value target is computed from final winner separately
        r = 0.0
        if done:
            info["winner_black"] = _winner(next_state.board)
        info["chance"] = 0
        info["after_board"] = next_state.board.copy()
        info["after_player"] = next_state.player
        info["done"] = done
        return next_state, r, valid, info

    def is_terminal(self, state: OthelloState) -> bool:
        # terminal when neither player has a legal move
        if len(_legal_moves(state.board, 1)) > 0:
            return False
        if len(_legal_moves(state.board, -1)) > 0:
            return False
        return True

    def chance_mask(self, afterstate: OthelloState, action_info: Dict[str, Any]) -> np.ndarray:
        return np.ones((1,), dtype=np.bool_)

    def chance_probs(self, afterstate: OthelloState, action_info: Dict[str, Any]) -> np.ndarray:
        return np.ones((1,), dtype=np.float32)

    def sample_chance(self, afterstate: OthelloState, action_info: Dict[str, Any], rng: np.random.RandomState) -> int:
        return 0

    def apply_chance(self, afterstate: OthelloState, chance: int, action_info: Dict[str, Any]) -> OthelloState:
        return afterstate  # deterministic; already next_state

    def encode_aux(self, state: OthelloState) -> Dict[str, np.ndarray]:
        # board classes: 0 empty, 1 black, 2 white
        board_cls = np.zeros((8, 8), dtype=np.int64)
        board_cls[state.board == 1] = 1
        board_cls[state.board == -1] = 2
        player_cls = np.int64(0 if state.player == 1 else 1)
        return {"board": board_cls, "player": np.array(player_cls, dtype=np.int64)}

    def render(self, state: OthelloState, style_id: int) -> np.ndarray:
        """Render a grayscale observation (H,W) uint8.

        Includes a small top-left indicator of player-to-move:
          - black to move: dark square
          - white to move: bright square
        """
        rng = np.random.RandomState(style_id * 9973 + 17)
        img_size = self.img_size
        cell = img_size // 8

        # board background + grid
        bg = rng.randint(60, 120)
        grid = rng.randint(130, 200)
        blackv = rng.randint(10, 60)
        whitev = rng.randint(200, 245)

        img = np.full((img_size, img_size), bg, dtype=np.float32)

        # subtle texture
        tex = rng.randn(img_size, img_size).astype(np.float32) * rng.uniform(0.0, 6.0)
        img = np.clip(img + tex, 0, 255)

        # grid
        for i in range(9):
            x = i * cell
            if x < img_size:
                img[:, max(0, x-1):min(img_size, x+1)] = grid
                img[max(0, x-1):min(img_size, x+1), :] = grid

        # pieces as circles
        yy, xx = np.mgrid[0:img_size, 0:img_size]
        for r in range(8):
            for c in range(8):
                v = int(state.board[r, c])
                if v == 0:
                    continue
                cx = c * cell + cell // 2
                cy = r * cell + cell // 2
                rad = cell * 0.35
                mask = ((xx - cx)**2 + (yy - cy)**2) <= rad*rad
                img[mask] = blackv if v == 1 else whitev

        # player indicator
        ind = 0 if state.player == 1 else 255
        img[0:6, 0:6] = ind

        return np.clip(img, 0, 255).astype(np.uint8)
