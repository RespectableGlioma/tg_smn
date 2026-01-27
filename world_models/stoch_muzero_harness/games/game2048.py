from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np

from .common import Game


@dataclass
class Game2048State:
    grid: np.ndarray  # int32 (4,4) tile values: 0 empty, else 2^k


def _spawn_candidates(grid: np.ndarray):
    empties = np.argwhere(grid == 0)
    return [(int(r), int(c)) for r, c in empties]


def _compress_and_merge(line: np.ndarray):
    """2048 merge on a 1D line (length 4). Returns (new_line, reward, changed)."""
    assert line.shape == (4,)
    vals = [int(x) for x in line if int(x) != 0]
    reward = 0
    out = []
    i = 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i+1]:
            merged = vals[i] * 2
            out.append(merged)
            reward += merged
            i += 2
        else:
            out.append(vals[i])
            i += 1
    out = out + [0] * (4 - len(out))
    new_line = np.array(out, dtype=np.int32)
    changed = not np.array_equal(new_line, line.astype(np.int32))
    return new_line, reward, changed


def _apply_move(grid: np.ndarray, action: int):
    """Apply deterministic move. action: 0 up, 1 down, 2 left, 3 right."""
    g = grid.astype(np.int32)
    reward = 0
    changed_any = False
    newg = g.copy()

    if action == 0:  # up
        for c in range(4):
            col = g[:, c]
            new_col, r, ch = _compress_and_merge(col)
            newg[:, c] = new_col
            reward += r
            changed_any = changed_any or ch
    elif action == 1:  # down
        for c in range(4):
            col = g[::-1, c]
            new_col, r, ch = _compress_and_merge(col)
            newg[::-1, c] = new_col
            reward += r
            changed_any = changed_any or ch
    elif action == 2:  # left
        for r0 in range(4):
            row = g[r0, :]
            new_row, r, ch = _compress_and_merge(row)
            newg[r0, :] = new_row
            reward += r
            changed_any = changed_any or ch
    elif action == 3:  # right
        for r0 in range(4):
            row = g[r0, ::-1]
            new_row, r, ch = _compress_and_merge(row)
            newg[r0, ::-1] = new_row
            reward += r
            changed_any = changed_any or ch
    else:
        raise ValueError(f"invalid action {action}")

    return newg, float(reward), bool(changed_any)


def _has_legal_move(grid: np.ndarray) -> bool:
    if np.any(grid == 0):
        return True
    # check merges
    for r in range(4):
        for c in range(4):
            v = int(grid[r, c])
            if r + 1 < 4 and int(grid[r+1, c]) == v:
                return True
            if c + 1 < 4 and int(grid[r, c+1]) == v:
                return True
    return False


def _chance_decode(chance_id: int):
    # 0 = no_spawn
    # 1..32: pos 0..15, valbit 0->2, 1->4
    if chance_id == 0:
        return None
    k = chance_id - 1
    pos = k // 2  # 0..15
    valbit = k % 2
    r, c = divmod(pos, 4)
    val = 2 if valbit == 0 else 4
    return r, c, val


def _chance_encode(r: int, c: int, val: int) -> int:
    pos = r * 4 + c
    valbit = 0 if val == 2 else 1
    return 1 + 2 * pos + valbit


class Game2048(Game):
    name = "2048"

    def __init__(self, img_size: int = 64, num_styles: int = 16, p2: float = 0.9):
        assert img_size % 4 == 0
        self.img_size = img_size
        self.obs_shape = (img_size, img_size)
        self.num_styles = num_styles
        self.action_size = 4
        self.chance_size = 33  # none + 16 positions * 2 values
        self.p2 = float(p2)

    def reset(self, rng: np.random.RandomState) -> Game2048State:
        grid = np.zeros((4, 4), dtype=np.int32)
        # spawn two tiles
        for _ in range(2):
            grid = self._spawn_tile(grid, rng)
        return Game2048State(grid=grid)

    def _spawn_tile(self, grid: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        empties = _spawn_candidates(grid)
        if len(empties) == 0:
            return grid
        r, c = empties[rng.randint(0, len(empties))]
        val = 2 if rng.rand() < self.p2 else 4
        newg = grid.copy()
        newg[r, c] = val
        return newg

    def legal_actions(self, state: Game2048State) -> np.ndarray:
        mask = np.zeros((self.action_size,), dtype=np.bool_)
        g = state.grid
        for a in range(4):
            ng, _r, changed = _apply_move(g, a)
            mask[a] = changed
        # if no action changes, game is terminal
        return mask

    def apply_action(self, state: Game2048State, action: int) -> Tuple[Game2048State, float, bool, Dict[str, Any]]:
        g = state.grid
        ng, r, changed = _apply_move(g, action)
        valid = bool(changed)  # treat non-changing move as invalid
        after = Game2048State(grid=ng)
        info: Dict[str, Any] = {"changed": changed, "after_grid": ng.copy()}
        return after, float(r), valid, info

    def is_terminal(self, state: Game2048State) -> bool:
        return not _has_legal_move(state.grid)

    def chance_mask(self, afterstate: Game2048State, action_info: Dict[str, Any]) -> np.ndarray:
        mask = np.zeros((self.chance_size,), dtype=np.bool_)
        changed = bool(action_info.get("changed", True))
        empties = _spawn_candidates(afterstate.grid)
        if (not changed) or len(empties) == 0:
            mask[0] = True  # no spawn
            return mask
        # spawn must happen
        for (r, c) in empties:
            mask[_chance_encode(r, c, 2)] = True
            mask[_chance_encode(r, c, 4)] = True
        return mask

    def chance_probs(self, afterstate: Game2048State, action_info: Dict[str, Any]) -> np.ndarray:
        probs = np.zeros((self.chance_size,), dtype=np.float32)
        mask = self.chance_mask(afterstate, action_info)
        if mask[0]:
            probs[0] = 1.0
            return probs
        empties = _spawn_candidates(afterstate.grid)
        n = len(empties)
        # uniform position, then 2/4
        p2 = self.p2
        p4 = 1.0 - p2
        for (r, c) in empties:
            probs[_chance_encode(r, c, 2)] = p2 / n
            probs[_chance_encode(r, c, 4)] = p4 / n
        return probs

    def sample_chance(self, afterstate: Game2048State, action_info: Dict[str, Any], rng: np.random.RandomState) -> int:
        probs = self.chance_probs(afterstate, action_info)
        return int(rng.choice(np.arange(self.chance_size), p=probs))

    def apply_chance(self, afterstate: Game2048State, chance: int, action_info: Dict[str, Any]) -> Game2048State:
        g = afterstate.grid.copy()
        dec = _chance_decode(int(chance))
        if dec is None:
            return Game2048State(grid=g)
        r, c, val = dec
        # only spawn if empty
        if g[r, c] == 0:
            g[r, c] = val
        return Game2048State(grid=g)

    def encode_aux(self, state: Game2048State) -> Dict[str, np.ndarray]:
        # tile exponent classes: 0 empty, k for 2^k
        g = state.grid
        out = np.zeros((4, 4), dtype=np.int64)
        nonzero = g > 0
        if np.any(nonzero):
            exps = np.round(np.log2(g[nonzero])).astype(np.int64)
            exps = np.clip(exps, 1, 15)
            out[nonzero] = exps
        return {"grid": out}

    def render(self, state: Game2048State, style_id: int) -> np.ndarray:
        rng = np.random.RandomState(style_id * 7919 + 23)
        img_size = self.img_size
        cell = img_size // 4

        bg = rng.randint(40, 90)
        gridv = rng.randint(130, 200)

        img = np.full((img_size, img_size), bg, dtype=np.float32)
        tex = rng.randn(img_size, img_size).astype(np.float32) * rng.uniform(0.0, 5.0)
        img = np.clip(img + tex, 0, 255)

        # grid lines
        for i in range(5):
            x = i * cell
            if x < img_size:
                img[:, max(0, x-1):min(img_size, x+1)] = gridv
                img[max(0, x-1):min(img_size, x+1), :] = gridv

        # tiles: brightness by exponent (higher -> brighter)
        g = state.grid
        for r in range(4):
            for c in range(4):
                v = int(g[r, c])
                y0, y1 = r * cell + 2, (r + 1) * cell - 2
                x0, x1 = c * cell + 2, (c + 1) * cell - 2
                if v == 0:
                    tile = rng.randint(70, 110)
                else:
                    exp = int(np.clip(np.round(np.log2(v)), 1, 15))
                    tile = int(np.clip(60 + exp * 12, 0, 255))
                img[y0:y1, x0:x1] = tile

        return np.clip(img, 0, 255).astype(np.uint8)
