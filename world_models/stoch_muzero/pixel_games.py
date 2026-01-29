"""
Pixel Rendering for Board Games

Converts discrete board states to pixel images, enabling the pixel-based
world model to learn from visual input rather than symbolic state.

This bridges the gap between:
- Your Atari RSSM work (learns from pixels)
- Board game world models (clear rule structure)

The rendering is simple but sufficient:
- 2048: colored tiles with numbers
- Othello: black/white pieces on green board

Key: The model must learn to PARSE these pixels to extract the causal state,
then learn the dynamics. This is the "perception problem" we were avoiding.
"""

import numpy as np
from typing import Tuple, Optional
import torch


def render_2048_board(
    board: np.ndarray,
    cell_size: int = 16,
    padding: int = 1,
) -> np.ndarray:
    """
    Render a 2048 board as grayscale pixels.
    
    Args:
        board: [4, 4] array with tile values (0, 2, 4, 8, ...)
        cell_size: pixels per cell
        padding: pixels between cells
        
    Returns:
        [H, W] grayscale image in [0, 1]
    """
    grid_size = 4
    img_size = grid_size * cell_size + (grid_size + 1) * padding
    img = np.ones((img_size, img_size), dtype=np.float32) * 0.8  # background
    
    # Grayscale mapping for tile values
    # Higher values → darker (more visible)
    # 0 (empty) → light gray
    # 2 → slightly darker, ... up to 2048+ → very dark
    def tile_intensity(val):
        if val == 0:
            return 0.9  # empty cell - very light
        # log2 scale: 2→1, 4→2, 8→3, ..., 2048→11
        level = int(np.log2(val))
        # Map to intensity: higher level → darker
        return max(0.1, 0.9 - level * 0.07)
    
    for i in range(grid_size):
        for j in range(grid_size):
            val = board[i, j]
            y0 = padding + i * (cell_size + padding)
            x0 = padding + j * (cell_size + padding)
            
            intensity = tile_intensity(val)
            img[y0:y0+cell_size, x0:x0+cell_size] = intensity
            
            # Add a simple pattern to distinguish values
            # (since we're grayscale, we encode value in a small pattern)
            if val > 0:
                level = min(int(np.log2(val)), 11)
                # Draw dots proportional to log2(value)
                for k in range(min(level, 4)):
                    dy = 2 + (k // 2) * 4
                    dx = 2 + (k % 2) * 4
                    if dy < cell_size - 2 and dx < cell_size - 2:
                        img[y0+dy:y0+dy+2, x0+dx:x0+dx+2] = 0.0  # black dot
    
    return img


def render_othello_board(
    board: np.ndarray,
    cell_size: int = 8,
    padding: int = 1,
) -> np.ndarray:
    """
    Render an Othello board as grayscale pixels.
    
    Args:
        board: [8, 8] array with values:
            0 = empty
            1 = black piece
            -1 = white piece
        cell_size: pixels per cell
        padding: pixels between cells
        
    Returns:
        [H, W] grayscale image in [0, 1]
    """
    grid_size = 8
    img_size = grid_size * cell_size + (grid_size + 1) * padding
    img = np.ones((img_size, img_size), dtype=np.float32) * 0.4  # green-ish board (gray)
    
    for i in range(grid_size):
        for j in range(grid_size):
            val = board[i, j]
            y0 = padding + i * (cell_size + padding)
            x0 = padding + j * (cell_size + padding)
            
            # Cell background
            img[y0:y0+cell_size, x0:x0+cell_size] = 0.5  # slightly lighter
            
            if val != 0:
                # Draw circle (approximated as filled region)
                cy, cx = cell_size // 2, cell_size // 2
                radius = cell_size // 2 - 1
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dy*dy + dx*dx <= radius*radius:
                            py, px = y0 + cy + dy, x0 + cx + dx
                            if 0 <= py < img_size and 0 <= px < img_size:
                                img[py, px] = 0.1 if val == 1 else 0.95  # black or white
    
    return img


class PixelGame2048:
    """
    2048 game with pixel rendering.
    
    Wraps the discrete game logic and adds pixel observation.
    """
    
    def __init__(self, cell_size: int = 16, img_size: int = 64):
        self.cell_size = cell_size
        self.img_size = img_size
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset game and return pixel observation."""
        self.board = np.zeros((4, 4), dtype=np.int32)
        # Spawn two initial tiles
        self._spawn_tile()
        self._spawn_tile()
        return self._render()
    
    def _spawn_tile(self):
        """Spawn a new tile (90% 2, 10% 4) in random empty cell."""
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            i, j = empty[np.random.randint(len(empty))]
            self.board[i, j] = 2 if np.random.random() < 0.9 else 4
    
    def _slide_row_left(self, row: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Slide and merge a single row to the left."""
        # Remove zeros
        tiles = row[row != 0]
        if len(tiles) == 0:
            return row.copy(), False
        
        # Merge adjacent equal tiles
        merged = []
        skip_next = False
        changed = False
        
        for i in range(len(tiles)):
            if skip_next:
                skip_next = False
                continue
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                merged.append(tiles[i] * 2)
                skip_next = True
                changed = True
            else:
                merged.append(tiles[i])
        
        # Pad with zeros
        result = np.zeros(4, dtype=np.int32)
        result[:len(merged)] = merged
        
        if not np.array_equal(result, row):
            changed = True
            
        return result, changed
    
    def _apply_action(self, action: int) -> bool:
        """
        Apply action (0=up, 1=right, 2=down, 3=left).
        Returns True if board changed.
        """
        changed = False
        
        # Rotate board so we always slide left
        if action == 0:  # up
            board = self.board.T
        elif action == 1:  # right
            board = np.flip(self.board, axis=1)
        elif action == 2:  # down
            board = np.flip(self.board.T, axis=0)
        else:  # left
            board = self.board
        
        new_board = np.zeros_like(board)
        for i in range(4):
            new_board[i], row_changed = self._slide_row_left(board[i])
            changed = changed or row_changed
        
        # Rotate back
        if action == 0:
            self.board = new_board.T
        elif action == 1:
            self.board = np.flip(new_board, axis=1)
        elif action == 2:
            self.board = np.flip(new_board, axis=0).T
        else:
            self.board = new_board
            
        return changed
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take action, spawn new tile if board changed.
        
        Returns:
            obs: pixel observation
            reward: score gained (sum of merged tiles)
            done: True if game over
        """
        board_before = self.board.copy()
        changed = self._apply_action(action)
        
        reward = 0.0
        if changed:
            # Reward = sum of new tiles from merges
            # (simplified: just check what appeared)
            reward = float(self.board.sum() - board_before.sum()) / 2
            self._spawn_tile()
        
        done = self._is_game_over()
        return self._render(), reward, done
    
    def _is_game_over(self) -> bool:
        """Check if no moves possible."""
        if (self.board == 0).any():
            return False
        # Check for adjacent equal tiles
        for i in range(4):
            for j in range(4):
                val = self.board[i, j]
                if i < 3 and self.board[i+1, j] == val:
                    return False
                if j < 3 and self.board[i, j+1] == val:
                    return False
        return True
    
    def _render(self) -> np.ndarray:
        """Render to pixels and resize to img_size."""
        img = render_2048_board(self.board, cell_size=self.cell_size)
        # Resize to target size
        img = _resize_image(img, self.img_size)
        return img
    
    def get_afterstate(self, action: int) -> np.ndarray:
        """Get afterstate (board after slide but before spawn)."""
        board_backup = self.board.copy()
        self._apply_action(action)
        afterstate = self.board.copy()
        self.board = board_backup
        return afterstate
    
    def get_valid_actions(self) -> list:
        """Return list of actions that change the board."""
        valid = []
        for a in range(4):
            board_backup = self.board.copy()
            if self._apply_action(a):
                valid.append(a)
            self.board = board_backup
        return valid if valid else [0]  # return at least one


class PixelOthello:
    """
    Othello game with pixel rendering.
    """
    
    DIRECTIONS = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    def __init__(self, cell_size: int = 8, img_size: int = 64):
        self.cell_size = cell_size
        self.img_size = img_size
        self.board = np.zeros((8, 8), dtype=np.int32)
        self.current_player = 1  # 1 = black, -1 = white
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset to starting position."""
        self.board = np.zeros((8, 8), dtype=np.int32)
        # Initial position
        self.board[3, 3] = -1
        self.board[3, 4] = 1
        self.board[4, 3] = 1
        self.board[4, 4] = -1
        self.current_player = 1
        return self._render()
    
    def _is_valid_move(self, row: int, col: int, player: int) -> bool:
        """Check if move is valid."""
        if self.board[row, col] != 0:
            return False
        
        for dr, dc in self.DIRECTIONS:
            if self._would_flip(row, col, dr, dc, player):
                return True
        return False
    
    def _would_flip(self, row: int, col: int, dr: int, dc: int, player: int) -> bool:
        """Check if placing at (row,col) would flip pieces in direction (dr,dc)."""
        r, c = row + dr, col + dc
        found_opponent = False
        
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r, c] == -player:
                found_opponent = True
            elif self.board[r, c] == player:
                return found_opponent
            else:
                return False
            r, c = r + dr, c + dc
        return False
    
    def _flip_pieces(self, row: int, col: int, player: int):
        """Flip pieces after placing at (row, col)."""
        for dr, dc in self.DIRECTIONS:
            if self._would_flip(row, col, dr, dc, player):
                r, c = row + dr, col + dc
                while self.board[r, c] == -player:
                    self.board[r, c] = player
                    r, c = r + dr, c + dc
    
    def get_valid_moves(self) -> list:
        """Return list of valid moves as (row, col) tuples."""
        moves = []
        for i in range(8):
            for j in range(8):
                if self._is_valid_move(i, j, self.current_player):
                    moves.append((i, j))
        return moves
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take action (flattened index 0-63).
        
        Returns:
            obs: pixel observation
            reward: pieces gained
            done: True if game over
        """
        row, col = action // 8, action % 8
        
        if not self._is_valid_move(row, col, self.current_player):
            # Invalid move - no change, penalize
            return self._render(), -1.0, False
        
        # Count pieces before
        my_pieces_before = (self.board == self.current_player).sum()
        
        # Place and flip
        self.board[row, col] = self.current_player
        self._flip_pieces(row, col, self.current_player)
        
        # Count pieces after
        my_pieces_after = (self.board == self.current_player).sum()
        reward = float(my_pieces_after - my_pieces_before)
        
        # Switch player
        self.current_player = -self.current_player
        
        # Check if new player can move
        if not self.get_valid_moves():
            self.current_player = -self.current_player
            if not self.get_valid_moves():
                # Game over
                return self._render(), reward, True
        
        return self._render(), reward, False
    
    def _render(self) -> np.ndarray:
        """Render to pixels."""
        img = render_othello_board(self.board, cell_size=self.cell_size)
        img = _resize_image(img, self.img_size)
        return img
    
    def get_valid_actions(self) -> list:
        """Return valid actions as flattened indices."""
        moves = self.get_valid_moves()
        if not moves:
            return [0]  # pass
        return [r * 8 + c for r, c in moves]


def _resize_image(img: np.ndarray, target_size: int) -> np.ndarray:
    """Simple nearest-neighbor resize."""
    h, w = img.shape
    if h == target_size and w == target_size:
        return img
    
    result = np.zeros((target_size, target_size), dtype=np.float32)
    scale_y = h / target_size
    scale_x = w / target_size
    
    for i in range(target_size):
        for j in range(target_size):
            src_y = int(i * scale_y)
            src_x = int(j * scale_x)
            result[i, j] = img[src_y, src_x]
    
    return result


# =============================================================================
# Data Collection
# =============================================================================

def collect_pixel_episodes(
    game_class,
    n_episodes: int = 100,
    max_steps: int = 500,
    img_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect episodes with pixel observations.
    
    Returns:
        observations: [N, T+1, H, W] pixel sequences
        actions: [N, T] action sequences
        dones: [N, T] done flags
    """
    all_obs = []
    all_actions = []
    all_dones = []
    
    for ep in range(n_episodes):
        game = game_class(img_size=img_size)
        obs = game.reset()
        
        obs_seq = [obs]
        act_seq = []
        done_seq = []
        
        for t in range(max_steps):
            valid = game.get_valid_actions()
            action = valid[np.random.randint(len(valid))]
            
            next_obs, reward, done = game.step(action)
            
            act_seq.append(action)
            done_seq.append(done)
            obs_seq.append(next_obs)
            
            if done:
                break
        
        # Pad to max_steps
        while len(act_seq) < max_steps:
            act_seq.append(0)
            done_seq.append(True)
            obs_seq.append(obs_seq[-1])
        
        all_obs.append(np.stack(obs_seq[:max_steps+1], axis=0))
        all_actions.append(np.array(act_seq[:max_steps]))
        all_dones.append(np.array(done_seq[:max_steps]))
    
    return (
        np.stack(all_obs, axis=0),
        np.stack(all_actions, axis=0),
        np.stack(all_dones, axis=0),
    )


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing pixel renderers...")
    
    # Test 2048
    game = PixelGame2048(img_size=64)
    obs = game.reset()
    print(f"2048 observation shape: {obs.shape}")
    print(f"  range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    for _ in range(5):
        valid = game.get_valid_actions()
        action = valid[np.random.randint(len(valid))]
        obs, reward, done = game.step(action)
        if done:
            break
    print(f"  After 5 steps: board sum = {game.board.sum()}")
    
    # Test Othello
    game = PixelOthello(img_size=64)
    obs = game.reset()
    print(f"\nOthello observation shape: {obs.shape}")
    print(f"  range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    for _ in range(5):
        valid = game.get_valid_actions()
        action = valid[np.random.randint(len(valid))]
        obs, reward, done = game.step(action)
        if done:
            break
    print(f"  After 5 steps: pieces = {(game.board != 0).sum()}")
    
    # Test data collection
    print("\nCollecting 10 episodes of 2048...")
    obs, act, done = collect_pixel_episodes(PixelGame2048, n_episodes=10, max_steps=100)
    print(f"  obs shape: {obs.shape}")
    print(f"  act shape: {act.shape}")
    
    print("\nDone!")
