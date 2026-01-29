"""
Training script for VQ-VAE World Model.

Usage:
    python -m world_models.stoch_muzero.train_vq --game 2048 --train_steps 20000
    python -m world_models.stoch_muzero.train_vq --game othello --train_steps 20000

The key test is whether entropy distributions differ:
- 2048: Should show BIMODAL entropy (low for deterministic slides, high for random spawns)
- Othello: Should show UNIMODAL LOW entropy (all transitions deterministic)

No oracle needed - entropy emerges from learned transition statistics!
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .vq_model import VQWorldModel, VQWorldModelConfig


# =============================================================================
# Data Generation (reuse from pixel_games.py)
# =============================================================================

def render_2048_board(board: np.ndarray, size: int = 64) -> np.ndarray:
    """Render 2048 board as grayscale image."""
    img = np.zeros((size, size), dtype=np.float32)
    cell_size = size // 4
    
    for i in range(4):
        for j in range(4):
            val = board[i, j]
            if val > 0:
                # Log-scale intensity
                intensity = min(np.log2(val) / 11.0, 1.0)  # max tile ~2048
            else:
                intensity = 0.0
            
            y0, y1 = i * cell_size, (i + 1) * cell_size
            x0, x1 = j * cell_size, (j + 1) * cell_size
            img[y0+1:y1-1, x0+1:x1-1] = intensity
    
    return img


def render_othello_board(board: np.ndarray, size: int = 64) -> np.ndarray:
    """Render Othello board as grayscale image."""
    img = np.ones((size, size), dtype=np.float32) * 0.3  # Board color
    cell_size = size // 8
    
    for i in range(8):
        for j in range(8):
            cy = i * cell_size + cell_size // 2
            cx = j * cell_size + cell_size // 2
            r = cell_size // 2 - 2
            
            if board[i, j] == 1:  # Black
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if dy*dy + dx*dx <= r*r:
                            img[cy+dy, cx+dx] = 0.0
            elif board[i, j] == -1:  # White
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if dy*dy + dx*dx <= r*r:
                            img[cy+dy, cx+dx] = 1.0
    
    return img


class Game2048:
    """Simple 2048 game for data generation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros((4, 4), dtype=np.int32)
        self._add_random_tile()
        self._add_random_tile()
        return self.board.copy()
    
    def _add_random_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            i, j = empty[np.random.randint(len(empty))]
            self.board[i, j] = 2 if np.random.random() < 0.9 else 4
    
    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        """action: 0=up, 1=right, 2=down, 3=left"""
        old_board = self.board.copy()
        
        # Rotate so we always merge left
        rotated = np.rot90(self.board, action)
        merged = self._merge_left(rotated)
        self.board = np.rot90(merged, -action)
        
        moved = not np.array_equal(old_board, self.board)
        if moved:
            self._add_random_tile()
        
        done = self._is_game_over()
        return self.board.copy(), done
    
    def _merge_left(self, board):
        new_board = np.zeros_like(board)
        for i in range(4):
            row = board[i][board[i] != 0]
            new_row = []
            skip = False
            for j, val in enumerate(row):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(row) and row[j] == row[j + 1]:
                    new_row.append(val * 2)
                    skip = True
                else:
                    new_row.append(val)
            new_board[i, :len(new_row)] = new_row
        return new_board
    
    def _is_game_over(self):
        if 0 in self.board:
            return False
        for i in range(4):
            for j in range(4):
                if j < 3 and self.board[i, j] == self.board[i, j+1]:
                    return False
                if i < 3 and self.board[i, j] == self.board[i+1, j]:
                    return False
        return True


class Othello:
    """Simple Othello game for data generation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.board[3, 3] = self.board[4, 4] = -1  # White
        self.board[3, 4] = self.board[4, 3] = 1   # Black
        self.current_player = 1
        return self.board.copy()
    
    def get_valid_moves(self) -> List[int]:
        moves = []
        for i in range(8):
            for j in range(8):
                if self._is_valid_move(i, j, self.current_player):
                    moves.append(i * 8 + j)
        return moves
    
    def _is_valid_move(self, row, col, player):
        if self.board[row, col] != 0:
            return False
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dr, dc in directions:
            if self._would_flip(row, col, dr, dc, player):
                return True
        return False
    
    def _would_flip(self, row, col, dr, dc, player):
        r, c = row + dr, col + dc
        if not (0 <= r < 8 and 0 <= c < 8):
            return False
        if self.board[r, c] != -player:
            return False
        
        r, c = r + dr, c + dc
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r, c] == 0:
                return False
            if self.board[r, c] == player:
                return True
            r, c = r + dr, c + dc
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        row, col = action // 8, action % 8
        
        if not self._is_valid_move(row, col, self.current_player):
            # Invalid move, try to pass or end
            self.current_player *= -1
            if not self.get_valid_moves():
                return self.board.copy(), True
            return self.board.copy(), False
        
        self.board[row, col] = self.current_player
        self._flip_pieces(row, col, self.current_player)
        
        self.current_player *= -1
        if not self.get_valid_moves():
            self.current_player *= -1
            if not self.get_valid_moves():
                return self.board.copy(), True
        
        return self.board.copy(), False
    
    def _flip_pieces(self, row, col, player):
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dr, dc in directions:
            if self._would_flip(row, col, dr, dc, player):
                r, c = row + dr, col + dc
                while self.board[r, c] == -player:
                    self.board[r, c] = player
                    r, c = r + dr, c + dc


def generate_trajectories(
    game: str,
    n_trajectories: int,
    max_steps: int,
    img_size: int = 64,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data as (observations, actions) pairs.
    
    Returns:
        obs: [N, T+1, 1, H, W] float32 images in [0, 1]
        actions: [N, T] int64 actions
    """
    all_obs = []
    all_actions = []
    
    iterator = range(n_trajectories)
    if verbose:
        iterator = tqdm(iterator, desc=f"Generating {game} trajectories")
    
    for _ in iterator:
        if game == '2048':
            env = Game2048()
            n_actions = 4
            render_fn = render_2048_board
        elif game == 'othello':
            env = Othello()
            n_actions = 64
            render_fn = render_othello_board
        else:
            raise ValueError(f"Unknown game: {game}")
        
        board = env.reset()
        trajectory_obs = [render_fn(board, img_size)]
        trajectory_actions = []
        
        for t in range(max_steps):
            if game == '2048':
                action = np.random.randint(n_actions)
            else:  # othello
                valid = env.get_valid_moves()
                if not valid:
                    break
                action = np.random.choice(valid)
            
            trajectory_actions.append(action)
            board, done = env.step(action)
            trajectory_obs.append(render_fn(board, img_size))
            
            if done:
                break
        
        # Pad if necessary
        while len(trajectory_obs) < max_steps + 1:
            trajectory_obs.append(trajectory_obs[-1])
        while len(trajectory_actions) < max_steps:
            trajectory_actions.append(0)
        
        # Truncate to exact length
        trajectory_obs = trajectory_obs[:max_steps + 1]
        trajectory_actions = trajectory_actions[:max_steps]
        
        all_obs.append(np.stack(trajectory_obs))
        all_actions.append(np.array(trajectory_actions))
    
    obs = np.stack(all_obs)[:, :, np.newaxis, :, :]  # [N, T+1, 1, H, W]
    actions = np.stack(all_actions)  # [N, T]
    
    return obs.astype(np.float32), actions.astype(np.int64)


# =============================================================================
# Training
# =============================================================================

def train_vq_world_model(
    game: str,
    outdir: str,
    train_steps: int = 20000,
    batch_size: int = 32,
    lr: float = 3e-4,
    n_trajectories: int = 2000,
    trajectory_len: int = 50,
    unroll_steps: int = 5,
    img_size: int = 64,
    codebook_size: int = 512,
    code_dim: int = 64,
    eval_every: int = 1000,
    save_every: int = 5000,
    device: str = 'cuda',
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    """Train a VQ-VAE world model."""
    
    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    Path(outdir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Training VQ World Model for {game}")
        print(f"  Device: {device}")
        print(f"  Output: {outdir}")
    
    # Generate data
    if verbose:
        print("Generating training data...")
    obs, actions = generate_trajectories(
        game, n_trajectories, trajectory_len, img_size, verbose=verbose
    )
    
    n_val = max(1, n_trajectories // 10)
    val_obs, val_actions = obs[:n_val], actions[:n_val]
    train_obs, train_actions = obs[n_val:], actions[n_val:]
    
    if verbose:
        print(f"  Train: {train_obs.shape[0]} trajectories")
        print(f"  Val: {val_obs.shape[0]} trajectories")
    
    # Create model
    n_actions = 4 if game == '2048' else 64
    
    cfg = VQWorldModelConfig(
        img_size=img_size,
        n_actions=n_actions,
        codebook_size=codebook_size,
        code_dim=code_dim,
    )
    model = VQWorldModel(cfg).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training
    train_obs_t = torch.from_numpy(train_obs).to(device)
    train_actions_t = torch.from_numpy(train_actions).to(device)
    val_obs_t = torch.from_numpy(val_obs).to(device)
    val_actions_t = torch.from_numpy(val_actions).to(device)
    
    metrics_history = []
    best_val_loss = float('inf')
    
    model.train()
    pbar = tqdm(range(1, train_steps + 1), desc="Training", disable=not verbose)
    
    for step in pbar:
        # Sample batch
        idx = torch.randint(0, train_obs_t.shape[0], (batch_size,))
        obs_batch = train_obs_t[idx]
        action_batch = train_actions_t[idx]
        
        # Forward + loss
        losses = model.compute_loss(
            obs_batch, action_batch, unroll_steps=unroll_steps
        )
        
        # Backward
        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Log
        if step % 100 == 0:
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'trans': f"{losses['transition_loss'].item():.4f}",
                'ent': f"{losses['entropy'].item():.2f}",
                'perp': f"{losses['perplexity'].item():.0f}",
            })
        
        # Evaluate
        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_losses = model.compute_loss(
                    val_obs_t, val_actions_t, unroll_steps=unroll_steps
                )
                
                # Collect entropy statistics
                entropy_stats = analyze_entropy(
                    model, val_obs_t[:100], val_actions_t[:100], device
                )
            model.train()
            
            metrics = {
                'step': step,
                'train_loss': losses['total_loss'].item(),
                'train_recon': losses['recon_loss'].item(),
                'train_vq': losses['vq_loss'].item(),
                'train_transition': losses['transition_loss'].item(),
                'train_entropy': losses['entropy'].item(),
                'train_perplexity': losses['perplexity'].item(),
                'val_loss': val_losses['total_loss'].item(),
                'val_transition': val_losses['transition_loss'].item(),
                'val_entropy': val_losses['entropy'].item(),
                **{f'val_{k}': v for k, v in entropy_stats.items()},
            }
            metrics_history.append(metrics)
            
            if verbose:
                print(f"\nStep {step}:")
                print(f"  Val loss: {val_losses['total_loss'].item():.4f}")
                print(f"  Val transition: {val_losses['transition_loss'].item():.4f}")
                print(f"  Val entropy: {val_losses['entropy'].item():.2f} bits")
                print(f"  Entropy stats: mean={entropy_stats['mean_entropy']:.2f}, "
                      f"std={entropy_stats['std_entropy']:.2f}")
                print(f"  Frac below 0.5: {entropy_stats['frac_below_0.5']*100:.1f}%, "
                      f"above 2.0: {entropy_stats['frac_above_2.0']*100:.1f}%")
            
            # Save best
            if val_losses['total_loss'].item() < best_val_loss:
                best_val_loss = val_losses['total_loss'].item()
                torch.save(model.state_dict(), os.path.join(outdir, 'best_model.pt'))
        
        # Checkpoint
        if step % save_every == 0:
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': asdict(cfg),
            }, os.path.join(outdir, f'checkpoint_{step}.pt'))
    
    # Final save
    torch.save(model.state_dict(), os.path.join(outdir, 'final_model.pt'))
    
    # Save metrics
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_stats = analyze_entropy(model, val_obs_t, val_actions_t, device)
    
    if verbose:
        print("\n" + "="*60)
        print("FINAL ENTROPY ANALYSIS")
        print("="*60)
        print(f"Mean entropy: {final_stats['mean_entropy']:.3f} bits")
        print(f"Std entropy: {final_stats['std_entropy']:.3f}")
        print(f"Min/Max: {final_stats['min_entropy']:.3f} / {final_stats['max_entropy']:.3f}")
        print(f"Frac below 0.5 bits: {final_stats['frac_below_0.5']*100:.1f}%")
        print(f"Frac below 1.0 bits: {final_stats['frac_below_1.0']*100:.1f}%")
        print(f"Frac above 2.0 bits: {final_stats['frac_above_2.0']*100:.1f}%")
        print(f"Frac above 3.0 bits: {final_stats['frac_above_3.0']*100:.1f}%")
        print("="*60)
        
        if game == '2048':
            print("\nExpected for 2048 (stochastic):")
            print("  - BIMODAL distribution (some low, some high entropy)")
            print("  - frac_below_0.5 < 80% (not all deterministic)")
            print("  - frac_above_2.0 > 5% (some high uncertainty)")
        else:
            print("\nExpected for Othello (deterministic):")
            print("  - UNIMODAL LOW entropy")
            print("  - frac_below_0.5 > 80% (mostly deterministic)")
            print("  - frac_above_2.0 < 5% (very little uncertainty)")
    
    with open(os.path.join(outdir, 'final_stats.json'), 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    return final_stats


def analyze_entropy(
    model: VQWorldModel,
    obs_batch: torch.Tensor,
    actions_batch: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Analyze entropy distribution of learned transitions."""
    model.eval()
    all_entropies = []
    
    B, Tp1, C, H, W = obs_batch.shape
    T = Tp1 - 1
    
    with torch.no_grad():
        for b in range(min(B, 100)):  # Limit for speed
            # Encode initial state
            enc = model.encode(obs_batch[b:b+1, 0], training=False)
            z_q = enc['z_q']
            
            for t in range(min(T, 20)):  # Limit timesteps
                action = actions_batch[b:b+1, t]
                
                # Get transition entropy
                result = model.step(z_q, action, sample=False)
                all_entropies.append(result['entropy'].item())
                
                # Teacher forcing: use actual next observation
                enc = model.encode(obs_batch[b:b+1, t+1], training=False)
                z_q = enc['z_q']
    
    all_entropies = np.array(all_entropies)
    
    return {
        'mean_entropy': float(all_entropies.mean()),
        'std_entropy': float(all_entropies.std()),
        'min_entropy': float(all_entropies.min()),
        'max_entropy': float(all_entropies.max()),
        'median_entropy': float(np.median(all_entropies)),
        'frac_below_0.5': float((all_entropies < 0.5).mean()),
        'frac_below_1.0': float((all_entropies < 1.0).mean()),
        'frac_above_2.0': float((all_entropies > 2.0).mean()),
        'frac_above_3.0': float((all_entropies > 3.0).mean()),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE World Model")
    parser.add_argument('--game', type=str, default='2048', choices=['2048', 'othello'])
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--train_steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_trajectories', type=int, default=2000)
    parser.add_argument('--trajectory_len', type=int, default=50)
    parser.add_argument('--unroll_steps', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--codebook_size', type=int, default=512)
    parser.add_argument('--code_dim', type=int, default=64)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    
    if args.outdir is None:
        args.outdir = f'outputs_vq_{args.game}'
    
    train_vq_world_model(
        game=args.game,
        outdir=args.outdir,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        n_trajectories=args.n_trajectories,
        trajectory_len=args.trajectory_len,
        unroll_steps=args.unroll_steps,
        img_size=args.img_size,
        codebook_size=args.codebook_size,
        code_dim=args.code_dim,
        eval_every=args.eval_every,
        save_every=args.save_every,
        device=args.device,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
