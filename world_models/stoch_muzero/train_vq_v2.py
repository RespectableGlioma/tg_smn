"""
Train VQ-VAE World Model v2 with improved codebook utilization.

Usage:
    python -m world_models.stoch_muzero.train_vq_v2 --game 2048 --train_steps 20000
    python -m world_models.stoch_muzero.train_vq_v2 --game othello --train_steps 20000

Improvements over train_vq.py:
- Uses vq_model_v2 with dead code reset
- Reports codebook utilization during training  
- Detailed entropy analysis at the end
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, Dict, Optional

from .vq_model_v2 import VQWorldModel, VQWorldModelConfig


def generate_2048_trajectories(
    n_trajectories: int,
    max_steps: int,
    img_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2048 trajectories with rendered images."""
    try:
        from ..grid.game_2048 import Game2048
        from ..grid.renderer import GameRenderer
    except ImportError:
        from world_models.grid.game_2048 import Game2048
        from world_models.grid.renderer import GameRenderer
    
    renderer = GameRenderer(img_size=img_size)
    
    all_obs = []
    all_actions = []
    
    for _ in tqdm(range(n_trajectories), desc="Generating 2048"):
        game = Game2048()
        obs_traj = [renderer.render_2048(game.board)]
        act_traj = []
        
        for _ in range(max_steps):
            action = np.random.randint(0, 4)
            prev_board = game.board.copy()
            game.step(action)
            
            if game.done or np.array_equal(prev_board, game.board):
                # Game over or invalid move
                if game.done:
                    break
            
            obs_traj.append(renderer.render_2048(game.board))
            act_traj.append(action)
            
            if len(act_traj) >= max_steps:
                break
        
        if len(act_traj) >= 2:
            all_obs.append(np.stack(obs_traj[:len(act_traj)+1]))
            all_actions.append(np.array(act_traj))
    
    # Pad to same length
    max_len = max(len(a) for a in all_actions)
    obs_padded = []
    act_padded = []
    
    for obs, act in zip(all_obs, all_actions):
        T = len(act)
        if T < max_len:
            obs_pad = np.concatenate([obs, np.repeat(obs[-1:], max_len - T, axis=0)], axis=0)
            act_pad = np.concatenate([act, np.zeros(max_len - T, dtype=np.int64)])
        else:
            obs_pad = obs[:max_len+1]
            act_pad = act[:max_len]
        obs_padded.append(obs_pad)
        act_padded.append(act_pad)
    
    obs_array = np.stack(obs_padded)[:, :, np.newaxis, :, :]  # [N, T+1, 1, H, W]
    act_array = np.stack(act_padded)  # [N, T]
    
    return obs_array.astype(np.float32), act_array.astype(np.int64)


def generate_othello_trajectories(
    n_trajectories: int,
    max_steps: int,
    img_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Othello trajectories with rendered images."""
    try:
        from ..grid.othello import OthelloGame
        from ..grid.renderer import GameRenderer
    except ImportError:
        from world_models.grid.othello import OthelloGame
        from world_models.grid.renderer import GameRenderer
    
    renderer = GameRenderer(img_size=img_size)
    
    all_obs = []
    all_actions = []
    
    for _ in tqdm(range(n_trajectories), desc="Generating Othello"):
        game = OthelloGame()
        obs_traj = [renderer.render_othello(game.board, game.current_player)]
        act_traj = []
        
        for _ in range(max_steps):
            valid = game.get_valid_moves()
            if len(valid) == 0:
                game.current_player = -game.current_player
                valid = game.get_valid_moves()
                if len(valid) == 0:
                    break
            
            action = valid[np.random.randint(len(valid))]
            game.make_move(action)
            
            obs_traj.append(renderer.render_othello(game.board, game.current_player))
            act_traj.append(action)
        
        if len(act_traj) >= 2:
            all_obs.append(np.stack(obs_traj[:len(act_traj)+1]))
            all_actions.append(np.array(act_traj))
    
    # Pad
    max_len = max(len(a) for a in all_actions)
    obs_padded = []
    act_padded = []
    
    for obs, act in zip(all_obs, all_actions):
        T = len(act)
        if T < max_len:
            obs_pad = np.concatenate([obs, np.repeat(obs[-1:], max_len - T, axis=0)], axis=0)
            act_pad = np.concatenate([act, np.zeros(max_len - T, dtype=np.int64)])
        else:
            obs_pad = obs[:max_len+1]
            act_pad = act[:max_len]
        obs_padded.append(obs_pad)
        act_padded.append(act_pad)
    
    obs_array = np.stack(obs_padded)[:, :, np.newaxis, :, :]
    act_array = np.stack(act_padded)
    
    return obs_array.astype(np.float32), act_array.astype(np.int64)


def generate_trajectories(
    game: str,
    n_trajectories: int,
    max_steps: int,
    img_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    if game.lower() == '2048':
        return generate_2048_trajectories(n_trajectories, max_steps, img_size)
    elif game.lower() == 'othello':
        return generate_othello_trajectories(n_trajectories, max_steps, img_size)
    else:
        raise ValueError(f"Unknown game: {game}")


def analyze_entropy(
    model: VQWorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Analyze entropy distribution with detailed per-position stats.
    
    Returns:
        entropy_per_step: [n_steps, n_positions] entropy in bits
        positions_changed: [n_steps, n_positions] bool
    """
    model.eval()
    
    all_entropy = []
    all_changed = []
    all_indices_before = []
    all_indices_after = []
    
    with torch.no_grad():
        for i in range(min(1000, obs.shape[0])):
            obs_seq = obs[i:i+1].to(device)
            act_seq = actions[i:i+1].to(device)
            
            T = act_seq.shape[1]
            
            # Encode all frames
            B, Tp1, C, H, W = obs_seq.shape
            obs_flat = obs_seq.reshape(B * Tp1, C, H, W)
            enc = model.encode(obs_flat, training=False)
            all_indices = enc['indices'].reshape(B, Tp1, -1)  # [1, T+1, N]
            
            for t in range(T):
                z_q = model.quantizer.embedding(all_indices[:, t])
                step_result = model.step(z_q, act_seq[:, t], sample=False)
                
                entropy_t = step_result['entropy'].cpu().numpy()  # Mean entropy
                
                # Per-position entropy from logits
                logits = step_result['logits']  # [1, N, K]
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                ent_per_pos = -(probs * log_probs).sum(dim=-1) / np.log(2)  # [1, N]
                ent_per_pos = ent_per_pos[0].cpu().numpy()  # [N]
                
                # Which positions changed?
                idx_before = all_indices[0, t].cpu().numpy()
                idx_after = all_indices[0, t+1].cpu().numpy()
                changed = (idx_before != idx_after)
                
                all_entropy.append(ent_per_pos)
                all_changed.append(changed)
                all_indices_before.append(idx_before)
                all_indices_after.append(idx_after)
    
    model.train()
    
    return {
        'entropy_per_position': np.stack(all_entropy),  # [n_steps, n_positions]
        'positions_changed': np.stack(all_changed),      # [n_steps, n_positions]
        'indices_before': np.stack(all_indices_before),
        'indices_after': np.stack(all_indices_after),
    }


def train(
    game: str = '2048',
    train_steps: int = 20000,
    batch_size: int = 32,
    lr: float = 3e-4,
    n_trajectories: int = 2000,
    max_traj_len: int = 50,
    img_size: int = 64,
    codebook_size: int = 512,
    code_dim: int = 64,
    device: str = None,
    seed: int = 42,
):
    """Train VQ-VAE world model v2."""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"VQ-VAE World Model v2 Training")
    print(f"Game: {game} | Steps: {train_steps} | Device: {device}")
    print(f"{'='*60}\n")
    
    # Generate data
    print("Generating trajectories...")
    n_actions = 4 if game.lower() == '2048' else 64
    obs, actions = generate_trajectories(game, n_trajectories, max_traj_len, img_size)
    
    # Keep data on CPU, only move batches to GPU (saves ~663 MB VRAM)
    obs_t = torch.from_numpy(obs)
    actions_t = torch.from_numpy(actions)
    
    print(f"Data: {obs.shape[0]} trajectories, {obs.shape[1]-1} steps each")
    print(f"Obs shape: {obs.shape}, Actions shape: {actions.shape}\n")
    
    # Create model
    cfg = VQWorldModelConfig(
        img_size=img_size,
        n_actions=n_actions,
        codebook_size=codebook_size,
        code_dim=code_dim,
        ema_decay=0.95,            # Lower for faster adaptation
        dead_code_threshold=2,     # Reset codes used < 2 times
        reset_every=100,           # Check every 100 steps
    )
    
    model = VQWorldModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Training loop
    model.train()
    pbar = tqdm(range(1, train_steps + 1))
    
    total_resets = 0
    best_usage = 0.0
    
    for step in pbar:
        idx = torch.randint(0, obs_t.shape[0], (batch_size,))
        obs_batch = obs_t[idx].to(device)
        action_batch = actions_t[idx].to(device)

        losses = model.compute_loss(obs_batch, action_batch, unroll_steps=3)
        
        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_resets += losses['codes_reset']
        best_usage = max(best_usage, losses['codebook_usage'])
        
        if step % 100 == 0:
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'recon': f"{losses['recon_loss'].item():.4f}",
                'trans': f"{losses['transition_loss'].item():.4f}",
                'ent': f"{losses['entropy'].item():.3f}",
                'codes': f"{losses['unique_codes']}/{codebook_size}",
            })
        
        if step % 2000 == 0:
            stats = model.get_codebook_stats()
            print(f"\n[Step {step}] Codebook: {stats['active_codes']}/{codebook_size} "
                  f"({100*stats['active_codes']/codebook_size:.1f}%), "
                  f"Total resets: {total_resets}")
    
    # Final analysis
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - Final Analysis")
    print(f"{'='*60}\n")
    
    # Codebook stats
    stats = model.get_codebook_stats()
    print(f"Final codebook utilization:")
    print(f"  Active codes: {stats['active_codes']}/{codebook_size} ({100*stats['active_codes']/codebook_size:.1f}%)")
    print(f"  Best usage seen: {100*best_usage:.1f}%")
    print(f"  Total codes reset: {total_resets}")
    
    # Entropy analysis
    print("\nAnalyzing entropy distribution...")
    analysis = analyze_entropy(model, obs_t, actions_t, device)
    
    entropy_all = analysis['entropy_per_position']  # [n_steps, n_pos]
    changed = analysis['positions_changed']         # [n_steps, n_pos]
    
    # Overall stats
    print(f"\nOverall entropy (per-position):")
    print(f"  Mean: {entropy_all.mean():.4f} bits")
    print(f"  Std:  {entropy_all.std():.4f}")
    print(f"  Max:  {entropy_all.max():.4f}")
    
    # Key metric: entropy conditioned on change
    entropy_changed = entropy_all[changed]
    entropy_unchanged = entropy_all[~changed]
    
    print(f"\nEntropy vs actual code changes:")
    print(f"  Positions that CHANGED ({len(entropy_changed)} samples):")
    print(f"    Mean entropy: {entropy_changed.mean():.4f} bits")
    print(f"    Median:       {np.median(entropy_changed):.4f} bits")
    
    print(f"  Positions that STAYED SAME ({len(entropy_unchanged)} samples):")
    print(f"    Mean entropy: {entropy_unchanged.mean():.4f} bits")
    print(f"    Median:       {np.median(entropy_unchanged):.4f} bits")
    
    if len(entropy_changed) > 0 and entropy_unchanged.mean() > 1e-6:
        ratio = entropy_changed.mean() / entropy_unchanged.mean()
        print(f"\n  ** RATIO: {ratio:.1f}x higher entropy for changed positions **")
    
    # Distribution analysis
    print(f"\nEntropy distribution:")
    print(f"  <0.1 bits (certain):    {100*(entropy_all < 0.1).mean():.1f}%")
    print(f"  0.1-0.5 bits:           {100*((entropy_all >= 0.1) & (entropy_all < 0.5)).mean():.1f}%")
    print(f"  0.5-2.0 bits:           {100*((entropy_all >= 0.5) & (entropy_all < 2.0)).mean():.1f}%")
    print(f"  >2.0 bits (uncertain):  {100*(entropy_all > 2.0).mean():.1f}%")
    
    # Unique codes used
    all_idx = np.concatenate([analysis['indices_before'].flatten(), 
                              analysis['indices_after'].flatten()])
    unique = len(np.unique(all_idx))
    print(f"\nCodes used in analysis: {unique}/{codebook_size} ({100*unique/codebook_size:.1f}%)")
    
    return model, analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='2048', choices=['2048', 'othello'])
    parser.add_argument('--train_steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_trajectories', type=int, default=2000)
    parser.add_argument('--max_traj_len', type=int, default=50)
    parser.add_argument('--codebook_size', type=int, default=512)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train(
        game=args.game,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        n_trajectories=args.n_trajectories,
        max_traj_len=args.max_traj_len,
        codebook_size=args.codebook_size,
        device=args.device,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
