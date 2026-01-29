"""
Pixel World Model Training V2 - With Entropy Regularization

Key fix: The original training had no entropy supervision. The model could:
- Collapse to deterministic (2048 case): predict mode, ignore uncertainty
- Stay high entropy (Othello case): no signal to learn certainty

This version adds:
1. ENTROPY BONUS for stochastic games (prevent collapse, encourage modeling uncertainty)
2. ENTROPY PENALTY for deterministic games (encourage confident predictions)

Usage:
    # 2048 - stochastic game, use entropy bonus to prevent collapse
    python -m world_models.stoch_muzero.train_pixel_v2 --game 2048 --entropy_bonus 0.01

    # Othello - deterministic game, use entropy penalty to encourage certainty  
    python -m world_models.stoch_muzero.train_pixel_v2 --game othello --entropy_penalty 0.1
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .pixel_model import (
    PixelModelConfig,
    PixelWorldModel,
    reconstruction_loss,
)
from .pixel_games import (
    PixelGame2048,
    PixelOthello,
    collect_pixel_episodes,
)


def compute_training_loss_v2(
    model: PixelWorldModel,
    obs_batch: torch.Tensor,      # [B, T+1, C, H, W]
    action_batch: torch.Tensor,   # [B, T]
    unroll_steps: int = 5,
    w_recon: float = 1.0,
    w_consistency: float = 1.0,
    w_dynamics: float = 1.0,
    entropy_bonus: float = 0.0,   # NEW: Positive = encourage high entropy (for stochastic games)
    entropy_penalty: float = 0.0, # NEW: Positive = encourage low entropy (for deterministic games)
) -> Dict[str, torch.Tensor]:
    """
    Training loss with entropy regularization.
    
    Key insight:
    - For 2048: entropy_bonus > 0 prevents collapse to deterministic predictions
    - For Othello: entropy_penalty > 0 encourages learning that it's deterministic
    """
    B, Tp1, C, H, W = obs_batch.shape
    T = Tp1 - 1
    
    unroll = min(unroll_steps, T)
    
    # Encode initial observation
    obs_0 = obs_batch[:, 0]
    state = model.encode(obs_0)
    
    total_recon_loss = 0.0
    total_consistency_loss = 0.0
    total_dynamics_loss = 0.0
    total_entropy = 0.0
    
    for t in range(unroll):
        # Decode current state → pixel prediction
        pred_obs = model.decode(state)
        target_obs = obs_batch[:, t]
        recon_loss_t = reconstruction_loss(pred_obs, target_obs)
        total_recon_loss = total_recon_loss + recon_loss_t
        
        # Take dynamics step
        action = action_batch[:, t]
        result = model.step(state, action, sample_chance=False)
        
        afterstate = result['afterstate']
        next_state_pred = result['next_state']
        chance_entropy = result['chance_entropy']
        
        total_entropy = total_entropy + chance_entropy.mean()
        
        # Consistency: predicted next state should match encoded next observation
        obs_next = obs_batch[:, t + 1]
        state_next_encoded = model.encode(obs_next)
        consistency_loss_t = F.mse_loss(next_state_pred, state_next_encoded.detach())
        total_consistency_loss = total_consistency_loss + consistency_loss_t
        
        # Dynamics loss: decode predicted state, compare to actual next obs
        pred_next_obs = model.decode(next_state_pred)
        dynamics_loss_t = reconstruction_loss(pred_next_obs, obs_next)
        total_dynamics_loss = total_dynamics_loss + dynamics_loss_t
        
        # Teacher forcing
        state = state_next_encoded
    
    # Average over steps
    n_steps = float(unroll)
    avg_recon = total_recon_loss / n_steps
    avg_consistency = total_consistency_loss / n_steps
    avg_dynamics = total_dynamics_loss / n_steps
    avg_entropy = total_entropy / n_steps
    
    # ENTROPY REGULARIZATION
    # Bonus: subtract entropy (encourage high entropy to prevent collapse)
    # Penalty: add entropy (encourage low entropy for deterministic games)
    entropy_loss = -entropy_bonus * avg_entropy + entropy_penalty * avg_entropy
    
    total_loss = (
        w_recon * avg_recon +
        w_consistency * avg_consistency +
        w_dynamics * avg_dynamics +
        entropy_loss
    )
    
    return {
        'total_loss': total_loss,
        'recon_loss': avg_recon,
        'consistency_loss': avg_consistency,
        'dynamics_loss': avg_dynamics,
        'entropy_loss': entropy_loss,
        'avg_entropy': avg_entropy,
    }


@torch.no_grad()
def evaluate_model(
    model: PixelWorldModel,
    val_loader: DataLoader,
    unroll_steps: int = 5,
    device: str = 'cpu',
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_recon = 0.0
    total_dynamics = 0.0
    total_entropy = 0.0
    n_batches = 0
    
    for obs_batch, action_batch in val_loader:
        obs_batch = obs_batch.to(device)
        action_batch = action_batch.to(device)
        
        losses = compute_training_loss_v2(
            model, obs_batch, action_batch,
            unroll_steps=unroll_steps,
        )
        
        total_recon += losses['recon_loss'].item()
        total_dynamics += losses['dynamics_loss'].item()
        total_entropy += losses['avg_entropy'].item()
        n_batches += 1
    
    model.train()
    
    return {
        'val_recon': total_recon / max(1, n_batches),
        'val_dynamics': total_dynamics / max(1, n_batches),
        'val_entropy': total_entropy / max(1, n_batches),
    }


def analyze_entropy_distribution(
    model: PixelWorldModel,
    val_loader: DataLoader,
    unroll_steps: int = 5,
    device: str = 'cpu',
) -> Dict[str, float]:
    """Analyze the distribution of predicted entropies."""
    model.eval()
    
    all_entropies = []
    
    with torch.no_grad():
        for obs_batch, action_batch in val_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            
            B, Tp1, C, H, W = obs_batch.shape
            T = Tp1 - 1
            unroll = min(unroll_steps, T)
            
            state = model.encode(obs_batch[:, 0])
            
            for t in range(unroll):
                result = model.step(state, action_batch[:, t], sample_chance=False)
                all_entropies.extend(result['chance_entropy'].cpu().numpy().tolist())
                state = model.encode(obs_batch[:, t + 1])
    
    model.train()
    
    all_entropies = np.array(all_entropies)
    
    return {
        'mean_entropy': float(all_entropies.mean()),
        'std_entropy': float(all_entropies.std()),
        'min_entropy': float(all_entropies.min()),
        'max_entropy': float(all_entropies.max()),
        'median_entropy': float(np.median(all_entropies)),
        'p10_entropy': float(np.percentile(all_entropies, 10)),
        'p90_entropy': float(np.percentile(all_entropies, 90)),
        'frac_below_0.1': float((all_entropies < 0.1).mean()),
        'frac_below_0.5': float((all_entropies < 0.5).mean()),
        'frac_below_1.0': float((all_entropies < 1.0).mean()),
        'frac_above_2.0': float((all_entropies > 2.0).mean()),
    }


def train_pixel_model_v2(
    game: str = '2048',
    # Data collection
    collect_episodes: int = 500,
    max_steps_per_episode: int = 200,
    # Model config
    img_size: int = 64,
    state_dim: int = 256,
    afterstate_dim: int = 256,
    n_chance_outcomes: int = 32,
    hidden_dim: int = 256,
    # Training
    train_steps: int = 15000,
    batch_size: int = 32,
    lr: float = 1e-3,
    unroll_steps: int = 5,
    # Loss weights
    w_recon: float = 1.0,
    w_consistency: float = 1.0,
    w_dynamics: float = 1.0,
    entropy_bonus: float = 0.0,   # For stochastic games
    entropy_penalty: float = 0.0, # For deterministic games
    # Logging
    eval_every: int = 2000,
    log_every: int = 100,
    # Output
    outdir: str = 'outputs_pixel_v2',
    device: str = None,
) -> Dict:
    """Train pixel-based world model with entropy regularization."""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Configure for game type
    if game == '2048':
        GameClass = PixelGame2048
        n_actions = 4
        is_stochastic = True
        # Auto-set entropy bonus if not specified
        if entropy_bonus == 0.0 and entropy_penalty == 0.0:
            entropy_bonus = 0.01
            print(f"Auto-setting entropy_bonus={entropy_bonus} for stochastic game 2048")
    elif game == 'othello':
        GameClass = PixelOthello
        n_actions = 64
        is_stochastic = False
        # Auto-set entropy penalty if not specified
        if entropy_bonus == 0.0 and entropy_penalty == 0.0:
            entropy_penalty = 0.1
            print(f"Auto-setting entropy_penalty={entropy_penalty} for deterministic game Othello")
    else:
        raise ValueError(f"Unknown game: {game}")
    
    print(f"\n{'='*60}")
    print(f"Training {game.upper()} (stochastic={is_stochastic})")
    print(f"Loss weights: recon={w_recon}, consistency={w_consistency}, dynamics={w_dynamics}")
    print(f"Entropy: bonus={entropy_bonus}, penalty={entropy_penalty}")
    print(f"{'='*60}\n")
    
    # Collect data
    print(f"Collecting {collect_episodes} episodes...")
    obs, actions, dones = collect_pixel_episodes(
        GameClass,
        n_episodes=collect_episodes,
        max_steps=max_steps_per_episode,
        img_size=img_size,
    )
    
    print(f"Data shape: obs={obs.shape}, actions={actions.shape}")
    
    # Prepare data
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(2)  # [N, T, 1, H, W]
    actions_tensor = torch.from_numpy(actions).long()
    
    # Split train/val
    n_total = obs_tensor.shape[0]
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    
    train_obs = obs_tensor[:n_train]
    train_actions = actions_tensor[:n_train]
    val_obs = obs_tensor[n_train:]
    val_actions = actions_tensor[n_train:]
    
    train_dataset = TensorDataset(train_obs, train_actions)
    val_dataset = TensorDataset(val_obs, val_actions)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    cfg = PixelModelConfig(
        img_size=img_size,
        n_actions=n_actions,
        state_dim=state_dim,
        afterstate_dim=afterstate_dim,
        n_chance_outcomes=n_chance_outcomes,
        hidden_dim=hidden_dim,
    )
    model = PixelWorldModel(cfg).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\nTraining for {train_steps} steps...")
    
    step = 0
    train_iter = iter(train_loader)
    
    pbar = tqdm(total=train_steps, desc='Training')
    
    while step < train_steps:
        # Get batch
        try:
            obs_batch, action_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            obs_batch, action_batch = next(train_iter)
        
        obs_batch = obs_batch.to(device)
        action_batch = action_batch.to(device)
        
        # Forward pass
        losses = compute_training_loss_v2(
            model, obs_batch, action_batch,
            unroll_steps=unroll_steps,
            w_recon=w_recon,
            w_consistency=w_consistency,
            w_dynamics=w_dynamics,
            entropy_bonus=entropy_bonus,
            entropy_penalty=entropy_penalty,
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        
        # Logging
        if step % log_every == 0:
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'dyn': f"{losses['dynamics_loss'].item():.4f}",
                'ent': f"{losses['avg_entropy'].item():.2f}",
            })
        
        # Evaluation
        if step % eval_every == 0:
            val_metrics = evaluate_model(model, val_loader, unroll_steps, device)
            print(f"\n[Step {step}] "
                  f"val_recon={val_metrics['val_recon']:.4f} "
                  f"val_dyn={val_metrics['val_dynamics']:.4f} "
                  f"val_ent={val_metrics['val_entropy']:.2f}")
        
        pbar.update(1)
    
    pbar.close()
    
    # Analyze entropy distribution
    print("\nAnalyzing entropy distribution...")
    entropy_stats = analyze_entropy_distribution(model, val_loader, unroll_steps, device)
    
    print("\nEntropy Distribution:")
    for k, v in entropy_stats.items():
        print(f"  {k}: {v:.4f}")
    
    # Expected vs actual
    print("\n" + "="*60)
    if is_stochastic:
        print(f"EXPECTED: Bimodal entropy (some low, some high)")
        print(f"  - Deterministic transitions: entropy ~ 0")
        print(f"  - Stochastic transitions (tile spawn): entropy ~ 2-4 bits")
    else:
        print(f"EXPECTED: All near-zero entropy (deterministic)")
        print(f"  - All transitions should have entropy < 0.5")
    
    print(f"\nACTUAL:")
    print(f"  - Mean entropy: {entropy_stats['mean_entropy']:.2f} bits")
    print(f"  - Fraction below 0.5: {entropy_stats['frac_below_0.5']*100:.1f}%")
    print(f"  - Fraction above 2.0: {entropy_stats['frac_above_2.0']*100:.1f}%")
    print("="*60)
    
    # Save results
    torch.save(model.state_dict(), outdir / 'model_final.pt')
    
    results = {
        'config': {
            'game': game,
            'is_stochastic': is_stochastic,
            'img_size': img_size,
            'state_dim': state_dim,
            'train_steps': train_steps,
            'entropy_bonus': entropy_bonus,
            'entropy_penalty': entropy_penalty,
        },
        'entropy_stats': entropy_stats,
    }
    
    with open(outdir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {outdir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train pixel world model with entropy regularization')
    parser.add_argument('--game', type=str, default='2048', choices=['2048', 'othello'])
    parser.add_argument('--collect_episodes', type=int, default=500)
    parser.add_argument('--train_steps', type=int, default=15000)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--state_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--w_recon', type=float, default=1.0)
    parser.add_argument('--w_consistency', type=float, default=1.0)
    parser.add_argument('--w_dynamics', type=float, default=1.0)
    parser.add_argument('--entropy_bonus', type=float, default=0.0,
                        help='Entropy bonus (for stochastic games, prevents collapse)')
    parser.add_argument('--entropy_penalty', type=float, default=0.0,
                        help='Entropy penalty (for deterministic games, encourages certainty)')
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--outdir', type=str, default='outputs_pixel_v2')
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    train_pixel_model_v2(
        game=args.game,
        collect_episodes=args.collect_episodes,
        train_steps=args.train_steps,
        img_size=args.img_size,
        state_dim=args.state_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        w_recon=args.w_recon,
        w_consistency=args.w_consistency,
        w_dynamics=args.w_dynamics,
        entropy_bonus=args.entropy_bonus,
        entropy_penalty=args.entropy_penalty,
        eval_every=args.eval_every,
        outdir=args.outdir,
        device=args.device,
    )


if __name__ == '__main__':
    main()
