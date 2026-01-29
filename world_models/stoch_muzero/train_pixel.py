"""
Training script for pixel-based world model.

This trains the full pipeline:
    Pixels → Encoder → State → Dynamics → Afterstate → Chance → Next State
                                    ↑                      ↑
                              RULE CORE              STOCHASTIC
                         (should compress)         (should match oracle)

Usage:
    python -m world_models.stoch_muzero.train_pixel --game 2048 --train_steps 10000
    python -m world_models.stoch_muzero.train_pixel --game othello --train_steps 10000
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

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
    analyze_entropy_distribution,
)
from .pixel_games import (
    PixelGame2048,
    PixelOthello,
    collect_pixel_episodes,
)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Loss Functions
# =============================================================================

def compute_training_loss(
    model: PixelWorldModel,
    obs_batch: torch.Tensor,      # [B, T+1, C, H, W]
    action_batch: torch.Tensor,   # [B, T]
    unroll_steps: int = 5,
    w_recon: float = 1.0,
    w_consistency: float = 1.0,
    w_dynamics: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute training loss for the pixel world model.
    
    Loss components:
    1. Reconstruction: decode encoded states back to pixels
    2. Consistency: predicted next state ≈ encoded next observation
    3. Dynamics: afterstate prediction accuracy (via decoding)
    
    Returns dict of losses.
    """
    B, Tp1, C, H, W = obs_batch.shape
    T = Tp1 - 1
    device = obs_batch.device
    
    # Limit unroll to available trajectory length
    unroll = min(unroll_steps, T)
    
    # Encode initial observation
    obs_0 = obs_batch[:, 0]  # [B, C, H, W]
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
        
        # Decode afterstate (dynamics supervision)
        # Note: we don't have ground truth afterstate pixels, so this is weaker
        # supervision. The main signal comes from consistency.
        
        # Consistency: predicted next state should match encoded next observation
        obs_next = obs_batch[:, t + 1]  # [B, C, H, W]
        state_next_encoded = model.encode(obs_next)
        
        consistency_loss_t = F.mse_loss(next_state_pred, state_next_encoded.detach())
        total_consistency_loss = total_consistency_loss + consistency_loss_t
        
        # For dynamics loss: decode predicted state, compare to actual next obs
        pred_next_obs = model.decode(next_state_pred)
        dynamics_loss_t = reconstruction_loss(pred_next_obs, obs_next)
        total_dynamics_loss = total_dynamics_loss + dynamics_loss_t
        
        # Move to next state (using encoded for teacher forcing)
        state = state_next_encoded
    
    # Average over steps
    n_steps = float(unroll)
    avg_recon = total_recon_loss / n_steps
    avg_consistency = total_consistency_loss / n_steps
    avg_dynamics = total_dynamics_loss / n_steps
    avg_entropy = total_entropy / n_steps
    
    total_loss = (
        w_recon * avg_recon +
        w_consistency * avg_consistency +
        w_dynamics * avg_dynamics
    )
    
    return {
        'total_loss': total_loss,
        'recon_loss': avg_recon,
        'consistency_loss': avg_consistency,
        'dynamics_loss': avg_dynamics,
        'avg_entropy': avg_entropy,
    }


def compute_training_loss_imagination(
    model: PixelWorldModel,
    obs_batch: torch.Tensor,      # [B, T+1, C, H, W]
    action_batch: torch.Tensor,   # [B, T]
    unroll_steps: int = 5,
    w_recon: float = 1.0,
    w_imagination: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Alternative loss: imagine forward from initial state WITHOUT teacher forcing.
    
    This tests whether the model can maintain accurate predictions over time,
    which is critical for MCTS planning.
    """
    B, Tp1, C, H, W = obs_batch.shape
    T = Tp1 - 1
    
    unroll = min(unroll_steps, T)
    
    # Encode initial observation
    state = model.encode(obs_batch[:, 0])
    
    total_recon_loss = 0.0
    total_imagination_loss = 0.0
    total_entropy = 0.0
    
    for t in range(unroll):
        # Decode and compare
        pred_obs = model.decode(state)
        target_obs = obs_batch[:, t]
        total_recon_loss = total_recon_loss + reconstruction_loss(pred_obs, target_obs)
        
        # Step dynamics (no teacher forcing - use predicted state)
        action = action_batch[:, t]
        result = model.step(state, action, sample_chance=False)
        state = result['next_state']  # Use predicted, not encoded
        
        total_entropy = total_entropy + result['chance_entropy'].mean()
    
    # Final step prediction
    pred_obs_final = model.decode(state)
    target_obs_final = obs_batch[:, unroll]
    imagination_loss = reconstruction_loss(pred_obs_final, target_obs_final)
    total_imagination_loss = imagination_loss * unroll  # weight by horizon
    
    n_steps = float(unroll)
    
    return {
        'total_loss': w_recon * (total_recon_loss / n_steps) + w_imagination * imagination_loss,
        'recon_loss': total_recon_loss / n_steps,
        'imagination_loss': imagination_loss,
        'avg_entropy': total_entropy / n_steps,
    }


# =============================================================================
# Evaluation
# =============================================================================

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
        
        losses = compute_training_loss(
            model, obs_batch, action_batch,
            unroll_steps=unroll_steps,
        )
        
        total_recon += losses['recon_loss'].item()
        total_dynamics += losses['dynamics_loss'].item()
        total_entropy += losses['avg_entropy'].item()
        n_batches += 1
    
    model.train()
    
    return {
        'val_recon_loss': total_recon / max(1, n_batches),
        'val_dynamics_loss': total_dynamics / max(1, n_batches),
        'val_avg_entropy': total_entropy / max(1, n_batches),
    }


# =============================================================================
# Main Training
# =============================================================================

def train_pixel_model(
    game: str = '2048',
    collect_episodes: int = 500,
    max_steps_per_episode: int = 200,
    train_steps: int = 10000,
    batch_size: int = 32,
    unroll_steps: int = 5,
    lr: float = 3e-4,
    img_size: int = 64,
    state_dim: int = 256,
    n_chance_outcomes: int = 32,
    w_recon: float = 1.0,
    w_consistency: float = 1.0,
    w_dynamics: float = 1.0,
    eval_every: int = 500,
    log_every: int = 100,
    seed: int = 0,
    device: str = None,
    output_dir: str = 'outputs_pixel_wm',
):
    """Main training function."""
    
    set_seed(seed)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_dir = Path(output_dir) / game
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training pixel world model for {game}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Select game
    if game.lower() == '2048':
        game_class = PixelGame2048
        n_actions = 4
    elif game.lower() == 'othello':
        game_class = PixelOthello
        n_actions = 64
    else:
        raise ValueError(f"Unknown game: {game}")
    
    # Collect data
    print(f"\nCollecting {collect_episodes} episodes...")
    obs_data, action_data, done_data = collect_pixel_episodes(
        game_class,
        n_episodes=collect_episodes,
        max_steps=max_steps_per_episode,
        img_size=img_size,
    )
    print(f"  obs shape: {obs_data.shape}")
    print(f"  actions shape: {action_data.shape}")
    
    # Convert to tensors
    # obs_data: [N, T+1, H, W] → [N, T+1, 1, H, W]
    obs_tensor = torch.from_numpy(obs_data).float().unsqueeze(2)
    action_tensor = torch.from_numpy(action_data).long()
    
    # Split train/val
    n_total = obs_tensor.shape[0]
    n_val = max(10, n_total // 10)
    n_train = n_total - n_val
    
    train_dataset = TensorDataset(obs_tensor[:n_train], action_tensor[:n_train])
    val_dataset = TensorDataset(obs_tensor[n_train:], action_tensor[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  train episodes: {n_train}")
    print(f"  val episodes: {n_val}")
    
    # Build model
    cfg = PixelModelConfig(
        in_channels=1,
        img_size=img_size,
        state_dim=state_dim,
        afterstate_dim=state_dim,
        n_actions=n_actions,
        n_chance_outcomes=n_chance_outcomes,
    )
    
    model = PixelWorldModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Save config
    config = {
        'game': game,
        'collect_episodes': collect_episodes,
        'max_steps_per_episode': max_steps_per_episode,
        'train_steps': train_steps,
        'batch_size': batch_size,
        'unroll_steps': unroll_steps,
        'lr': lr,
        'img_size': img_size,
        'state_dim': state_dim,
        'n_chance_outcomes': n_chance_outcomes,
        'n_actions': n_actions,
        'n_params': n_params,
        'seed': seed,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print(f"\nTraining for {train_steps} steps...")
    
    metrics_log = []
    step = 0
    epoch = 0
    best_val_loss = float('inf')
    
    pbar = tqdm(total=train_steps, desc='Training')
    
    while step < train_steps:
        epoch += 1
        
        for obs_batch, action_batch in train_loader:
            if step >= train_steps:
                break
            
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            
            # Compute loss
            losses = compute_training_loss(
                model, obs_batch, action_batch,
                unroll_steps=unroll_steps,
                w_recon=w_recon,
                w_consistency=w_consistency,
                w_dynamics=w_dynamics,
            )
            
            # Backward
            optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            pbar.update(1)
            
            # Logging
            if step % log_every == 0:
                metrics = {
                    'step': step,
                    'epoch': epoch,
                    'total_loss': losses['total_loss'].item(),
                    'recon_loss': losses['recon_loss'].item(),
                    'consistency_loss': losses['consistency_loss'].item(),
                    'dynamics_loss': losses['dynamics_loss'].item(),
                    'avg_entropy': losses['avg_entropy'].item(),
                }
                metrics_log.append(metrics)
                
                pbar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'dyn': f"{losses['dynamics_loss'].item():.4f}",
                    'ent': f"{losses['avg_entropy'].item():.2f}",
                })
            
            # Evaluation
            if step % eval_every == 0:
                val_metrics = evaluate_model(model, val_loader, unroll_steps, device)
                
                pbar.write(
                    f"[Step {step}] "
                    f"val_recon={val_metrics['val_recon_loss']:.4f} "
                    f"val_dyn={val_metrics['val_dynamics_loss']:.4f} "
                    f"val_ent={val_metrics['val_avg_entropy']:.2f}"
                )
                
                # Save best
                val_loss = val_metrics['val_dynamics_loss']
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model': model.state_dict(),
                        'config': cfg,
                        'step': step,
                        'val_metrics': val_metrics,
                    }, output_dir / 'ckpt_best.pt')
    
    pbar.close()
    
    # Final save
    torch.save({
        'model': model.state_dict(),
        'config': cfg,
        'step': step,
    }, output_dir / 'ckpt_final.pt')
    
    # Save metrics
    import pandas as pd
    pd.DataFrame(metrics_log).to_csv(output_dir / 'metrics.csv', index=False)
    
    # Final entropy analysis
    print("\nAnalyzing entropy distribution...")
    
    # Get a batch for analysis
    obs_sample = obs_tensor[:min(50, n_total)].to(device)
    action_sample = action_tensor[:min(50, n_total)].to(device)
    
    entropy_stats = analyze_entropy_distribution(model, obs_sample, action_sample)
    
    print("Entropy Distribution:")
    for k, v in entropy_stats.items():
        print(f"  {k}: {v:.4f}")
    
    with open(output_dir / 'entropy_analysis.json', 'w') as f:
        json.dump(entropy_stats, f, indent=2)
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if game.lower() == '2048':
        print("""
For 2048:
- Afterstate (slide+merge) is DETERMINISTIC
- Chance node (tile spawn) is STOCHASTIC

Expected entropy distribution:
- Most transitions: LOW entropy (deterministic slides)
- Tile spawn steps: HIGHER entropy (uncertainty about position)

If mean_entropy is very low (<0.5 bits), the model may be:
1. Collapsing chance predictions to a single outcome
2. Not capturing the tile spawn randomness

If mean_entropy is very high (>3 bits), the model:
1. Hasn't learned the deterministic structure
2. May need more training
        """)
    else:
        print("""
For Othello:
- ALL transitions are DETERMINISTIC (no randomness)

Expected entropy distribution:
- ALL transitions should have near-zero entropy

If mean_entropy is high (>0.5 bits), the model:
1. Hasn't learned that the game is deterministic
2. May need more training or capacity
        """)
    
    print(f"\nTraining complete! Outputs saved to: {output_dir}")
    return model, entropy_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='2048', choices=['2048', 'othello'])
    parser.add_argument('--collect_episodes', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--train_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--unroll_steps', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--state_dim', type=int, default=256)
    parser.add_argument('--n_chance_outcomes', type=int, default=32)
    parser.add_argument('--w_recon', type=float, default=1.0)
    parser.add_argument('--w_consistency', type=float, default=1.0)
    parser.add_argument('--w_dynamics', type=float, default=1.0)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs_pixel_wm')
    
    args = parser.parse_args()
    
    train_pixel_model(
        game=args.game,
        collect_episodes=args.collect_episodes,
        max_steps_per_episode=args.max_steps,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        unroll_steps=args.unroll_steps,
        lr=args.lr,
        img_size=args.img_size,
        state_dim=args.state_dim,
        n_chance_outcomes=args.n_chance_outcomes,
        w_recon=args.w_recon,
        w_consistency=args.w_consistency,
        w_dynamics=args.w_dynamics,
        eval_every=args.eval_every,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
