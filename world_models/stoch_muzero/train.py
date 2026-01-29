"""
Training harness for the Causal World Model.

Key features:
1. Proper loss balancing (avoid collapse from value scale)
2. Class-weighted auxiliary losses (fix empty cell dominance)
3. Separate supervision for afterstate (rule core) and chance (stochastic)
4. Invariance training for style separation
5. Comprehensive logging of entropy and accuracy metrics
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .model import CausalWorldModel, ModelConfig, create_model
from .games import Game2048Env, OthelloEnv, collect_2048_game, collect_othello_game


@dataclass
class TrainConfig:
    """Training configuration."""
    game: str = "2048"
    
    # Data collection
    collect_episodes: int = 2000
    max_steps_per_episode: int = 500
    valid_actions_only: bool = True
    
    # Training
    train_steps: int = 30000
    batch_size: int = 128
    unroll_length: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    
    # Loss weights
    w_aux: float = 1.0          # Board prediction from state
    w_after_aux: float = 2.0    # Board prediction from afterstate (RULE CORE)
    w_chance: float = 1.0       # Chance prediction
    w_style: float = 0.2        # Style classification
    w_inv: float = 1.0          # Invariance between augmentations
    w_policy: float = 0.0       # Policy (set to 0 for world-model pretraining)
    w_value: float = 0.0        # Value (set to 0 for world-model pretraining)
    w_reward: float = 0.0       # Reward prediction
    
    # Class balancing (crucial for avoiding "predict empty" collapse)
    empty_weight: float = 0.2   # Weight for empty class in aux loss
    nonempty_weight: float = 1.0
    changed_cell_bonus: float = 3.0  # Extra weight for cells that changed
    
    # Augmentation / styles
    num_styles: int = 1         # Number of rendering styles
    
    # Logging
    log_every: int = 200
    eval_every: int = 2000
    save_every: int = 5000
    
    # Paths
    output_dir: str = "./outputs_stoch_muzero"
    
    def __post_init__(self):
        self.output_dir = os.path.join(self.output_dir, self.game)


class TransitionDataset(Dataset):
    """Dataset of collected game transitions."""
    
    def __init__(self, transitions: List[dict], unroll_length: int, num_classes: int):
        self.transitions = transitions
        self.unroll_length = unroll_length
        self.num_classes = num_classes
        
        # Group by episode for unrolling
        self.episode_starts = []
        current_start = 0
        for i, t in enumerate(transitions):
            if i == 0 or transitions[i-1].get('done', False):
                self.episode_starts.append(i)
        self.episode_starts.append(len(transitions))  # Sentinel
    
    def __len__(self) -> int:
        return max(0, len(self.transitions) - self.unroll_length)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get unroll window
        items = self.transitions[idx:idx + self.unroll_length + 1]
        
        # Handle episode boundaries
        valid_len = self.unroll_length
        for i, t in enumerate(items[:-1]):
            if t.get('done', False):
                valid_len = i
                break
        
        # Pad if necessary
        while len(items) < self.unroll_length + 1:
            items.append(items[-1])
        
        obs = items[0]['obs']
        actions = np.array([t['action'] for t in items[:self.unroll_length]])
        
        # Board targets
        board_before = items[0]['board_before']
        afterstate_boards = np.array([
            t.get('afterstate_board', t['board_after']) 
            for t in items[:self.unroll_length]
        ])
        boards_after = np.array([t['board_after'] for t in items[1:self.unroll_length+1]])
        
        # Chance targets (for 2048)
        spawn_positions = np.array([
            t.get('spawn_position', -1) for t in items[:self.unroll_length]
        ])
        spawn_values = np.array([
            t.get('spawn_value', 0) for t in items[:self.unroll_length]
        ])
        empty_masks = np.array([
            t.get('empty_mask', np.ones(16, dtype=bool))
            for t in items[:self.unroll_length]
        ])
        
        # Compute which cells changed (for weighted loss)
        changed_masks = (afterstate_boards != np.tile(board_before, (self.unroll_length, 1)))
        
        return {
            'obs': torch.from_numpy(obs).float(),
            'actions': torch.from_numpy(actions).long(),
            'board_before': torch.from_numpy(board_before).long(),
            'afterstate_boards': torch.from_numpy(afterstate_boards).long(),
            'boards_after': torch.from_numpy(boards_after).long(),
            'spawn_positions': torch.from_numpy(spawn_positions).long(),
            'spawn_values': torch.from_numpy(spawn_values).long(),
            'empty_masks': torch.from_numpy(empty_masks).bool(),
            'changed_masks': torch.from_numpy(changed_masks).bool(),
            'valid_len': valid_len,
        }


def compute_weighted_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    empty_weight: float = 0.2,
    nonempty_weight: float = 1.0,
    changed_mask: Optional[torch.Tensor] = None,
    changed_bonus: float = 3.0,
) -> torch.Tensor:
    """
    Compute class-weighted cross-entropy loss.
    
    This is crucial for avoiding the "predict empty everywhere" collapse.
    
    Args:
        logits: (B, cells, classes) or (B, classes)
        targets: (B, cells) or (B,)
        empty_weight: Weight for class 0 (empty)
        nonempty_weight: Weight for non-empty classes
        changed_mask: (B, cells) bool - extra weight for cells that changed
        changed_bonus: Multiplier for changed cells
    """
    if logits.dim() == 3:
        B, C, K = logits.shape
        logits = logits.reshape(B * C, K)
        targets = targets.reshape(B * C)
        if changed_mask is not None:
            changed_mask = changed_mask.reshape(B * C)
    
    # Per-class weights
    num_classes = logits.shape[-1]
    class_weights = torch.ones(num_classes, device=logits.device)
    class_weights[0] = empty_weight  # Empty class
    class_weights[1:] = nonempty_weight
    
    # Compute per-sample loss
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')
    
    # Extra weight for changed cells
    if changed_mask is not None:
        weight_mult = torch.ones_like(ce)
        weight_mult[changed_mask] = changed_bonus
        ce = ce * weight_mult
    
    return ce.mean()


def compute_chance_nll(
    position_logits: torch.Tensor,
    value_logits: torch.Tensor,
    target_position: torch.Tensor,
    target_value: torch.Tensor,
    empty_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute negative log-likelihood for chance prediction.
    
    This is the proper metric for stochastic prediction quality.
    
    Args:
        position_logits: (B, cells)
        value_logits: (B, 2)
        target_position: (B,) - actual spawn position
        target_value: (B,) - actual spawn value (1=2-tile, 2=4-tile)
        empty_mask: (B, cells) - True for empty cells
    
    Returns:
        position_nll: (B,)
        value_nll: (B,)
        total_nll: (B,)
    """
    # Mask position logits to valid cells
    masked_logits = position_logits.masked_fill(~empty_mask, float('-inf'))
    
    # Handle all-masked case
    all_masked = (masked_logits == float('-inf')).all(dim=-1)
    masked_logits = masked_logits.masked_fill(all_masked.unsqueeze(-1), 0.0)
    
    # Position NLL
    log_probs_pos = F.log_softmax(masked_logits, dim=-1)
    valid_pos = (target_position >= 0) & (target_position < position_logits.shape[-1])
    position_nll = torch.zeros_like(target_position, dtype=torch.float)
    position_nll[valid_pos] = -log_probs_pos[valid_pos, target_position[valid_pos]]
    
    # Value NLL
    log_probs_val = F.log_softmax(value_logits, dim=-1)
    target_val_idx = (target_value - 1).clamp(0, 1)  # Convert 1,2 → 0,1
    value_nll = -log_probs_val[range(len(target_val_idx)), target_val_idx]
    
    return position_nll, value_nll, position_nll + value_nll


def compute_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute entropy of a probability distribution."""
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=dim)


def train_step(
    model: CausalWorldModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: str,
) -> Dict[str, float]:
    """
    Single training step.
    
    Returns dict of loss components and metrics.
    """
    model.train()
    
    # Move to device
    obs = batch['obs'].to(device)
    actions = batch['actions'].to(device)
    board_before = batch['board_before'].to(device)
    afterstate_boards = batch['afterstate_boards'].to(device)
    boards_after = batch['boards_after'].to(device)
    spawn_positions = batch['spawn_positions'].to(device)
    spawn_values = batch['spawn_values'].to(device)
    empty_masks = batch['empty_masks'].to(device)
    changed_masks = batch['changed_masks'].to(device)
    
    B, T = actions.shape
    
    # Initial inference
    init_out = model.initial_inference(obs)
    state = init_out['state']
    
    # Auxiliary loss: predict initial board
    aux_loss = compute_weighted_ce_loss(
        init_out['board_logits'],
        board_before,
        empty_weight=cfg.empty_weight,
        nonempty_weight=cfg.nonempty_weight,
    )
    
    # Unroll and accumulate losses
    after_aux_loss = 0.0
    chance_loss = 0.0
    chance_nll_total = 0.0
    chance_entropy_total = 0.0
    
    for t in range(T):
        step_out = model.recurrent_inference(
            state, 
            actions[:, t],
            empty_mask=empty_masks[:, t] if model.cfg.has_chance else None
        )
        
        # Afterstate auxiliary loss (RULE CORE supervision)
        after_aux_loss = after_aux_loss + compute_weighted_ce_loss(
            step_out['afterstate_board_logits'],
            afterstate_boards[:, t],
            empty_weight=cfg.empty_weight,
            nonempty_weight=cfg.nonempty_weight,
            changed_mask=changed_masks[:, t],
            changed_bonus=cfg.changed_cell_bonus,
        )
        
        # Chance loss (STOCHASTIC supervision)
        if model.cfg.has_chance and 'chance_position_logits' in step_out:
            pos_nll, val_nll, total_nll = compute_chance_nll(
                step_out['chance_position_logits'],
                step_out['chance_value_logits'],
                spawn_positions[:, t],
                spawn_values[:, t],
                empty_masks[:, t],
            )
            chance_loss = chance_loss + total_nll.mean()
            chance_nll_total = chance_nll_total + float(total_nll.mean().item())
            chance_entropy_total = chance_entropy_total + float(step_out['chance_entropy'].mean().item())
        
        state = step_out['afterstate']
    
    # Average over time
    after_aux_loss = after_aux_loss / T
    chance_loss = chance_loss / T if T > 0 else torch.tensor(0.0, device=device)
    
    # Total loss
    loss = (
        cfg.w_aux * aux_loss +
        cfg.w_after_aux * after_aux_loss +
        cfg.w_chance * chance_loss
    )
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    
    return {
        'loss': float(loss.item()),
        'aux': float(aux_loss.item()),
        'after_aux': float(after_aux_loss.item()),
        'chance': float(chance_loss.item()) if isinstance(chance_loss, torch.Tensor) else chance_loss,
        'chance_nll': chance_nll_total / T if T > 0 else 0.0,
        'chance_entropy': chance_entropy_total / T if T > 0 else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: CausalWorldModel,
    loader: DataLoader,
    device: str,
    cfg: TrainConfig,
    max_batches: int = 50,
) -> Dict[str, float]:
    """
    Comprehensive evaluation with the RIGHT metrics.
    
    Key metrics:
    - after_cell_acc: Per-cell accuracy (mean over all cells)
    - after_nonempty_acc: Accuracy on non-empty cells (class imbalance check)
    - after_changed_acc: Accuracy on cells that changed (RULE CORE test)
    - after_exact: Fraction of fully correct boards
    - chance_nll: NLL of chance prediction (with teacher mask)
    - chance_entropy: Entropy of chance distribution
    """
    model.eval()
    
    metrics = {
        'after_cell_acc': [],
        'after_nonempty_acc': [],
        'after_changed_acc': [],
        'after_exact': [],
        'chance_nll': [],
        'chance_pos_entropy': [],
        'chance_val_entropy': [],
    }
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        
        obs = batch['obs'].to(device)
        actions = batch['actions'].to(device)
        afterstate_boards = batch['afterstate_boards'].to(device)
        spawn_positions = batch['spawn_positions'].to(device)
        spawn_values = batch['spawn_values'].to(device)
        empty_masks = batch['empty_masks'].to(device)
        changed_masks = batch['changed_masks'].to(device)
        
        B, T = actions.shape
        
        # Initial inference
        init_out = model.initial_inference(obs)
        state = init_out['state']
        
        for t in range(min(T, 3)):  # Only eval first few steps
            step_out = model.recurrent_inference(
                state,
                actions[:, t],
                empty_mask=empty_masks[:, t] if model.cfg.has_chance else None
            )
            
            # Afterstate prediction
            pred_board = step_out['afterstate_board_logits'].argmax(dim=-1)  # (B, cells)
            target_board = afterstate_boards[:, t]  # (B, cells)
            
            # Per-cell accuracy
            correct = (pred_board == target_board)
            metrics['after_cell_acc'].append(float(correct.float().mean().item()))
            
            # Non-empty accuracy
            nonempty = target_board != 0
            if nonempty.any():
                nonempty_acc = correct[nonempty].float().mean()
                metrics['after_nonempty_acc'].append(float(nonempty_acc.item()))
            
            # Changed cell accuracy
            changed = changed_masks[:, t]
            if changed.any():
                changed_acc = correct[changed].float().mean()
                metrics['after_changed_acc'].append(float(changed_acc.item()))
            
            # Exact board accuracy
            exact = correct.all(dim=-1).float()
            metrics['after_exact'].append(float(exact.mean().item()))
            
            # Chance metrics
            if model.cfg.has_chance and 'chance_position_logits' in step_out:
                pos_nll, val_nll, total_nll = compute_chance_nll(
                    step_out['chance_position_logits'],
                    step_out['chance_value_logits'],
                    spawn_positions[:, t],
                    spawn_values[:, t],
                    empty_masks[:, t],
                )
                metrics['chance_nll'].append(float(total_nll.mean().item()))
                
                # Position entropy
                pos_probs = step_out['chance_position_probs']
                pos_entropy = compute_entropy(pos_probs)
                metrics['chance_pos_entropy'].append(float(pos_entropy.mean().item()))
                
                # Value entropy
                val_probs = step_out['chance_value_probs']
                val_entropy = compute_entropy(val_probs)
                metrics['chance_val_entropy'].append(float(val_entropy.mean().item()))
            
            state = step_out['afterstate']
    
    # Average all metrics
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def collect_data(game: str, cfg: TrainConfig) -> List[dict]:
    """Collect random game trajectories."""
    all_transitions = []
    
    print(f"Collecting {cfg.collect_episodes} episodes of {game}...")
    
    for ep in tqdm(range(cfg.collect_episodes)):
        seed = ep * 12345  # Reproducible but different seeds
        
        if game.lower() == "2048":
            transitions = collect_2048_game(
                max_moves=cfg.max_steps_per_episode,
                seed=seed,
                valid_only=cfg.valid_actions_only,
            )
        elif game.lower() == "othello":
            transitions = collect_othello_game(max_moves=cfg.max_steps_per_episode)
        else:
            raise ValueError(f"Unknown game: {game}")
        
        all_transitions.extend(transitions)
    
    print(f"Collected {len(all_transitions)} transitions")
    return all_transitions


def train(cfg: TrainConfig):
    """Main training loop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Create model
    model = create_model(
        cfg.game,
        num_styles=cfg.num_styles,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Collect data
    transitions = collect_data(cfg.game, cfg)
    
    # Create dataset and loader
    num_classes = 17 if cfg.game.lower() == "2048" else 3
    dataset = TransitionDataset(transitions, cfg.unroll_length, num_classes)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    
    # Training loop
    step = 0
    epoch = 0
    
    while step < cfg.train_steps:
        epoch += 1
        
        for batch in loader:
            if step >= cfg.train_steps:
                break
            
            metrics = train_step(model, batch, optimizer, cfg, device)
            step += 1
            
            if step % cfg.log_every == 0:
                print(
                    f"step {step:06d} | "
                    f"loss {metrics['loss']:.4f} | "
                    f"aux {metrics['aux']:.4f} | "
                    f"after_aux {metrics['after_aux']:.4f} | "
                    f"chance {metrics['chance']:.4f} | "
                    f"chance_ent {metrics['chance_entropy']:.3f}"
                )
            
            if step % cfg.eval_every == 0:
                eval_metrics = evaluate(model, loader, device, cfg)
                print(
                    f"[EVAL step {step}] "
                    f"cell_acc={eval_metrics['after_cell_acc']:.3f} | "
                    f"nonempty_acc={eval_metrics['after_nonempty_acc']:.3f} | "
                    f"changed_acc={eval_metrics['after_changed_acc']:.3f} | "
                    f"exact={eval_metrics['after_exact']:.3f} | "
                    f"chance_nll={eval_metrics['chance_nll']:.3f} | "
                    f"chance_ent={eval_metrics['chance_pos_entropy']:.3f}"
                )
            
            if step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"ckpt_step{step}.pt")
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'config': asdict(cfg),
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
    
    # Final save
    final_path = os.path.join(cfg.output_dir, "ckpt_final.pt")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'config': asdict(cfg),
    }, final_path)
    print(f"Training complete. Final model: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Causal World Model")
    
    # Game
    parser.add_argument("--game", type=str, default="2048", choices=["2048", "othello"])
    
    # Data
    parser.add_argument("--collect_episodes", type=int, default=2000)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    parser.add_argument("--valid_actions_only", type=int, default=1)
    
    # Training
    parser.add_argument("--train_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--unroll_length", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    # Loss weights
    parser.add_argument("--w_aux", type=float, default=1.0)
    parser.add_argument("--w_after_aux", type=float, default=2.0)
    parser.add_argument("--w_chance", type=float, default=1.0)
    parser.add_argument("--w_style", type=float, default=0.2)
    parser.add_argument("--w_inv", type=float, default=0.0)
    parser.add_argument("--w_policy", type=float, default=0.0)
    parser.add_argument("--w_value", type=float, default=0.0)
    
    # Class balancing
    parser.add_argument("--empty_weight", type=float, default=0.2)
    parser.add_argument("--changed_cell_bonus", type=float, default=3.0)
    
    # Logging
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="./outputs_stoch_muzero")
    
    args = parser.parse_args()
    
    cfg = TrainConfig(
        game=args.game,
        collect_episodes=args.collect_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        valid_actions_only=bool(args.valid_actions_only),
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        lr=args.lr,
        w_aux=args.w_aux,
        w_after_aux=args.w_after_aux,
        w_chance=args.w_chance,
        w_style=args.w_style,
        w_inv=args.w_inv,
        w_policy=args.w_policy,
        w_value=args.w_value,
        empty_weight=args.empty_weight,
        changed_cell_bonus=args.changed_cell_bonus,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
    )
    
    train(cfg)


if __name__ == "__main__":
    main()
