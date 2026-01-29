"""
Evaluation script for the Causal World Model.

This implements the RIGHT metrics to assess rule learning:

1. Board Prediction Accuracy:
   - Per-cell accuracy (not just exact board)
   - Non-empty cell accuracy (class imbalance check)
   - Changed cell accuracy (RULE CORE verification)
   - Exact board accuracy (strict)

2. Chance Prediction Quality:
   - NLL with teacher mask (measures prediction quality independently)
   - Entropy vs oracle entropy (should match for well-trained model)
   - Entropy histogram (bimodal = good separation)

3. Rollout Quality:
   - Multi-step prediction error
   - Error accumulation rate
   - Visual rollout comparison

4. Macro Discovery Metrics:
   - Compressibility rate
   - Effective planning depth
   - Macro reuse rate
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import CausalWorldModel, create_model
from .games import Game2048Env, OthelloEnv, collect_2048_game, collect_othello_game
from .games.game2048 import compute_oracle_chance_entropy
from .macro_cache import MacroCache, analyze_entropy_distribution


def load_model(ckpt_path: str, device: str = "cpu") -> Tuple[CausalWorldModel, dict]:
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('config', {})
    
    game = cfg.get('game', '2048')
    model = create_model(game).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    return model, cfg


@torch.no_grad()
def evaluate_board_prediction(
    model: CausalWorldModel,
    transitions: List[dict],
    device: str = "cpu",
    max_samples: int = 1000,
) -> Dict[str, float]:
    """
    Evaluate board prediction accuracy with proper metrics.
    
    Returns:
        - cell_acc: Per-cell accuracy
        - nonempty_acc: Accuracy on non-empty cells
        - changed_acc: Accuracy on cells that changed
        - exact_acc: Fraction of fully correct boards
    """
    model.eval()
    
    cell_correct = []
    nonempty_correct = []
    changed_correct = []
    exact_correct = []
    
    for t in tqdm(transitions[:max_samples], desc="Eval board"):
        obs = torch.from_numpy(t['obs']).float().unsqueeze(0).to(device)
        action = torch.tensor([t['action']], device=device)
        afterstate_target = torch.from_numpy(t['afterstate_board']).long().to(device)
        board_before = torch.from_numpy(t['board_before']).long().to(device)
        
        # Forward pass
        init = model.initial_inference(obs)
        out = model.recurrent_inference(init['state'], action)
        
        # Predict board
        pred_board = out['afterstate_board_logits'].argmax(dim=-1).squeeze(0)  # (cells,)
        
        # Metrics
        correct = (pred_board == afterstate_target)
        cell_correct.append(float(correct.float().mean().item()))
        
        # Non-empty
        nonempty = afterstate_target != 0
        if nonempty.any():
            nonempty_correct.append(float(correct[nonempty].float().mean().item()))
        
        # Changed
        changed = afterstate_target != board_before
        if changed.any():
            changed_correct.append(float(correct[changed].float().mean().item()))
        
        # Exact
        exact_correct.append(float(correct.all().float().item()))
    
    return {
        'cell_acc': float(np.mean(cell_correct)),
        'nonempty_acc': float(np.mean(nonempty_correct)) if nonempty_correct else 0.0,
        'changed_acc': float(np.mean(changed_correct)) if changed_correct else 0.0,
        'exact_acc': float(np.mean(exact_correct)),
    }


@torch.no_grad()
def evaluate_chance_prediction(
    model: CausalWorldModel,
    transitions: List[dict],
    device: str = "cpu",
    max_samples: int = 1000,
) -> Dict[str, float]:
    """
    Evaluate chance prediction with proper metrics.
    
    Key insight: Compare predicted entropy to ORACLE entropy.
    A perfect model should match: entropy ≈ log(#empties) + H({0.9, 0.1})
    
    Returns:
        - nll: Negative log-likelihood of true outcome
        - pos_nll: NLL of position prediction
        - val_nll: NLL of value prediction
        - entropy: Predicted entropy
        - oracle_entropy: True entropy
        - entropy_error: |predicted - oracle|
    """
    if not model.cfg.has_chance:
        return {'no_chance': True}
    
    model.eval()
    
    pos_nlls = []
    val_nlls = []
    pred_entropies = []
    oracle_entropies = []
    
    for t in tqdm(transitions[:max_samples], desc="Eval chance"):
        obs = torch.from_numpy(t['obs']).float().unsqueeze(0).to(device)
        action = torch.tensor([t['action']], device=device)
        empty_mask = torch.from_numpy(t['empty_mask']).bool().unsqueeze(0).to(device)
        spawn_pos = t.get('spawn_position', -1)
        spawn_val = t.get('spawn_value', 0)
        afterstate_board = t.get('afterstate_board', t['board_after'])
        
        # Forward pass
        init = model.initial_inference(obs)
        out = model.recurrent_inference(init['state'], action, empty_mask)
        
        # Position NLL
        pos_logits = out['chance_position_logits']
        masked_logits = pos_logits.masked_fill(~empty_mask, float('-inf'))
        log_probs_pos = F.log_softmax(masked_logits, dim=-1)
        
        if spawn_pos >= 0 and spawn_pos < pos_logits.shape[-1]:
            pos_nll = -log_probs_pos[0, spawn_pos].item()
            pos_nlls.append(pos_nll)
        
        # Value NLL  
        val_logits = out['chance_value_logits']
        log_probs_val = F.log_softmax(val_logits, dim=-1)
        if spawn_val in [1, 2]:
            val_idx = spawn_val - 1
            val_nll = -log_probs_val[0, val_idx].item()
            val_nlls.append(val_nll)
        
        # Predicted entropy
        pred_entropy = out['chance_entropy'][0].item()
        pred_entropies.append(pred_entropy)
        
        # Oracle entropy
        oracle_ent = compute_oracle_chance_entropy(
            np.array(afterstate_board).reshape(4, 4)
        )
        oracle_entropies.append(oracle_ent)
    
    return {
        'pos_nll': float(np.mean(pos_nlls)) if pos_nlls else 0.0,
        'val_nll': float(np.mean(val_nlls)) if val_nlls else 0.0,
        'total_nll': float(np.mean(pos_nlls)) + float(np.mean(val_nlls)) if pos_nlls else 0.0,
        'pred_entropy': float(np.mean(pred_entropies)),
        'oracle_entropy': float(np.mean(oracle_entropies)),
        'entropy_error': float(np.mean(np.abs(
            np.array(pred_entropies) - np.array(oracle_entropies)
        ))),
    }


@torch.no_grad()
def evaluate_rollout(
    model: CausalWorldModel,
    env,
    num_episodes: int = 50,
    max_steps: int = 200,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate multi-step rollout quality.
    
    Compares model imagination to actual game execution.
    """
    model.eval()
    
    step_errors = []  # Error at each step distance
    episode_scores = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        
        # Get initial state
        init = model.initial_inference(obs_t)
        model_state = init['state']
        
        episode_score = 0
        ep_errors = []
        
        for step in range(max_steps):
            # Get valid actions
            valid_actions = env.get_valid_actions() if hasattr(env, 'get_valid_actions') else list(range(env.num_actions))
            if len(valid_actions) == 0:
                break
            
            # Random action
            action = int(np.random.choice(valid_actions))
            action_t = torch.tensor([action], device=device)
            
            # Model prediction
            if model.cfg.has_chance:
                afterstate_t, afterstate_board, _ = env.get_afterstate(action)
                empty_mask = torch.from_numpy(
                    (afterstate_board.flatten() == 0)
                ).bool().unsqueeze(0).to(device)
                model_out = model.recurrent_inference(model_state, action_t, empty_mask)
            else:
                model_out = model.recurrent_inference(model_state, action_t)
            
            # Actual step
            next_obs, reward, done, _, info = env.step(action)
            episode_score += reward
            
            # Compare model prediction to actual
            actual_board = env.get_board_target()
            pred_board = model_out['afterstate_board_logits'].argmax(dim=-1).squeeze().cpu().numpy()
            
            error = float(np.mean(pred_board != actual_board))
            ep_errors.append(error)
            
            # Update model state (continue rollout in model space)
            model_state = model_out['afterstate']
            
            if done:
                break
        
        episode_scores.append(episode_score)
        step_errors.extend(ep_errors)
    
    return {
        'avg_score': float(np.mean(episode_scores)),
        'std_score': float(np.std(episode_scores)),
        'avg_step_error': float(np.mean(step_errors)),
        'error_at_step_1': float(np.mean(step_errors[:len(episode_scores)])) if step_errors else 0.0,
    }


def render_rollout_comparison(
    model: CausalWorldModel,
    env,
    num_steps: int = 16,
    device: str = "cpu",
) -> str:
    """
    Create ASCII visualization comparing model prediction vs actual.
    """
    model.eval()
    
    obs, _ = env.reset(seed=42)
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    
    init = model.initial_inference(obs_t)
    model_state = init['state']
    
    lines = ["=" * 60]
    lines.append("ROLLOUT COMPARISON: Model (top) vs Actual (bottom)")
    lines.append("=" * 60)
    
    for step in range(num_steps):
        valid_actions = env.get_valid_actions() if hasattr(env, 'get_valid_actions') else list(range(env.num_actions))
        if len(valid_actions) == 0:
            break
        
        action = int(np.random.choice(valid_actions))
        action_t = torch.tensor([action], device=device)
        
        # Model prediction
        with torch.no_grad():
            model_out = model.recurrent_inference(model_state, action_t)
            pred_board = model_out['afterstate_board_logits'].argmax(dim=-1).squeeze().cpu().numpy()
        
        # Actual
        next_obs, reward, done, _, _ = env.step(action)
        actual_board = env.get_board_target()
        
        # Render
        lines.append(f"\nStep {step + 1}: Action {action}")
        lines.append("Model prediction:")
        lines.append(_render_board(pred_board, model.cfg.board_size))
        lines.append("Actual:")
        lines.append(_render_board(actual_board, model.cfg.board_size))
        
        match = np.array_equal(pred_board, actual_board)
        lines.append(f"Match: {'✓' if match else '✗'}")
        
        model_state = model_out['afterstate']
        
        if done:
            break
    
    return "\n".join(lines)


def _render_board(board: np.ndarray, size: int) -> str:
    """Render a flat board array as ASCII."""
    board = board.reshape(size, size)
    lines = []
    for row in board:
        cells = []
        for val in row:
            if val == 0:
                cells.append("  .")
            else:
                cells.append(f"{2**val:3d}" if val > 0 else f"{val:3d}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def full_evaluation(ckpt_path: str, output_dir: str = None, device: str = "cpu"):
    """Run complete evaluation suite."""
    print(f"Loading model from {ckpt_path}")
    model, cfg = load_model(ckpt_path, device)
    game = cfg.get('game', '2048')
    
    print(f"Evaluating {game} model")
    
    # Collect fresh test data
    print("Collecting test data...")
    if game.lower() == "2048":
        transitions = collect_2048_game(max_moves=500, seed=99999, valid_only=True)
        env = Game2048Env(seed=12345)
    else:
        transitions = collect_othello_game(max_moves=100)
        env = OthelloEnv()
    
    print(f"Collected {len(transitions)} test transitions")
    
    results = {}
    
    # Board prediction
    print("\n=== Board Prediction Metrics ===")
    board_metrics = evaluate_board_prediction(model, transitions, device)
    results['board'] = board_metrics
    for k, v in board_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Chance prediction (if applicable)
    if model.cfg.has_chance:
        print("\n=== Chance Prediction Metrics ===")
        chance_metrics = evaluate_chance_prediction(model, transitions, device)
        results['chance'] = chance_metrics
        for k, v in chance_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # Entropy distribution
    print("\n=== Entropy Distribution ===")
    entropy_stats = analyze_entropy_distribution(model, transitions, device)
    results['entropy'] = entropy_stats
    for k, v in entropy_stats.items():
        print(f"  {k}: {v:.4f}")
    
    # Rollout evaluation
    print("\n=== Rollout Quality ===")
    rollout_metrics = evaluate_rollout(model, env, num_episodes=20, device=device)
    results['rollout'] = rollout_metrics
    for k, v in rollout_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Visual rollout
    print("\n=== Sample Rollout ===")
    rollout_vis = render_rollout_comparison(model, env, num_steps=8, device=device)
    print(rollout_vis)
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "eval_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        rollout_path = os.path.join(output_dir, "sample_rollout.txt")
        with open(rollout_path, 'w') as f:
            f.write(rollout_vis)
        print(f"Rollout saved to {rollout_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Causal World Model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    full_evaluation(args.ckpt, args.output_dir, args.device)


if __name__ == "__main__":
    main()
