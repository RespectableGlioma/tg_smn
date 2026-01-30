"""
Train VQ-VAE World Model v3 - Fixed training approach.

Key fixes over v2:
1. Warmup phase (5000 steps) - Train reconstruction ONLY first, let codebook stabilize
2. No dead code reset - Once codes are stable, dynamics can learn
3. Track accuracy - Shows if model actually predicts transitions correctly
4. Better evaluation - Separate accuracy for changed vs unchanged positions

Root cause of v2 failure:
- 51,200 code resets over 20,000 steps = 2.5 codes reset per step
- Transition loss never converged (2.0+ for 2048, 4.7+ for Othello)
- Entropy ratio = 1.0x for both games (should be ~14x for 2048, ~0 for Othello)
- The aggressive dead code reset prevented dynamics from learning stable code->code mappings

Usage:
    python -m world_models.stoch_muzero.train_vq_v3 --game 2048 --train_steps 20000
    python -m world_models.stoch_muzero.train_vq_v3 --game othello --train_steps 20000

Colab usage:
    from world_models.stoch_muzero.train_vq_v3 import train
    model, analysis = train(game='2048', train_steps=20000)
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
                if game.done:
                    break

            obs_traj.append(renderer.render_2048(game.board))
            act_traj.append(action)

            if len(act_traj) >= max_steps:
                break

        if len(act_traj) >= 2:
            all_obs.append(np.stack(obs_traj[:len(act_traj)+1]))
            all_actions.append(np.array(act_traj))

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


def compute_transition_accuracy(
    model: VQWorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    n_samples: int = 1000,
) -> Dict[str, float]:
    """
    Compute transition accuracy metrics.

    Returns:
        overall_acc: Top-1 accuracy across all positions
        changed_acc: Accuracy for positions that changed
        unchanged_acc: Accuracy for positions that stayed same
        n_changed: Number of changed positions
        n_unchanged: Number of unchanged positions
    """
    model.eval()

    correct_total = 0
    total_positions = 0
    correct_changed = 0
    total_changed = 0
    correct_unchanged = 0
    total_unchanged = 0

    with torch.no_grad():
        for i in range(min(n_samples, obs.shape[0])):
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

                # Get predictions (argmax)
                pred_indices = step_result['logits'].argmax(dim=-1)  # [1, N]
                target_indices = all_indices[:, t+1]  # [1, N]

                # Overall accuracy
                correct = (pred_indices == target_indices)
                correct_total += correct.sum().item()
                total_positions += correct.numel()

                # Split by changed vs unchanged
                prev_indices = all_indices[:, t]
                changed_mask = (prev_indices != target_indices)
                unchanged_mask = ~changed_mask

                correct_changed += (correct & changed_mask).sum().item()
                total_changed += changed_mask.sum().item()

                correct_unchanged += (correct & unchanged_mask).sum().item()
                total_unchanged += unchanged_mask.sum().item()

    model.train()

    return {
        'overall_acc': correct_total / max(total_positions, 1),
        'changed_acc': correct_changed / max(total_changed, 1),
        'unchanged_acc': correct_unchanged / max(total_unchanged, 1),
        'n_changed': total_changed,
        'n_unchanged': total_unchanged,
        'pct_changed': total_changed / max(total_positions, 1),
    }


def analyze_entropy(
    model: VQWorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Analyze entropy distribution with detailed per-position stats.
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

            B, Tp1, C, H, W = obs_seq.shape
            obs_flat = obs_seq.reshape(B * Tp1, C, H, W)
            enc = model.encode(obs_flat, training=False)
            all_indices = enc['indices'].reshape(B, Tp1, -1)

            for t in range(T):
                z_q = model.quantizer.embedding(all_indices[:, t])
                step_result = model.step(z_q, act_seq[:, t], sample=False)

                logits = step_result['logits']
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                ent_per_pos = -(probs * log_probs).sum(dim=-1) / np.log(2)
                ent_per_pos = ent_per_pos[0].cpu().numpy()

                idx_before = all_indices[0, t].cpu().numpy()
                idx_after = all_indices[0, t+1].cpu().numpy()
                changed = (idx_before != idx_after)

                all_entropy.append(ent_per_pos)
                all_changed.append(changed)
                all_indices_before.append(idx_before)
                all_indices_after.append(idx_after)

    model.train()

    return {
        'entropy_per_position': np.stack(all_entropy),
        'positions_changed': np.stack(all_changed),
        'indices_before': np.stack(all_indices_before),
        'indices_after': np.stack(all_indices_after),
    }


def train(
    game: str = '2048',
    train_steps: int = 20000,
    warmup_steps: int = 5000,
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
    """
    Train VQ-VAE world model v3 with warmup phase.

    Key changes from v2:
    1. Warmup phase trains reconstruction only (no dynamics loss)
    2. No dead code reset - let codebook stabilize naturally
    3. Track transition accuracy in addition to loss
    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"VQ-VAE World Model v3 Training")
    print(f"Game: {game} | Steps: {train_steps} | Warmup: {warmup_steps}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    print("Key changes in v3:")
    print("  - Warmup phase: reconstruction only for first {warmup_steps} steps")
    print("  - NO dead code reset (let codebook stabilize)")
    print("  - Track transition accuracy, not just loss")
    print()

    # Generate data
    print("Generating trajectories...")
    n_actions = 4 if game.lower() == '2048' else 64
    obs, actions = generate_trajectories(game, n_trajectories, max_traj_len, img_size)

    obs_t = torch.from_numpy(obs)
    actions_t = torch.from_numpy(actions)

    print(f"Data: {obs.shape[0]} trajectories, {obs.shape[1]-1} steps each")
    print(f"Obs shape: {obs.shape}, Actions shape: {actions.shape}\n")

    # Create model - NO dead code reset
    cfg = VQWorldModelConfig(
        img_size=img_size,
        n_actions=n_actions,
        codebook_size=codebook_size,
        code_dim=code_dim,
        ema_decay=0.99,           # Standard EMA (was 0.95 in v2)
        dead_code_threshold=0,    # Effectively disabled
        reset_every=999999,       # Never reset
    )

    model = VQWorldModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Training loop
    model.train()
    pbar = tqdm(range(1, train_steps + 1))

    unroll_steps = 3

    # Track metrics
    recon_history = []
    trans_history = []
    acc_history = []

    for step in pbar:
        # Determine training phase
        in_warmup = step <= warmup_steps

        idx = torch.randint(0, obs_t.shape[0], (batch_size,))
        n_timesteps_needed = unroll_steps + 1
        obs_batch = obs_t[idx, :n_timesteps_needed].to(device)
        action_batch = actions_t[idx, :unroll_steps].to(device)

        # During warmup: only train reconstruction (w_transition=0)
        # After warmup: train both
        w_transition = 0.0 if in_warmup else 1.0

        losses = model.compute_loss(
            obs_batch,
            action_batch,
            unroll_steps=unroll_steps,
            w_transition=w_transition,
            reset_dead_codes=False,  # Never reset codes
        )

        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        recon_history.append(float(losses['recon_loss']))
        trans_history.append(float(losses['transition_loss']))

        phase = "WARMUP" if in_warmup else "FULL"

        if step % 100 == 0:
            pbar.set_postfix({
                'phase': phase,
                'recon': f"{losses['recon_loss'].item():.4f}",
                'trans': f"{losses['transition_loss'].item():.4f}",
                'codes': f"{int(losses['unique_codes'].item())}/{codebook_size}",
            })

        # Periodic detailed stats
        if step % 2000 == 0 or step == warmup_steps:
            stats = model.get_codebook_stats()
            acc = compute_transition_accuracy(model, obs_t, actions_t, device, n_samples=200)
            acc_history.append(acc)

            print(f"\n[Step {step}] Phase: {phase}")
            print(f"  Codebook: {stats['active_codes']}/{codebook_size} ({100*stats['active_codes']/codebook_size:.1f}%)")
            print(f"  Transition accuracy: {100*acc['overall_acc']:.1f}%")
            print(f"    Changed positions:   {100*acc['changed_acc']:.1f}% (n={acc['n_changed']})")
            print(f"    Unchanged positions: {100*acc['unchanged_acc']:.1f}% (n={acc['n_unchanged']})")

            if step == warmup_steps:
                print(f"\n{'='*40}")
                print("WARMUP COMPLETE - Enabling dynamics training")
                print(f"{'='*40}\n")

            torch.cuda.empty_cache()

    # Final analysis
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - Final Analysis")
    print(f"{'='*60}\n")

    # Codebook stats
    stats = model.get_codebook_stats()
    print(f"Final codebook utilization:")
    print(f"  Active codes: {stats['active_codes']}/{codebook_size} ({100*stats['active_codes']/codebook_size:.1f}%)")

    # Final accuracy
    final_acc = compute_transition_accuracy(model, obs_t, actions_t, device, n_samples=1000)
    print(f"\nFinal transition accuracy:")
    print(f"  Overall:   {100*final_acc['overall_acc']:.1f}%")
    print(f"  Changed:   {100*final_acc['changed_acc']:.1f}% ({final_acc['n_changed']} positions)")
    print(f"  Unchanged: {100*final_acc['unchanged_acc']:.1f}% ({final_acc['n_unchanged']} positions)")
    print(f"  % Changed: {100*final_acc['pct_changed']:.1f}%")

    # Entropy analysis
    print("\nAnalyzing entropy distribution...")
    analysis = analyze_entropy(model, obs_t, actions_t, device)

    entropy_all = analysis['entropy_per_position']
    changed = analysis['positions_changed']

    print(f"\nOverall entropy (per-position):")
    print(f"  Mean: {entropy_all.mean():.4f} bits")
    print(f"  Std:  {entropy_all.std():.4f}")
    print(f"  Max:  {entropy_all.max():.4f}")

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

        # Success criteria
        if game.lower() == '2048':
            expected_ratio = 10  # Should be high (stochastic spawns)
            if ratio > expected_ratio:
                print(f"  SUCCESS: Ratio > {expected_ratio}x indicates stochasticity learned!")
            else:
                print(f"  NEEDS WORK: Expected ratio > {expected_ratio}x for 2048")
        else:  # Othello
            expected_ratio = 2  # Should be low (deterministic)
            if ratio < expected_ratio:
                print(f"  SUCCESS: Low ratio indicates deterministic game learned!")
            else:
                print(f"  NEEDS WORK: Expected ratio < {expected_ratio}x for Othello")

    # Entropy distribution
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

    # Add accuracy history to analysis
    analysis['accuracy_history'] = acc_history
    analysis['final_accuracy'] = final_acc

    return model, analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='2048', choices=['2048', 'othello'])
    parser.add_argument('--train_steps', type=int, default=20000)
    parser.add_argument('--warmup_steps', type=int, default=5000)
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
        warmup_steps=args.warmup_steps,
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
