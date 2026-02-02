#!/usr/bin/env python3
"""
Training script for Stochastic MuZero with Learned Temporal Abstractions.

This trains a MuZero agent on 2048 (or other games) with:
- Afterstate/chance separation for stochastic transitions
- Macro-operator discovery for compressible dynamics
- Hierarchical planning with learned abstractions

Usage:
    python train.py --config config/game_2048.yaml
    python train.py --game 2048 --num_episodes 1000
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import torch
from tqdm import tqdm

from utils.config import Config, load_config, save_config
from games.game_2048 import Game2048
from networks.muzero_network import MuZeroNetwork
from mcts.macro_cache import MacroCache
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayConfig, run_self_play
from training.trainer import Trainer, TrainerConfig, train_epoch


def create_game(config: Config):
    """Create game environment based on config."""
    if config.game.lower() == "2048":
        return Game2048()
    else:
        raise ValueError(f"Unknown game: {config.game}")


def create_model(config: Config, device: torch.device) -> MuZeroNetwork:
    """Create MuZero network based on config."""
    return MuZeroNetwork(
        observation_dim=config.observation_dim,
        action_space_size=config.action_space_size,
        chance_space_size=config.chance_space_size,
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        support_size=config.support_size,
    ).to(device)


def train(
    config: Config,
    checkpoint_dir: str,
    num_iterations: int = 1000,
    games_per_iteration: int = 10,
    batches_per_iteration: int = 100,
    save_interval: int = 100,
    log_interval: int = 10,
    resume_from: Optional[str] = None,
):
    """
    Main training loop.

    Args:
        config: Training configuration
        checkpoint_dir: Directory for saving checkpoints
        num_iterations: Total number of training iterations
        games_per_iteration: Number of self-play games per iteration
        batches_per_iteration: Number of training batches per iteration
        save_interval: Save checkpoint every N iterations
        log_interval: Print statistics every N iterations
        resume_from: Path to checkpoint to resume from
    """
    # Setup
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config
    save_config(config, os.path.join(checkpoint_dir, "config.yaml"))

    # Create components
    game = create_game(config)
    model = create_model(config, device)

    # Macro cache for temporal abstractions
    macro_cache = MacroCache(
        state_dim=config.state_dim,
        entropy_threshold=config.entropy_threshold,
        composition_threshold=config.composition_threshold,
        min_macro_length=config.min_macro_length,
        max_macro_length=config.max_macro_length,
        max_macros=config.max_macros,
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config.replay_buffer_size,
        batch_size=config.batch_size,
        num_unroll_steps=config.num_unroll_steps,
        td_steps=config.td_steps,
        discount=config.discount,
        priority_alpha=config.priority_alpha,
        priority_beta=config.priority_beta,
    )

    # Trainer
    trainer_config = TrainerConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        support_size=config.support_size,
    )
    trainer = Trainer(model, trainer_config, device)

    # Self-play config
    self_play_config = SelfPlayConfig(
        num_simulations=config.num_simulations,
        max_moves=config.max_moves,
        temperature_init=config.temperature_init,
        temperature_final=config.temperature_final,
        discount=config.discount,
        root_dirichlet_alpha=config.root_dirichlet_alpha,
        root_exploration_fraction=config.root_exploration_fraction,
        pb_c_base=config.pb_c_base,
        pb_c_init=config.pb_c_init,
        min_macro_length=config.min_macro_length,
        max_macro_length=config.max_macro_length,
    )

    # Resume from checkpoint
    start_iteration = 0
    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        trainer.load_checkpoint(resume_from)
        start_iteration = trainer.training_step // batches_per_iteration

    # Training statistics
    best_avg_reward = float("-inf")

    print(f"\nStarting training for {num_iterations} iterations")
    print(f"  Games per iteration: {games_per_iteration}")
    print(f"  Batches per iteration: {batches_per_iteration}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print()

    for iteration in range(start_iteration, num_iterations):
        # Self-play phase
        model.eval()
        histories, self_play_stats = run_self_play(
            game=game,
            model=model,
            num_games=games_per_iteration,
            config=self_play_config,
            macro_cache=macro_cache,
            device=device,
            training_step=trainer.training_step,
        )

        # Add games to replay buffer
        for history in histories:
            replay_buffer.save_game(history)

        # Training phase (only if we have enough data)
        if len(replay_buffer) >= config.batch_size:
            epoch_losses = train_epoch(
                trainer=trainer,
                replay_buffer=replay_buffer,
                num_batches=batches_per_iteration,
                device=device,
            )
        else:
            epoch_losses = {"total": 0.0, "policy": 0.0, "value": 0.0, "reward": 0.0, "chance": 0.0}

        # Logging
        if (iteration + 1) % log_interval == 0:
            buffer_stats = replay_buffer.get_statistics()
            macro_stats = macro_cache.get_statistics()
            trainer_stats = trainer.get_statistics()

            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            print(f"Self-play:")
            print(f"  Avg reward: {self_play_stats.get('avg_total_reward', 0):.1f}")
            print(f"  Avg game length: {self_play_stats.get('avg_game_length', 0):.1f}")
            print(f"  Avg max tile: {self_play_stats.get('avg_max_tile', 0):.0f}")
            print(f"  Max max tile: {self_play_stats.get('max_max_tile', 0):.0f}")
            print(f"  Avg entropy: {self_play_stats.get('avg_avg_entropy', 0):.3f}")

            print(f"Training:")
            print(f"  Total loss: {epoch_losses['total']:.4f}")
            print(f"  Policy loss: {epoch_losses['policy']:.4f}")
            print(f"  Value loss: {epoch_losses['value']:.4f}")
            print(f"  Reward loss: {epoch_losses['reward']:.4f}")
            print(f"  Chance loss: {epoch_losses['chance']:.4f}")
            print(f"  Learning rate: {trainer_stats.get('learning_rate', 0):.6f}")

            print(f"Buffer:")
            print(f"  Games: {buffer_stats['num_games']}")
            print(f"  Positions: {buffer_stats['total_positions']}")

            print(f"Macros:")
            print(f"  Discovered: {macro_stats['num_macros']}")
            print(f"  Total uses: {macro_stats['total_uses']}")
            print(f"  Success rate: {macro_stats['success_rate']:.2%}")
            print(f"  Avg confidence: {macro_stats['avg_confidence']:.3f}")

        # Save checkpoint
        if (iteration + 1) % save_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_{iteration + 1}.pt"
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Save best model
            avg_reward = self_play_stats.get("avg_total_reward", 0)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                trainer.save_checkpoint(best_path)
                print(f"New best model saved: {best_path} (reward: {avg_reward:.1f})")

    # Final save
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete. Final model saved: {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Stochastic MuZero with Learned Temporal Abstractions"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="2048",
        help="Game to train on (default: 2048)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--games_per_iter",
        type=int,
        default=10,
        help="Self-play games per iteration",
    )
    parser.add_argument(
        "--batches_per_iter",
        type=int,
        default=100,
        help="Training batches per iteration",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory for checkpoints (default: runs/<timestamp>)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config, game=args.game, device=args.device)

    # Set checkpoint directory
    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f"runs/{config.game}_{timestamp}"
    else:
        checkpoint_dir = args.checkpoint_dir

    # Train
    train(
        config=config,
        checkpoint_dir=checkpoint_dir,
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        batches_per_iteration=args.batches_per_iter,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
