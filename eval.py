#!/usr/bin/env python3
"""
Evaluation script for Stochastic MuZero with Learned Temporal Abstractions.

Evaluates a trained model on:
- Game performance (score, max tile, game length)
- Macro-operator statistics
- Planning efficiency metrics

Usage:
    python eval.py --checkpoint runs/2048_xxx/best_model.pt
    python eval.py --checkpoint runs/2048_xxx/best_model.pt --num_games 100
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

from utils.config import Config, load_config
from games.game_2048 import Game2048
from networks.muzero_network import MuZeroNetwork
from mcts.tree_search import StochasticMCTS, MCTSConfig
from mcts.macro_cache import MacroCache


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[MuZeroNetwork, Config]:
    """Load model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config from same directory
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        # Use defaults
        config = Config()

    # Create model
    model = MuZeroNetwork(
        observation_dim=config.observation_dim,
        action_space_size=config.action_space_size,
        chance_space_size=config.chance_space_size,
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        support_size=config.support_size,
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def play_game(
    game,
    model: MuZeroNetwork,
    config: Config,
    macro_cache: Optional[MacroCache] = None,
    device: torch.device = torch.device("cpu"),
    temperature: float = 0.0,  # Greedy by default for eval
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Play one game with MCTS and return statistics.

    Args:
        game: Game environment
        model: MuZero network
        config: Configuration
        macro_cache: Optional macro cache
        device: Device for computation
        temperature: Temperature for action selection (0 = greedy)
        verbose: Whether to print game state

    Returns:
        Dictionary of game statistics
    """
    # MCTS config (fewer simulations for faster eval)
    mcts_config = MCTSConfig(
        num_simulations=config.num_simulations,
        discount=config.discount,
        pb_c_base=config.pb_c_base,
        pb_c_init=config.pb_c_init,
        root_dirichlet_alpha=config.root_dirichlet_alpha,
        root_exploration_fraction=0.0,  # No exploration noise in eval
        use_macros=macro_cache is not None,
    )
    mcts = StochasticMCTS(model, mcts_config, macro_cache)

    # Play game
    state = game.reset()
    total_reward = 0.0
    move_count = 0
    entropy_sum = 0.0
    macro_uses = 0

    if verbose:
        print("Initial state:")
        print(game.render(state))

    while not game.is_terminal(state) and move_count < config.max_moves:
        # Get observation
        observation = game.encode_state(state).to(device)

        # Get legal actions
        legal_actions = game.legal_actions(state)
        if not legal_actions:
            break

        # Run MCTS
        root = mcts.search(
            observation.unsqueeze(0),
            legal_actions,
            add_exploration_noise=False,
        )

        # Select action (greedy or with temperature)
        action, _ = mcts.get_action_policy(root, temperature)

        # Execute
        result = game.step(state, action)

        # Track entropy
        with torch.no_grad():
            dynamics_out = model.recurrent_inference(
                root.hidden_state,
                torch.tensor([action], device=device),
            )
            entropy_sum += dynamics_out.chance_entropy.item()

        total_reward += result.reward
        move_count += 1
        state = result.next_state

        if verbose and move_count % 50 == 0:
            print(f"\nMove {move_count}:")
            print(game.render(state))

    # Final statistics
    stats = {
        "total_reward": total_reward,
        "game_length": move_count,
        "avg_entropy": entropy_sum / max(move_count, 1),
    }

    # Game-specific stats
    if hasattr(game, "get_max_tile"):
        stats["max_tile"] = game.get_max_tile(state)

    if verbose:
        print(f"\nFinal state:")
        print(game.render(state))
        print(f"Total reward: {total_reward}")
        print(f"Game length: {move_count}")
        if "max_tile" in stats:
            print(f"Max tile: {stats['max_tile']}")

    return stats


def evaluate(
    model: MuZeroNetwork,
    config: Config,
    num_games: int = 100,
    device: torch.device = torch.device("cpu"),
    temperature: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate model over multiple games.

    Args:
        model: MuZero network
        config: Configuration
        num_games: Number of games to play
        device: Device for computation
        temperature: Temperature for action selection

    Returns:
        Aggregated statistics
    """
    game = Game2048()

    # Macro cache for tracking
    macro_cache = MacroCache(
        state_dim=config.state_dim,
        entropy_threshold=config.entropy_threshold,
    )

    all_stats: List[Dict[str, float]] = []

    print(f"Evaluating on {num_games} games...")
    for i in tqdm(range(num_games)):
        stats = play_game(
            game=game,
            model=model,
            config=config,
            macro_cache=macro_cache,
            device=device,
            temperature=temperature,
        )
        all_stats.append(stats)

    # Aggregate statistics
    result = {}
    for key in all_stats[0].keys():
        values = [s[key] for s in all_stats]
        result[f"mean_{key}"] = np.mean(values)
        result[f"std_{key}"] = np.std(values)
        result[f"min_{key}"] = np.min(values)
        result[f"max_{key}"] = np.max(values)

    # Tile distribution (for 2048)
    if "max_tile" in all_stats[0]:
        tiles = [s["max_tile"] for s in all_stats]
        for threshold in [128, 256, 512, 1024, 2048, 4096, 8192]:
            result[f"tile_{threshold}_rate"] = sum(1 for t in tiles if t >= threshold) / len(tiles)

    # Macro statistics
    macro_stats = macro_cache.get_statistics()
    result.update({f"macro_{k}": v for k, v in macro_stats.items()})

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stochastic MuZero model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Number of games to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for action selection (0 = greedy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print game states during play",
    )
    parser.add_argument(
        "--single_game",
        action="store_true",
        help="Play and display a single game",
    )

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)

    if args.single_game:
        # Play single game with verbose output
        game = Game2048()
        stats = play_game(
            game=game,
            model=model,
            config=config,
            device=device,
            temperature=args.temperature,
            verbose=True,
        )
    else:
        # Full evaluation
        results = evaluate(
            model=model,
            config=config,
            num_games=args.num_games,
            device=device,
            temperature=args.temperature,
        )

        # Print results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)

        print("\nGame Performance:")
        print(f"  Mean reward: {results['mean_total_reward']:.1f} (+/- {results['std_total_reward']:.1f})")
        print(f"  Mean game length: {results['mean_game_length']:.1f}")

        if "mean_max_tile" in results:
            print(f"  Mean max tile: {results['mean_max_tile']:.0f}")
            print(f"  Max max tile: {results['max_max_tile']:.0f}")

            print("\nTile Achievement Rates:")
            for threshold in [128, 256, 512, 1024, 2048, 4096, 8192]:
                key = f"tile_{threshold}_rate"
                if key in results:
                    print(f"  {threshold}+: {results[key]:.1%}")

        print("\nPlanning Statistics:")
        print(f"  Mean entropy: {results['mean_avg_entropy']:.4f}")

        print("\nMacro Statistics:")
        print(f"  Macros discovered: {results.get('macro_num_macros', 0)}")
        print(f"  Total macro uses: {results.get('macro_total_uses', 0)}")
        print(f"  Macro success rate: {results.get('macro_success_rate', 0):.2%}")


if __name__ == "__main__":
    main()
