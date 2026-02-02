"""Self-play for generating training data.

Self-play uses MCTS to play games and collect trajectories
for training. Also performs macro discovery on completed games.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

from ..games.base import Game
from ..networks.muzero_network import MuZeroNetwork
from ..mcts.tree_search import StochasticMCTS, MCTSConfig
from ..mcts.macro_cache import MacroCache, discover_macros_from_trajectory
from .replay_buffer import GameHistory


@dataclass
class SelfPlayConfig:
    """Configuration for self-play."""

    num_simulations: int = 50
    max_moves: int = 10000
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_threshold_step: int = 30  # After this step, use final temp
    discount: float = 0.997

    # Exploration
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25

    # MCTS
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25

    # Macro discovery
    discover_macros: bool = True
    min_macro_length: int = 2
    max_macro_length: int = 8


def get_temperature(step: int, config: SelfPlayConfig) -> float:
    """Get temperature for action selection based on game step."""
    if step < config.temperature_threshold_step:
        return config.temperature_init
    return config.temperature_final


def self_play_game(
    game: Game,
    model: MuZeroNetwork,
    config: SelfPlayConfig,
    macro_cache: Optional[MacroCache] = None,
    device: torch.device = torch.device("cpu"),
    training_step: int = 0,
) -> Tuple[GameHistory, Dict[str, float]]:
    """
    Play one game using MCTS and collect trajectory.

    Args:
        game: Game environment
        model: MuZero network for inference
        config: Self-play configuration
        macro_cache: Optional macro cache for discovery/usage
        device: Device for tensor operations
        training_step: Current training step (for macro timestamps)

    Returns:
        (game_history, statistics)
    """
    model.eval()

    # Initialize MCTS
    mcts_config = MCTSConfig(
        num_simulations=config.num_simulations,
        discount=config.discount,
        pb_c_base=config.pb_c_base,
        pb_c_init=config.pb_c_init,
        root_dirichlet_alpha=config.root_dirichlet_alpha,
        root_exploration_fraction=config.root_exploration_fraction,
        use_macros=macro_cache is not None,
    )
    mcts = StochasticMCTS(model, mcts_config, macro_cache)

    # Initialize game
    state = game.reset()
    history = GameHistory()

    # Statistics
    macro_uses = 0
    total_entropy = 0.0
    step_count = 0

    # Play game
    for step in range(config.max_moves):
        if game.is_terminal(state):
            break

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
            add_exploration_noise=True,
        )

        # Get temperature and select action
        temperature = get_temperature(step, config)
        action, policy = mcts.get_action_policy(root, temperature)

        # Execute action
        result = game.step(state, action)

        # Get entropy from model
        with torch.no_grad():
            dynamics_out = model.recurrent_inference(
                root.hidden_state,
                torch.tensor([action], device=device),
            )
            entropy = dynamics_out.chance_entropy.item()

        # Record transition
        history.append(
            observation=observation.cpu(),
            action=action,
            reward=result.reward,
            policy=policy,
            root_value=root.value,
            chance_outcome=result.chance_outcome,
            entropy=entropy,
            latent_state=root.hidden_state.cpu() if root.hidden_state is not None else None,
            afterstate=dynamics_out.afterstate.cpu() if dynamics_out.afterstate is not None else None,
        )

        total_entropy += entropy
        step_count += 1
        state = result.next_state

    # Game metadata
    if hasattr(game, "get_max_tile"):
        history.max_tile = game.get_max_tile(state)

    # Discover macros from trajectory
    discovered_macros = 0
    if config.discover_macros and macro_cache is not None and history.length > config.min_macro_length:
        trajectory = []
        for i in range(history.length):
            trajectory.append({
                "state": history.latent_states[i] if i < len(history.latent_states) else None,
                "action": history.actions[i],
                "entropy": history.entropies[i],
                "next_state": history.latent_states[i + 1] if i + 1 < len(history.latent_states) else None,
            })

        # Filter out entries without states
        trajectory = [t for t in trajectory if t["state"] is not None]

        macros = discover_macros_from_trajectory(
            trajectory,
            macro_cache,
            min_length=config.min_macro_length,
            max_length=config.max_macro_length,
            training_step=training_step,
        )
        discovered_macros = len(macros)

    # Statistics
    stats = {
        "game_length": history.length,
        "total_reward": history.total_reward,
        "max_tile": history.max_tile,
        "avg_entropy": total_entropy / max(step_count, 1),
        "macro_uses": macro_uses,
        "macros_discovered": discovered_macros,
    }

    return history, stats


class SelfPlayWorker:
    """
    Worker for generating self-play games.

    Can be used with multiprocessing for parallel game generation.
    """

    def __init__(
        self,
        game: Game,
        model: MuZeroNetwork,
        config: SelfPlayConfig,
        macro_cache: Optional[MacroCache] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.game = game
        self.model = model
        self.config = config
        self.macro_cache = macro_cache
        self.device = device

        # Statistics
        self.games_played = 0
        self.total_stats: Dict[str, float] = {}

    def play_games(
        self,
        num_games: int,
        training_step: int = 0,
        progress_bar: bool = True,
    ) -> List[GameHistory]:
        """Play multiple games and return histories."""
        histories = []
        all_stats = []

        iterator = range(num_games)
        if progress_bar:
            iterator = tqdm(iterator, desc="Self-play")

        for _ in iterator:
            history, stats = self_play_game(
                game=self.game,
                model=self.model,
                config=self.config,
                macro_cache=self.macro_cache,
                device=self.device,
                training_step=training_step,
            )
            histories.append(history)
            all_stats.append(stats)
            self.games_played += 1

        # Aggregate statistics
        self._update_stats(all_stats)

        return histories

    def _update_stats(self, stats_list: List[Dict[str, float]]) -> None:
        """Update running statistics."""
        if not stats_list:
            return

        for key in stats_list[0].keys():
            values = [s[key] for s in stats_list]
            self.total_stats[f"avg_{key}"] = np.mean(values)
            self.total_stats[f"max_{key}"] = np.max(values)

    def get_statistics(self) -> Dict[str, float]:
        """Get worker statistics."""
        return {
            "games_played": self.games_played,
            **self.total_stats,
        }

    def update_model(self, model: MuZeroNetwork) -> None:
        """Update the model weights."""
        self.model = model


def run_self_play(
    game: Game,
    model: MuZeroNetwork,
    num_games: int,
    config: Optional[SelfPlayConfig] = None,
    macro_cache: Optional[MacroCache] = None,
    device: torch.device = torch.device("cpu"),
    training_step: int = 0,
) -> Tuple[List[GameHistory], Dict[str, float]]:
    """
    Convenience function to run self-play.

    Args:
        game: Game environment
        model: MuZero network
        num_games: Number of games to play
        config: Self-play configuration
        macro_cache: Optional macro cache
        device: Device for computation
        training_step: Current training step

    Returns:
        (histories, statistics)
    """
    if config is None:
        config = SelfPlayConfig()

    worker = SelfPlayWorker(game, model, config, macro_cache, device)
    histories = worker.play_games(num_games, training_step)

    return histories, worker.get_statistics()
