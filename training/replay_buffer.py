"""Replay buffer for Stochastic MuZero.

Stores game trajectories with:
- Observations, actions, rewards
- MCTS policies and values
- Chance outcomes (for stochastic games)
- Entropy at each transition (for macro discovery)

Supports prioritized experience replay based on TD error.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from collections import deque


@dataclass
class GameHistory:
    """
    Complete history of a single game/episode.

    Stores all information needed for training and macro discovery.
    """

    # Core trajectory data
    observations: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    # MCTS outputs
    policies: List[np.ndarray] = field(default_factory=list)
    root_values: List[float] = field(default_factory=list)

    # Stochastic MuZero specific
    chance_outcomes: List[int] = field(default_factory=list)
    afterstates: List[torch.Tensor] = field(default_factory=list)

    # Macro discovery data
    entropies: List[float] = field(default_factory=list)
    latent_states: List[torch.Tensor] = field(default_factory=list)

    # Priority sampling
    priorities: Optional[np.ndarray] = None
    game_priority: float = 1.0

    # Metadata
    total_reward: float = 0.0
    max_tile: int = 0  # For 2048
    length: int = 0

    def append(
        self,
        observation: torch.Tensor,
        action: int,
        reward: float,
        policy: np.ndarray,
        root_value: float,
        chance_outcome: int = 0,
        entropy: float = 0.0,
        latent_state: Optional[torch.Tensor] = None,
        afterstate: Optional[torch.Tensor] = None,
    ) -> None:
        """Append a transition to the history."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.root_values.append(root_value)
        self.chance_outcomes.append(chance_outcome)
        self.entropies.append(entropy)

        if latent_state is not None:
            self.latent_states.append(latent_state)
        if afterstate is not None:
            self.afterstates.append(afterstate)

        self.total_reward += reward
        self.length += 1

    def compute_target_values(
        self,
        discount: float,
        td_steps: int,
    ) -> List[float]:
        """
        Compute n-step return targets.

        target_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n v_{t+n}

        Args:
            discount: Discount factor γ
            td_steps: Number of steps for TD target (n)

        Returns:
            List of target values for each position
        """
        targets = []
        n = len(self.rewards)

        for i in range(n):
            value = 0.0
            for j in range(td_steps):
                if i + j < n:
                    value += (discount ** j) * self.rewards[i + j]
                else:
                    break

            # Bootstrap from value estimate
            bootstrap_idx = i + td_steps
            if bootstrap_idx < n:
                value += (discount ** td_steps) * self.root_values[bootstrap_idx]

            targets.append(value)

        return targets

    def compute_priorities(self, td_steps: int, discount: float) -> None:
        """Compute priority scores based on TD error."""
        targets = self.compute_target_values(discount, td_steps)

        # Priority = |target - root_value|
        self.priorities = np.array([
            abs(target - root_value)
            for target, root_value in zip(targets, self.root_values)
        ])

        # Game priority = max priority in game
        self.game_priority = float(np.max(self.priorities)) if len(self.priorities) > 0 else 1.0


@dataclass
class Batch:
    """Training batch."""

    observations: torch.Tensor  # (batch, observation_dim)
    actions: torch.Tensor  # (batch, K)
    target_values: torch.Tensor  # (batch, K+1)
    target_rewards: torch.Tensor  # (batch, K)
    target_policies: torch.Tensor  # (batch, K+1, action_space)
    chance_outcomes: torch.Tensor  # (batch, K)
    weights: torch.Tensor  # (batch,) importance sampling weights

    # Indices for priority updates
    game_indices: List[int] = field(default_factory=list)
    position_indices: List[int] = field(default_factory=list)


class ReplayBuffer:
    """
    Replay buffer with prioritized sampling.

    Stores complete game histories and samples positions
    for training with importance sampling.
    """

    def __init__(
        self,
        capacity: int = 100000,
        batch_size: int = 256,
        num_unroll_steps: int = 5,
        td_steps: int = 10,
        discount: float = 0.997,
        priority_alpha: float = 1.0,
        priority_beta: float = 1.0,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.discount = discount
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta

        # Storage
        self.games: deque = deque(maxlen=capacity)
        self.total_positions = 0

        # Statistics
        self.games_added = 0
        self.total_samples = 0

    def save_game(self, game: GameHistory) -> None:
        """Add a completed game to the buffer."""
        # Compute priorities
        game.compute_priorities(self.td_steps, self.discount)

        # Update total positions
        if len(self.games) == self.games.maxlen:
            old_game = self.games[0]
            self.total_positions -= old_game.length

        self.games.append(game)
        self.total_positions += game.length
        self.games_added += 1

    def sample_batch(self, device: torch.device = torch.device("cpu")) -> Batch:
        """
        Sample a batch of positions for training.

        Uses prioritized sampling at both game and position level.

        Returns:
            Batch object with all training data
        """
        # Compute game sampling probabilities
        game_priorities = np.array([g.game_priority ** self.priority_alpha for g in self.games])
        game_probs = game_priorities / game_priorities.sum()

        # Sample games
        game_indices = np.random.choice(
            len(self.games),
            size=self.batch_size,
            p=game_probs,
            replace=True,
        )

        # Sample positions within games
        observations = []
        actions = []
        target_values = []
        target_rewards = []
        target_policies = []
        chance_outcomes = []
        weights = []
        position_indices = []

        for game_idx in game_indices:
            game = self.games[game_idx]

            # Sample position within game
            if game.priorities is not None:
                pos_priorities = game.priorities ** self.priority_alpha
                pos_probs = pos_priorities / pos_priorities.sum()
                pos_idx = np.random.choice(len(game.observations), p=pos_probs)
            else:
                pos_idx = np.random.randint(len(game.observations))

            # Compute importance sampling weight
            total_prob = game_probs[game_idx]
            if game.priorities is not None:
                total_prob *= pos_probs[pos_idx]
            weight = (1.0 / (self.total_positions * total_prob)) ** self.priority_beta

            # Get observation
            obs = game.observations[pos_idx]
            observations.append(obs)

            # Get action sequence (pad if needed)
            action_seq = []
            reward_seq = []
            chance_seq = []
            value_targets = []
            policy_targets = []

            # Compute value targets
            all_targets = game.compute_target_values(self.discount, self.td_steps)
            value_targets.append(all_targets[pos_idx])

            for k in range(self.num_unroll_steps):
                idx = pos_idx + k
                if idx < len(game.actions):
                    action_seq.append(game.actions[idx])
                    reward_seq.append(game.rewards[idx])
                    chance_seq.append(game.chance_outcomes[idx])
                    policy_targets.append(game.policies[idx])
                    if idx + 1 < len(all_targets):
                        value_targets.append(all_targets[idx + 1])
                    else:
                        value_targets.append(0.0)
                else:
                    # Pad with zeros
                    action_seq.append(0)
                    reward_seq.append(0.0)
                    chance_seq.append(0)
                    policy_targets.append(game.policies[-1])
                    value_targets.append(0.0)

            # Initial policy
            policy_targets.insert(0, game.policies[pos_idx])

            actions.append(action_seq)
            target_rewards.append(reward_seq)
            target_values.append(value_targets)
            target_policies.append(policy_targets)
            chance_outcomes.append(chance_seq)
            weights.append(weight)
            position_indices.append(pos_idx)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.max()

        # Convert to tensors
        self.total_samples += self.batch_size

        # Stack observations
        obs_tensor = torch.stack(observations).to(device)

        # Convert action space size from policies
        action_space_size = len(target_policies[0][0])

        # Pad policies to same size
        padded_policies = []
        for policy_seq in target_policies:
            padded_seq = []
            for p in policy_seq:
                if len(p) < action_space_size:
                    padded = np.zeros(action_space_size)
                    padded[:len(p)] = p
                    padded_seq.append(padded)
                else:
                    padded_seq.append(p)
            padded_policies.append(padded_seq)

        return Batch(
            observations=obs_tensor,
            actions=torch.tensor(actions, dtype=torch.long, device=device),
            target_values=torch.tensor(target_values, dtype=torch.float32, device=device),
            target_rewards=torch.tensor(target_rewards, dtype=torch.float32, device=device),
            target_policies=torch.tensor(
                np.array(padded_policies), dtype=torch.float32, device=device
            ),
            chance_outcomes=torch.tensor(chance_outcomes, dtype=torch.long, device=device),
            weights=torch.tensor(weights, dtype=torch.float32, device=device),
            game_indices=list(game_indices),
            position_indices=position_indices,
        )

    def update_priorities(
        self,
        game_indices: List[int],
        position_indices: List[int],
        td_errors: np.ndarray,
    ) -> None:
        """Update priorities based on new TD errors."""
        for game_idx, pos_idx, error in zip(game_indices, position_indices, td_errors):
            if game_idx < len(self.games):
                game = self.games[game_idx]
                if game.priorities is not None and pos_idx < len(game.priorities):
                    game.priorities[pos_idx] = abs(error)
                    game.game_priority = float(np.max(game.priorities))

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        return {
            "num_games": len(self.games),
            "total_positions": self.total_positions,
            "games_added": self.games_added,
            "total_samples": self.total_samples,
            "avg_game_length": (
                self.total_positions / len(self.games) if self.games else 0.0
            ),
            "avg_total_reward": (
                np.mean([g.total_reward for g in self.games]) if self.games else 0.0
            ),
        }

    def __len__(self) -> int:
        return self.total_positions
