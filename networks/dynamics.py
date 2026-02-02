"""Dynamics networks for Stochastic MuZero.

The dynamics model is split into three components:
1. AfterstateDynamics: φ(s, a) → afterstate (deterministic rule core)
2. ChanceEncoder: σ(c | afterstate) → chance distribution + entropy
3. Dynamics: g(afterstate, c) → (next_state, reward)

This factorization enables:
- Learning deterministic rules explicitly
- Measuring transition entropy for macro discovery
- Separating controllable vs. stochastic parts of dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .representation import MLP


class AfterstateDynamics(nn.Module):
    """
    Deterministic afterstate dynamics: φ(s, a) → afterstate.

    This is the "rule core" - it captures the deterministic effect
    of actions, before any stochastic chance outcomes.

    For 2048: Models the slide/merge operation.
    For Go: Models stone placement + captures.
    """

    def __init__(
        self,
        state_dim: int,
        action_space_size: int,
        hidden_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_space_size = action_space_size

        # Input: state + one-hot action
        input_dim = state_dim + action_space_size

        self.net = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            num_layers=num_layers,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute afterstate from state and action.

        Args:
            state: Latent state of shape (batch, state_dim)
            action: One-hot action of shape (batch, action_space_size)
                   OR action indices of shape (batch,)

        Returns:
            Afterstate of shape (batch, state_dim)
        """
        # Convert action indices to one-hot if needed
        if action.dim() == 1:
            action = F.one_hot(action, self.action_space_size).float()

        x = torch.cat([state, action], dim=-1)
        afterstate = self.net(x)

        return afterstate


class ChanceEncoder(nn.Module):
    """
    Chance distribution predictor: σ(c | afterstate) → distribution.

    Predicts the probability distribution over chance outcomes
    given the afterstate. Also provides entropy for macro discovery.

    For 2048: Predicts where/what tile will spawn.
    For Go: Would be trivial (deterministic, entropy ≈ 0).
    """

    def __init__(
        self,
        state_dim: int,
        chance_space_size: int,
        hidden_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.chance_space_size = chance_space_size

        self.net = MLP(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=chance_space_size,
            num_layers=num_layers,
        )

    def forward(
        self, afterstate: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict chance distribution from afterstate.

        Args:
            afterstate: Afterstate of shape (batch, state_dim)

        Returns:
            logits: Chance logits of shape (batch, chance_space_size)
            entropy: Entropy of distribution of shape (batch,)
        """
        logits = self.net(afterstate)

        # Compute entropy for macro discovery
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        return logits, entropy

    def sample(self, afterstate: torch.Tensor) -> torch.Tensor:
        """Sample a chance outcome from the predicted distribution."""
        logits, _ = self.forward(afterstate)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def top_k(
        self, afterstate: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k most likely chance outcomes.

        Args:
            afterstate: Afterstate of shape (batch, state_dim)
            k: Number of outcomes to return

        Returns:
            indices: Top-k chance indices of shape (batch, k)
            probs: Corresponding probabilities of shape (batch, k)
        """
        logits, _ = self.forward(afterstate)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k, dim=-1)
        return top_indices, top_probs


class Dynamics(nn.Module):
    """
    Full dynamics: g(afterstate, c) → (next_state, reward).

    Given the afterstate and a chance outcome, produces the next
    latent state and the immediate reward.

    This is deterministic given the chance outcome.
    """

    def __init__(
        self,
        state_dim: int,
        chance_space_size: int,
        hidden_dim: int,
        support_size: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.chance_space_size = chance_space_size
        self.support_size = support_size

        # Input: afterstate + one-hot chance
        input_dim = state_dim + chance_space_size

        # Shared trunk
        self.trunk = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers - 1,
        )

        # State head
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

        # Reward head (categorical distribution over support)
        reward_support_dim = 2 * support_size + 1
        self.reward_head = nn.Linear(hidden_dim, reward_support_dim)

    def forward(
        self, afterstate: torch.Tensor, chance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute next state and reward from afterstate and chance.

        Args:
            afterstate: Afterstate of shape (batch, state_dim)
            chance: One-hot chance of shape (batch, chance_space_size)
                   OR chance indices of shape (batch,)

        Returns:
            next_state: Next latent state of shape (batch, state_dim)
            reward_logits: Reward distribution of shape (batch, 2*support_size+1)
        """
        # Convert chance indices to one-hot if needed
        if chance.dim() == 1:
            chance = F.one_hot(chance, self.chance_space_size).float()

        x = torch.cat([afterstate, chance], dim=-1)
        features = self.trunk(x)

        next_state = self.state_head(features)
        reward_logits = self.reward_head(features)

        return next_state, reward_logits


class DynamicsModel(nn.Module):
    """
    Combined dynamics model wrapping all three components.

    Provides convenience methods for:
    - Full transitions: state, action → next_state, reward, entropy
    - Macro-style jumps: state, action_sequence → final_state
    """

    def __init__(
        self,
        state_dim: int,
        action_space_size: int,
        chance_space_size: int,
        hidden_dim: int,
        support_size: int,
        num_layers: int = 2,
    ):
        super().__init__()

        self.afterstate_dynamics = AfterstateDynamics(
            state_dim=state_dim,
            action_space_size=action_space_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        self.chance_encoder = ChanceEncoder(
            state_dim=state_dim,
            chance_space_size=chance_space_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        self.dynamics = Dynamics(
            state_dim=state_dim,
            chance_space_size=chance_space_size,
            hidden_dim=hidden_dim,
            support_size=support_size,
            num_layers=num_layers,
        )

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        chance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full transition with given chance outcome.

        Args:
            state: Current state (batch, state_dim)
            action: Action indices (batch,)
            chance: Chance indices (batch,)

        Returns:
            afterstate: Afterstate (batch, state_dim)
            next_state: Next state (batch, state_dim)
            reward_logits: Reward distribution (batch, support_size)
            entropy: Chance entropy (batch,)
        """
        afterstate = self.afterstate_dynamics(state, action)
        _, entropy = self.chance_encoder(afterstate)
        next_state, reward_logits = self.dynamics(afterstate, chance)

        return afterstate, next_state, reward_logits, entropy

    def predict_transition(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict transition by sampling chance internally.

        Args:
            state: Current state (batch, state_dim)
            action: Action indices (batch,)

        Returns:
            afterstate, next_state, reward_logits, entropy
        """
        afterstate = self.afterstate_dynamics(state, action)
        chance_logits, entropy = self.chance_encoder(afterstate)
        chance = torch.multinomial(F.softmax(chance_logits, dim=-1), 1).squeeze(-1)
        next_state, reward_logits = self.dynamics(afterstate, chance)

        return afterstate, next_state, reward_logits, entropy

    def rollout(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        chances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-step rollout with given actions and chances.

        Args:
            state: Initial state (batch, state_dim)
            actions: Action sequence (batch, k)
            chances: Chance sequence (batch, k)

        Returns:
            final_state: State after k steps (batch, state_dim)
            total_reward_logits: Sum of reward logits (batch, support_size)
            max_entropy: Maximum entropy encountered (batch,)
        """
        current_state = state
        total_reward = None
        max_entropy = torch.zeros(state.shape[0], device=state.device)

        k = actions.shape[1]
        for i in range(k):
            action = actions[:, i]
            chance = chances[:, i]

            afterstate, next_state, reward_logits, entropy = self.step(
                current_state, action, chance
            )

            if total_reward is None:
                total_reward = reward_logits
            else:
                total_reward = total_reward + reward_logits

            max_entropy = torch.maximum(max_entropy, entropy)
            current_state = next_state

        return current_state, total_reward, max_entropy
