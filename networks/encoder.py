"""Encoder network for inferring chance outcomes from observations.

In Stochastic MuZero, the encoder is used during training to:
1. Infer what chance outcome actually occurred (from real observations)
2. Provide supervision for the chance predictor

This is necessary because we observe s_{t+1} but need to know what
chance outcome c led to this transition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .representation import MLP


class ChanceInferenceEncoder(nn.Module):
    """
    Infers chance outcomes from consecutive observations.

    Given (observation_t, action, observation_{t+1}), infers
    what chance outcome must have occurred.

    This is used during training to provide supervision for
    the chance predictor σ(c | afterstate).
    """

    def __init__(
        self,
        observation_dim: int,
        action_space_size: int,
        chance_space_size: int,
        hidden_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_space_size = action_space_size
        self.chance_space_size = chance_space_size

        # Input: obs_t + action + obs_{t+1}
        input_dim = 2 * observation_dim + action_space_size

        self.net = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=chance_space_size,
            num_layers=num_layers,
        )

    def forward(
        self,
        obs_t: torch.Tensor,
        action: torch.Tensor,
        obs_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Infer chance outcome from observations.

        Args:
            obs_t: Observation at time t (batch, observation_dim)
            action: Action taken (batch,) or (batch, action_space_size)
            obs_t1: Observation at time t+1 (batch, observation_dim)

        Returns:
            chance_logits: Inferred chance distribution (batch, chance_space_size)
        """
        # Convert action indices to one-hot if needed
        if action.dim() == 1:
            action = F.one_hot(action, self.action_space_size).float()

        x = torch.cat([obs_t, action, obs_t1], dim=-1)
        return self.net(x)

    def infer_chance(
        self,
        obs_t: torch.Tensor,
        action: torch.Tensor,
        obs_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get most likely chance outcome.

        Returns:
            chance: Most likely chance index (batch,)
        """
        logits = self.forward(obs_t, action, obs_t1)
        return logits.argmax(dim=-1)


class StateEncoder(nn.Module):
    """
    Alternative encoder that works in latent space.

    Given (state_t, action, state_{t+1}), infers the chance outcome.
    Used when we have access to latent states rather than raw observations.
    """

    def __init__(
        self,
        state_dim: int,
        action_space_size: int,
        chance_space_size: int,
        hidden_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.chance_space_size = chance_space_size

        # Input: state_t + action + state_{t+1}
        input_dim = 2 * state_dim + action_space_size

        self.net = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=chance_space_size,
            num_layers=num_layers,
        )

    def forward(
        self,
        state_t: torch.Tensor,
        action: torch.Tensor,
        state_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Infer chance outcome from latent states.

        Args:
            state_t: State at time t (batch, state_dim)
            action: Action taken (batch,) or (batch, action_space_size)
            state_t1: State at time t+1 (batch, state_dim)

        Returns:
            chance_logits: Inferred chance distribution (batch, chance_space_size)
        """
        if action.dim() == 1:
            action = F.one_hot(action, self.action_space_size).float()

        x = torch.cat([state_t, action, state_t1], dim=-1)
        return self.net(x)
