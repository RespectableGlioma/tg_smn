"""Prediction networks: state → (policy, value).

The prediction function f(s) outputs:
- Policy: probability distribution over actions
- Value: expected cumulative reward from this state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .representation import MLP


class PredictionNetwork(nn.Module):
    """
    Prediction network: state → (policy, value).

    Used for:
    1. MCTS: Getting initial policy priors and value estimates
    2. Training: Policy and value targets from MCTS results
    """

    def __init__(
        self,
        state_dim: int,
        action_space_size: int,
        hidden_dim: int,
        support_size: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.support_size = support_size

        # Shared trunk
        self.trunk = MLP(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers - 1,
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, action_space_size)

        # Value head (categorical distribution over support)
        value_support_dim = 2 * support_size + 1
        self.value_head = nn.Linear(hidden_dim, value_support_dim)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict policy and value from state.

        Args:
            state: Latent state of shape (batch, state_dim)

        Returns:
            policy_logits: Action logits of shape (batch, action_space_size)
            value_logits: Value distribution of shape (batch, 2*support_size+1)
        """
        features = self.trunk(state)
        policy_logits = self.policy_head(features)
        value_logits = self.value_head(features)

        return policy_logits, value_logits

    def predict_policy(self, state: torch.Tensor) -> torch.Tensor:
        """Get policy probabilities."""
        policy_logits, _ = self.forward(state)
        return F.softmax(policy_logits, dim=-1)

    def predict_value(self, state: torch.Tensor, support_size: int) -> torch.Tensor:
        """Get expected value (scalar)."""
        from ..utils.support import support_to_scalar

        _, value_logits = self.forward(state)
        value_probs = F.softmax(value_logits, dim=-1)
        return support_to_scalar(value_probs, support_size)


class AfterstatePrediction(nn.Module):
    """
    Afterstate prediction: afterstate → (policy, value).

    Used for planning from afterstates in stochastic MCTS.
    The afterstate represents the world after our action but before
    the environment's stochastic response.
    """

    def __init__(
        self,
        state_dim: int,
        action_space_size: int,
        hidden_dim: int,
        support_size: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.support_size = support_size

        # Shared trunk
        self.trunk = MLP(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers - 1,
        )

        # Policy head (for next action after chance resolves)
        self.policy_head = nn.Linear(hidden_dim, action_space_size)

        # Value head
        value_support_dim = 2 * support_size + 1
        self.value_head = nn.Linear(hidden_dim, value_support_dim)

    def forward(
        self, afterstate: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict policy and value from afterstate.

        Args:
            afterstate: Afterstate of shape (batch, state_dim)

        Returns:
            policy_logits: Action logits of shape (batch, action_space_size)
            value_logits: Value distribution of shape (batch, 2*support_size+1)
        """
        features = self.trunk(afterstate)
        policy_logits = self.policy_head(features)
        value_logits = self.value_head(features)

        return policy_logits, value_logits
