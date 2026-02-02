"""Combined MuZero network with all components.

This module combines all the network components into a single
MuZeroNetwork class that can be used for training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .representation import RepresentationNetwork
from .dynamics import AfterstateDynamics, ChanceEncoder, Dynamics
from .prediction import PredictionNetwork, AfterstatePrediction
from .encoder import ChanceInferenceEncoder


@dataclass
class NetworkOutput:
    """Output from a single network forward pass."""

    state: torch.Tensor  # Latent state
    policy_logits: torch.Tensor  # Policy distribution
    value_logits: torch.Tensor  # Value distribution


@dataclass
class DynamicsOutput:
    """Output from dynamics forward pass."""

    afterstate: torch.Tensor
    next_state: torch.Tensor
    reward_logits: torch.Tensor
    chance_logits: torch.Tensor
    chance_entropy: torch.Tensor
    afterstate_policy_logits: torch.Tensor
    afterstate_value_logits: torch.Tensor


class MuZeroNetwork(nn.Module):
    """
    Complete Stochastic MuZero network.

    Components:
    - Representation: h(o) → s
    - Afterstate dynamics: φ(s, a) → afterstate
    - Chance encoder: σ(c | afterstate) → chance distribution
    - Dynamics: g(afterstate, c) → (s', r)
    - Prediction: f(s) → (π, v)
    - Afterstate prediction: f_a(afterstate) → (π, v)
    - Inference encoder: enc(o_t, a, o_{t+1}) → c (for training)
    """

    def __init__(
        self,
        observation_dim: int,
        action_space_size: int,
        chance_space_size: int,
        state_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        support_size: int = 31,
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_space_size = action_space_size
        self.chance_space_size = chance_space_size
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.support_size = support_size

        # Representation network
        self.representation = RepresentationNetwork(
            observation_dim=observation_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Afterstate dynamics (deterministic rule)
        self.afterstate_dynamics = AfterstateDynamics(
            state_dim=state_dim,
            action_space_size=action_space_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Chance encoder (stochastic predictor)
        self.chance_encoder = ChanceEncoder(
            state_dim=state_dim,
            chance_space_size=chance_space_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Dynamics (given chance outcome)
        self.dynamics = Dynamics(
            state_dim=state_dim,
            chance_space_size=chance_space_size,
            hidden_dim=hidden_dim,
            support_size=support_size,
            num_layers=num_layers,
        )

        # Prediction networks
        self.prediction = PredictionNetwork(
            state_dim=state_dim,
            action_space_size=action_space_size,
            hidden_dim=hidden_dim,
            support_size=support_size,
            num_layers=num_layers,
        )

        self.afterstate_prediction = AfterstatePrediction(
            state_dim=state_dim,
            action_space_size=action_space_size,
            hidden_dim=hidden_dim,
            support_size=support_size,
            num_layers=num_layers,
        )

        # Inference encoder (for training)
        self.inference_encoder = ChanceInferenceEncoder(
            observation_dim=observation_dim,
            action_space_size=action_space_size,
            chance_space_size=chance_space_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def initial_inference(
        self, observation: torch.Tensor
    ) -> NetworkOutput:
        """
        Initial inference from observation.

        Used at the root of MCTS to get initial state, policy, and value.

        Args:
            observation: Raw observation (batch, observation_dim)

        Returns:
            NetworkOutput with state, policy_logits, value_logits
        """
        state = self.representation(observation)
        policy_logits, value_logits = self.prediction(state)

        return NetworkOutput(
            state=state,
            policy_logits=policy_logits,
            value_logits=value_logits,
        )

    def recurrent_inference(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        chance: Optional[torch.Tensor] = None,
    ) -> DynamicsOutput:
        """
        Recurrent inference for dynamics unrolling.

        Used during MCTS and training to predict transitions.

        Args:
            state: Current latent state (batch, state_dim)
            action: Action to take (batch,)
            chance: Chance outcome (batch,). If None, samples from distribution.

        Returns:
            DynamicsOutput with all transition information
        """
        # Compute afterstate (deterministic rule)
        afterstate = self.afterstate_dynamics(state, action)

        # Predict chance distribution
        chance_logits, chance_entropy = self.chance_encoder(afterstate)

        # Sample chance if not provided
        if chance is None:
            probs = F.softmax(chance_logits, dim=-1)
            chance = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Apply dynamics
        next_state, reward_logits = self.dynamics(afterstate, chance)

        # Get afterstate predictions (for stochastic MCTS)
        afterstate_policy_logits, afterstate_value_logits = self.afterstate_prediction(
            afterstate
        )

        return DynamicsOutput(
            afterstate=afterstate,
            next_state=next_state,
            reward_logits=reward_logits,
            chance_logits=chance_logits,
            chance_entropy=chance_entropy,
            afterstate_policy_logits=afterstate_policy_logits,
            afterstate_value_logits=afterstate_value_logits,
        )

    def predict_state(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get policy and value for a state."""
        return self.prediction(state)

    def predict_afterstate(
        self, afterstate: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get policy and value for an afterstate."""
        return self.afterstate_prediction(afterstate)

    def infer_chance(
        self,
        obs_t: torch.Tensor,
        action: torch.Tensor,
        obs_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Infer chance outcome from consecutive observations (for training)."""
        return self.inference_encoder(obs_t, action, obs_t1)

    def unroll(
        self,
        observation: torch.Tensor,
        actions: torch.Tensor,
        chances: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Unroll dynamics for K steps (used in training).

        Args:
            observation: Initial observation (batch, observation_dim)
            actions: Action sequence (batch, K)
            chances: Chance sequence (batch, K)

        Returns:
            Dictionary with lists of outputs for each timestep:
            - states: [s_0, s_1, ..., s_K]
            - afterstates: [a_1, ..., a_K]
            - policy_logits: [π_0, π_1, ..., π_K]
            - value_logits: [v_0, v_1, ..., v_K]
            - reward_logits: [r_1, ..., r_K]
            - chance_logits: [σ_1, ..., σ_K]
            - chance_entropies: [H_1, ..., H_K]
        """
        batch_size, K = actions.shape

        # Initial inference
        initial = self.initial_inference(observation)

        # Collect outputs
        states = [initial.state]
        afterstates = []
        policy_logits = [initial.policy_logits]
        value_logits = [initial.value_logits]
        reward_logits = []
        chance_logits = []
        chance_entropies = []

        current_state = initial.state

        # Unroll K steps
        for k in range(K):
            # Scale gradients by 0.5 through dynamics (MuZero trick)
            current_state = scale_gradient(current_state, 0.5)

            dynamics_out = self.recurrent_inference(
                current_state,
                actions[:, k],
                chances[:, k],
            )

            afterstates.append(dynamics_out.afterstate)
            states.append(dynamics_out.next_state)
            reward_logits.append(dynamics_out.reward_logits)
            chance_logits.append(dynamics_out.chance_logits)
            chance_entropies.append(dynamics_out.chance_entropy)

            # Prediction at next state
            next_policy, next_value = self.predict_state(dynamics_out.next_state)
            policy_logits.append(next_policy)
            value_logits.append(next_value)

            current_state = dynamics_out.next_state

        return {
            "states": states,
            "afterstates": afterstates,
            "policy_logits": policy_logits,
            "value_logits": value_logits,
            "reward_logits": reward_logits,
            "chance_logits": chance_logits,
            "chance_entropies": chance_entropies,
        }

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get all network weights as a dictionary."""
        return {name: param.data for name, param in self.named_parameters()}

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Set network weights from a dictionary."""
        for name, param in self.named_parameters():
            if name in weights:
                param.data.copy_(weights[name])


class ScaleGradient(torch.autograd.Function):
    """Scales gradients by a constant factor during backprop."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale gradients flowing through this tensor."""
    return ScaleGradient.apply(x, scale)
