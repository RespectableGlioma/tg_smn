"""Representation network: observation → latent state.

The representation function h(o) encodes raw observations into a
latent state representation that the dynamics model operates on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MLP(nn.Module):
    """Simple MLP with LayerNorm and ReLU."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RepresentationNetwork(nn.Module):
    """
    Encodes observations to latent states.

    For 2048: MLP on 496-dim binary input → state_dim latent state
    For Go: Would use ResNet on 19×19×17 planes (not implemented here)
    """

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.state_dim = state_dim

        self.encoder = MLP(
            input_dim=observation_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            num_layers=num_layers,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Encode observation to latent state.

        Args:
            observation: Tensor of shape (batch, observation_dim)

        Returns:
            Latent state of shape (batch, state_dim)
        """
        return self.encoder(observation)


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class ResNetRepresentation(nn.Module):
    """
    ResNet-based representation for image/board observations.

    Used for games like Go where the observation is a 2D board.
    """

    def __init__(
        self,
        input_channels: int,
        state_dim: int,
        num_channels: int = 128,
        num_blocks: int = 6,
        board_size: int = 19,
    ):
        super().__init__()
        self.board_size = board_size
        self.state_dim = state_dim

        # Initial convolution
        self.conv_initial = nn.Conv2d(
            input_channels, num_channels, 3, padding=1, bias=False
        )
        self.bn_initial = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        # Project to latent state
        self.fc = nn.Linear(num_channels * board_size * board_size, state_dim)
        self.ln = nn.LayerNorm(state_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Encode board observation to latent state.

        Args:
            observation: Tensor of shape (batch, channels, board_size, board_size)

        Returns:
            Latent state of shape (batch, state_dim)
        """
        x = F.relu(self.bn_initial(self.conv_initial(observation)))

        for block in self.res_blocks:
            x = block(x)

        # Flatten and project
        x = x.view(x.size(0), -1)
        x = self.ln(self.fc(x))

        return x
