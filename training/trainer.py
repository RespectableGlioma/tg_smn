"""Training loop for Stochastic MuZero.

Handles:
- Loss computation (policy, value, reward, chance, macro)
- Gradient updates with proper scaling
- Training statistics and logging
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..networks.muzero_network import MuZeroNetwork
from ..utils.support import scalar_to_support, compute_cross_entropy_loss
from .replay_buffer import ReplayBuffer, Batch


@dataclass
class TrainerConfig:
    """Configuration for training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0

    # Loss weights
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    reward_loss_weight: float = 1.0
    chance_loss_weight: float = 1.0

    # Support size for categorical distributions
    support_size: int = 31

    # Learning rate schedule
    lr_decay_steps: int = 100000
    lr_min: float = 1e-5


class Trainer:
    """
    Trainer for Stochastic MuZero.

    Handles the complete training loop including:
    - Unrolling dynamics for K steps
    - Computing all losses (policy, value, reward, chance)
    - Gradient updates with clipping
    """

    def __init__(
        self,
        model: MuZeroNetwork,
        config: TrainerConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model = self.model.to(device)

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.lr_decay_steps,
            eta_min=config.lr_min,
        )

        # Training statistics
        self.training_step = 0
        self.loss_history: Dict[str, List[float]] = {
            "total": [],
            "policy": [],
            "value": [],
            "reward": [],
            "chance": [],
        }

    def train_step(self, batch: Batch) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Training batch from replay buffer

        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Unroll model
        unroll_outputs = self.model.unroll(
            observation=batch.observations,
            actions=batch.actions,
            chances=batch.chance_outcomes,
        )

        # Compute losses
        losses = self._compute_losses(batch, unroll_outputs)

        # Total loss
        total_loss = (
            self.config.policy_loss_weight * losses["policy"]
            + self.config.value_loss_weight * losses["value"]
            + self.config.reward_loss_weight * losses["reward"]
            + self.config.chance_loss_weight * losses["chance"]
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        # Update statistics
        self.training_step += 1
        loss_dict = {
            "total": total_loss.item(),
            "policy": losses["policy"].item(),
            "value": losses["value"].item(),
            "reward": losses["reward"].item(),
            "chance": losses["chance"].item(),
            "grad_norm": grad_norm.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

        for key, value in loss_dict.items():
            if key in self.loss_history:
                self.loss_history[key].append(value)

        return loss_dict

    def _compute_losses(
        self,
        batch: Batch,
        unroll_outputs: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all training losses.

        Args:
            batch: Training batch
            unroll_outputs: Outputs from model.unroll()

        Returns:
            Dictionary of loss tensors
        """
        batch_size = batch.observations.shape[0]
        K = batch.actions.shape[1]  # Number of unroll steps

        # Policy loss: cross-entropy with MCTS policy
        policy_loss = torch.tensor(0.0, device=self.device)
        for k in range(K + 1):
            predicted_policy = F.log_softmax(unroll_outputs["policy_logits"][k], dim=-1)
            target_policy = batch.target_policies[:, k, :]

            # Mask out zero policies (padding)
            mask = target_policy.sum(dim=-1) > 0
            if mask.any():
                policy_loss += -(target_policy[mask] * predicted_policy[mask]).sum(dim=-1).mean()

        policy_loss = policy_loss / (K + 1)

        # Value loss: cross-entropy with n-step return
        value_loss = torch.tensor(0.0, device=self.device)
        for k in range(K + 1):
            value_loss += compute_cross_entropy_loss(
                unroll_outputs["value_logits"][k],
                batch.target_values[:, k],
                self.config.support_size,
            )
        value_loss = value_loss / (K + 1)

        # Reward loss: cross-entropy with observed reward (only for k > 0)
        reward_loss = torch.tensor(0.0, device=self.device)
        for k in range(K):
            reward_loss += compute_cross_entropy_loss(
                unroll_outputs["reward_logits"][k],
                batch.target_rewards[:, k],
                self.config.support_size,
            )
        reward_loss = reward_loss / max(K, 1)

        # Chance loss: cross-entropy with observed chance outcome
        chance_loss = torch.tensor(0.0, device=self.device)
        for k in range(K):
            predicted_chance = F.log_softmax(unroll_outputs["chance_logits"][k], dim=-1)
            target_chance = batch.chance_outcomes[:, k]

            # Only compute loss for non-padding transitions
            mask = target_chance >= 0
            if mask.any():
                chance_loss += F.nll_loss(
                    predicted_chance[mask],
                    target_chance[mask],
                    reduction="mean",
                )
        chance_loss = chance_loss / max(K, 1)

        # Apply importance sampling weights
        weights = batch.weights.unsqueeze(-1)

        return {
            "policy": (policy_loss * weights.squeeze()).mean(),
            "value": (value_loss * weights.squeeze()).mean(),
            "reward": reward_loss,  # Reward loss not weighted
            "chance": chance_loss,  # Chance loss not weighted
        }

    def get_td_errors(
        self,
        batch: Batch,
    ) -> np.ndarray:
        """
        Compute TD errors for priority updates.

        Args:
            batch: Training batch

        Returns:
            Array of TD errors for each sample
        """
        self.model.eval()

        with torch.no_grad():
            # Initial inference
            initial = self.model.initial_inference(batch.observations)

            # Get value prediction
            value_logits = initial.value_logits
            value_probs = F.softmax(value_logits, dim=-1)

            # Convert to scalar
            support = torch.arange(
                -self.config.support_size,
                self.config.support_size + 1,
                device=self.device,
                dtype=torch.float32,
            )
            predicted_values = (value_probs * support).sum(dim=-1)

            # TD error = |target - predicted|
            td_errors = torch.abs(batch.target_values[:, 0] - predicted_values)

        return td_errors.cpu().numpy()

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_step": self.training_step,
            "loss_history": self.loss_history,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_step = checkpoint["training_step"]
        self.loss_history = checkpoint.get("loss_history", self.loss_history)

    def get_statistics(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {
            "training_step": self.training_step,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

        # Recent loss averages
        window = min(100, len(self.loss_history.get("total", [])))
        if window > 0:
            for key in self.loss_history:
                stats[f"avg_{key}_loss"] = np.mean(self.loss_history[key][-window:])

        return stats


def train_epoch(
    trainer: Trainer,
    replay_buffer: ReplayBuffer,
    num_batches: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train for one epoch (multiple batches).

    Args:
        trainer: Trainer instance
        replay_buffer: Replay buffer with game histories
        num_batches: Number of batches to train on
        device: Device for computation

    Returns:
        Average losses for the epoch
    """
    epoch_losses = {
        "total": [],
        "policy": [],
        "value": [],
        "reward": [],
        "chance": [],
    }

    for _ in range(num_batches):
        # Sample batch
        batch = replay_buffer.sample_batch(device)

        # Train step
        losses = trainer.train_step(batch)

        # Record losses
        for key in epoch_losses:
            if key in losses:
                epoch_losses[key].append(losses[key])

        # Update priorities
        td_errors = trainer.get_td_errors(batch)
        replay_buffer.update_priorities(
            batch.game_indices,
            batch.position_indices,
            td_errors,
        )

    # Average losses
    return {key: np.mean(values) for key, values in epoch_losses.items()}
