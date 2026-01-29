"""
World Model Architecture: Causal Core + Stochastic Envelope

This implements the key separation:

1. CAUSAL CORE (Deterministic):
   - Representation: obs → latent state s
   - Dynamics: (s, a) → afterstate s' (DETERMINISTIC)
   - This is where the "rules" live
   
2. STOCHASTIC ENVELOPE (Chance):
   - Chance predictor: afterstate → distribution over outcomes
   - This captures irreducible randomness (e.g., tile spawns in 2048)
   
3. INVARIANCE STRUCTURE:
   - Separate "causal state" s from "nuisance/style" u
   - s should be invariant to appearance changes
   - u captures rendering variations
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for the world model."""
    # Game parameters
    board_size: int = 4  # 4 for 2048, 8 for Othello
    num_cell_classes: int = 17  # Number of distinct cell values
    num_actions: int = 4  # Number of actions
    
    # Latent dimensions
    hidden_dim: int = 256
    state_dim: int = 128  # Causal state dimension
    style_dim: int = 32   # Nuisance/style dimension (for invariance)
    
    # Architecture
    num_res_blocks: int = 4
    dropout: float = 0.1
    
    # Game-specific
    has_chance: bool = True  # Whether game has stochastic transitions
    num_styles: int = 1      # Number of rendering styles (for invariance training)


class ResBlock(nn.Module):
    """Residual block with optional dropout."""
    
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ConvResBlock(nn.Module):
    """Convolutional residual block."""
    
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.norm = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class BoardEncoder(nn.Module):
    """
    Encode board observation → latent representation.
    
    Input: (B, C, H, W) where C = num_cell_classes (one-hot per cell)
    Output: (B, state_dim + style_dim)
    
    The encoder produces two components:
    - state: causal state (should be invariant to style)
    - style: nuisance factors (appearance, etc.)
    """
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Convolutional encoder
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.num_cell_classes, 64, 3, padding=1),
            nn.ReLU(),
            ConvResBlock(64, cfg.dropout),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            ConvResBlock(128, cfg.dropout),
        )
        
        # Flatten and project
        flat_dim = 128 * cfg.board_size * cfg.board_size
        
        # Separate heads for state and style
        self.state_head = nn.Sequential(
            nn.Linear(flat_dim, cfg.hidden_dim),
            nn.ReLU(),
            ResBlock(cfg.hidden_dim, cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.state_dim),
        )
        
        self.style_head = nn.Sequential(
            nn.Linear(flat_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, cfg.style_dim),
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            state: (B, state_dim) - causal state
            style: (B, style_dim) - nuisance/style
        """
        B = obs.shape[0]
        h = self.conv(obs)
        h = h.reshape(B, -1)
        
        state = self.state_head(h)
        style = self.style_head(h)
        
        return state, style


class DynamicsNetwork(nn.Module):
    """
    DETERMINISTIC dynamics: (state, action) → afterstate
    
    This is the CAUSAL RULE CORE. Given the current state and action,
    it predicts the DETERMINISTIC result (before any chance events).
    
    Key property: This should have LOW or ZERO entropy.
    The same (state, action) should always produce the same afterstate.
    """
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Action embedding
        self.action_embed = nn.Embedding(cfg.num_actions, 32)
        
        # Dynamics MLP
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim + 32, cfg.hidden_dim),
            nn.ReLU(),
            ResBlock(cfg.hidden_dim, cfg.dropout),
            ResBlock(cfg.hidden_dim, cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.state_dim),
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
            action: (B,) long tensor
        
        Returns:
            afterstate: (B, state_dim) - deterministic next state
        """
        a_emb = self.action_embed(action)  # (B, 32)
        x = torch.cat([state, a_emb], dim=-1)
        return self.net(x)


class ChanceNetwork(nn.Module):
    """
    STOCHASTIC chance predictor: afterstate → distribution over outcomes
    
    For 2048: predicts spawn position (masked to empties) and value (2 or 4)
    For Othello: should predict near-deterministic (entropy ≈ 0)
    
    The chance network outputs logits for:
    - Position: (B, board_size^2) logits over cells
    - Value: (B, 2) logits for 2 vs 4 (or game-specific)
    
    The position logits should be masked to only allow empty cells.
    """
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.board_cells = cfg.board_size ** 2
        
        # Chance MLP
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.ReLU(),
            ResBlock(cfg.hidden_dim, cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        
        # Position head (where to spawn)
        self.position_head = nn.Linear(cfg.hidden_dim, self.board_cells)
        
        # Value head (what to spawn: 2 or 4)
        self.value_head = nn.Linear(cfg.hidden_dim, 2)
    
    def forward(self, afterstate: torch.Tensor, 
                empty_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            afterstate: (B, state_dim)
            empty_mask: (B, board_cells) bool - True for empty cells
        
        Returns:
            {
                'position_logits': (B, board_cells) - masked if empty_mask provided
                'value_logits': (B, 2) - logits for spawn value
                'position_probs': (B, board_cells) - softmax over valid positions
                'value_probs': (B, 2) - softmax over values
            }
        """
        h = self.net(afterstate)
        
        pos_logits = self.position_head(h)
        val_logits = self.value_head(h)
        
        # Mask invalid positions
        if empty_mask is not None:
            # Set non-empty positions to -inf
            pos_logits = pos_logits.masked_fill(~empty_mask, float('-inf'))
        
        # Check if all positions are masked (game over)
        all_masked = (pos_logits == float('-inf')).all(dim=-1, keepdim=True)
        pos_logits = pos_logits.masked_fill(all_masked, 0.0)  # Avoid NaN in softmax
        
        pos_probs = F.softmax(pos_logits, dim=-1)
        val_probs = F.softmax(val_logits, dim=-1)
        
        return {
            'position_logits': pos_logits,
            'value_logits': val_logits,
            'position_probs': pos_probs,
            'value_probs': val_probs,
        }
    
    def compute_entropy(self, afterstate: torch.Tensor,
                        empty_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute entropy of the chance distribution.
        
        For a well-trained model:
        - Othello: entropy ≈ 0 (deterministic)
        - 2048: entropy ≈ log(#empties) + H({0.9, 0.1})
        """
        out = self.forward(afterstate, empty_mask)
        
        # Position entropy
        pos_probs = out['position_probs']
        pos_entropy = -torch.sum(pos_probs * torch.log(pos_probs + 1e-10), dim=-1)
        
        # Value entropy
        val_probs = out['value_probs']
        val_entropy = -torch.sum(val_probs * torch.log(val_probs + 1e-10), dim=-1)
        
        return pos_entropy + val_entropy


class BoardDecoder(nn.Module):
    """
    Decode latent state → board prediction.
    
    This provides auxiliary supervision to ensure the latent state
    captures the board configuration correctly.
    
    Outputs logits for each cell's class.
    """
    
    def __init__(self, cfg: ModelConfig, use_style: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_style = use_style
        self.board_cells = cfg.board_size ** 2
        
        input_dim = cfg.state_dim + (cfg.style_dim if use_style else 0)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            ResBlock(cfg.hidden_dim, cfg.dropout),
            ResBlock(cfg.hidden_dim, cfg.dropout),
            nn.Linear(cfg.hidden_dim, self.board_cells * cfg.num_cell_classes),
        )
    
    def forward(self, state: torch.Tensor, 
                style: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns:
            logits: (B, board_cells, num_cell_classes)
        """
        if self.use_style and style is not None:
            x = torch.cat([state, style], dim=-1)
        else:
            x = state
        
        logits = self.net(x)
        return logits.reshape(-1, self.board_cells, self.cfg.num_cell_classes)


class PolicyHead(nn.Module):
    """Policy head for action probabilities."""
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.num_actions),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ValueHead(nn.Module):
    """Value head for state value estimation."""
    
    def __init__(self, cfg: ModelConfig, support_size: int = 51):
        super().__init__()
        self.support_size = support_size
        
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, support_size),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns categorical value distribution logits."""
        return self.net(state)


class CausalWorldModel(nn.Module):
    """
    Complete world model with causal core + stochastic envelope.
    
    Architecture:
        Observation → Encoder → (state, style)
        (state, action) → Dynamics → afterstate  [DETERMINISTIC RULE CORE]
        afterstate → Chance → distribution       [STOCHASTIC ENVELOPE]
        state → Decoder → board                  [AUXILIARY SUPERVISION]
        state → Policy/Value                     [RL HEADS]
    
    The key insight is that:
    - Dynamics should be DETERMINISTIC (same input → same output)
    - Chance captures IRREDUCIBLE randomness
    - The separation is testable via entropy metrics
    """
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Core components
        self.encoder = BoardEncoder(cfg)
        self.dynamics = DynamicsNetwork(cfg)
        self.chance = ChanceNetwork(cfg) if cfg.has_chance else None
        
        # Auxiliary decoders (for supervision)
        self.board_decoder = BoardDecoder(cfg, use_style=False)
        self.board_decoder_with_style = BoardDecoder(cfg, use_style=True)
        
        # RL heads (optional, for planning)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
    
    def initial_inference(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode observation into latent state.
        
        Args:
            obs: (B, C, H, W) observation tensor
        
        Returns:
            {
                'state': (B, state_dim),
                'style': (B, style_dim),
                'policy_logits': (B, num_actions),
                'value_logits': (B, support_size),
                'board_logits': (B, cells, classes),
            }
        """
        state, style = self.encoder(obs)
        
        return {
            'state': state,
            'style': style,
            'policy_logits': self.policy(state),
            'value_logits': self.value(state),
            'board_logits': self.board_decoder(state),
        }
    
    def recurrent_inference(self, state: torch.Tensor, action: torch.Tensor,
                           empty_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Apply dynamics and chance prediction.
        
        Args:
            state: (B, state_dim) current latent state
            action: (B,) action indices
            empty_mask: (B, cells) for masking chance (optional)
        
        Returns:
            {
                'afterstate': (B, state_dim) - DETERMINISTIC result
                'next_state': (B, state_dim) - same as afterstate if no chance
                'policy_logits': (B, num_actions),
                'value_logits': (B, support_size),
                'afterstate_board_logits': (B, cells, classes),
                
                # Chance outputs (if has_chance):
                'chance_position_logits': (B, cells),
                'chance_value_logits': (B, 2),
                'chance_entropy': (B,),
            }
        """
        # 1. DETERMINISTIC: Compute afterstate
        afterstate = self.dynamics(state, action)
        
        out = {
            'afterstate': afterstate,
            'next_state': afterstate,  # May be modified by chance
            'policy_logits': self.policy(afterstate),
            'value_logits': self.value(afterstate),
            'afterstate_board_logits': self.board_decoder(afterstate),
        }
        
        # 2. STOCHASTIC: Chance prediction (if applicable)
        if self.chance is not None:
            chance_out = self.chance(afterstate, empty_mask)
            out.update({
                'chance_position_logits': chance_out['position_logits'],
                'chance_value_logits': chance_out['value_logits'],
                'chance_position_probs': chance_out['position_probs'],
                'chance_value_probs': chance_out['value_probs'],
                'chance_entropy': self.chance.compute_entropy(afterstate, empty_mask),
            })
        
        return out
    
    def unroll(self, obs: torch.Tensor, actions: torch.Tensor,
               empty_masks: Optional[torch.Tensor] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Unroll dynamics for multiple steps.
        
        Args:
            obs: (B, C, H, W) initial observation
            actions: (B, T) action sequence
            empty_masks: (B, T, cells) empty masks per step (optional)
        
        Returns:
            List of T+1 outputs (initial + T steps)
        """
        B, T = actions.shape
        outputs = []
        
        # Initial inference
        init = self.initial_inference(obs)
        outputs.append(init)
        state = init['state']
        
        # Recurrent steps
        for t in range(T):
            mask = empty_masks[:, t] if empty_masks is not None else None
            step_out = self.recurrent_inference(state, actions[:, t], mask)
            outputs.append(step_out)
            state = step_out['afterstate']
        
        return outputs
    
    def compute_invariance_loss(self, obs1: torch.Tensor, obs2: torch.Tensor) -> torch.Tensor:
        """
        Compute invariance loss between two augmented views.
        
        The causal state should be INVARIANT to appearance changes,
        while the style captures the differences.
        
        This encourages the separation:
        - state: captures causal content (board configuration)
        - style: captures nuisance (rendering details)
        """
        state1, _ = self.encoder(obs1)
        state2, _ = self.encoder(obs2)
        
        # States should be similar
        return F.mse_loss(state1, state2)
    
    def compute_determinism_loss(self, state: torch.Tensor, action: torch.Tensor,
                                  n_samples: int = 3) -> torch.Tensor:
        """
        Encourage deterministic dynamics by penalizing variance.
        
        For the same (state, action), different forward passes should
        give the same afterstate. This loss penalizes stochasticity
        in the dynamics network itself.
        
        (Note: In standard PyTorch, this mainly affects dropout behavior)
        """
        afterstates = []
        for _ in range(n_samples):
            afterstates.append(self.dynamics(state, action))
        
        # Stack and compute variance
        stacked = torch.stack(afterstates, dim=0)  # (n_samples, B, state_dim)
        variance = torch.var(stacked, dim=0).mean()
        
        return variance


def create_model(game: str, **kwargs) -> CausalWorldModel:
    """
    Factory function to create model for a specific game.
    
    Args:
        game: "2048" or "othello"
        **kwargs: Override config parameters
    """
    if game.lower() == "2048":
        cfg = ModelConfig(
            board_size=4,
            num_cell_classes=17,  # 0=empty, 1-16=tile powers
            num_actions=4,
            has_chance=True,
            **kwargs
        )
    elif game.lower() == "othello":
        cfg = ModelConfig(
            board_size=8,
            num_cell_classes=3,  # empty, black, white
            num_actions=64,
            has_chance=False,  # Othello is deterministic
            **kwargs
        )
    else:
        raise ValueError(f"Unknown game: {game}")
    
    return CausalWorldModel(cfg)
