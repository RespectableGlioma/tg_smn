"""
Pixel-based World Model with Afterstate/Chance Separation

This unifies:
1. Pixel perception (CNN encoder, like ale_rssm_causal_stochastic_v2.py)
2. Afterstate/chance structure (like Stochastic MuZero)
3. Entropy-based rule detection (our contribution)

The key insight: Stochastic MuZero operates on pixels and learns rules implicitly.
We make the rule learning explicit by:
- Tracking entropy at each dynamics step
- Separating deterministic afterstates from stochastic chance outcomes
- Identifying compressible (low-entropy) transitions as "rules"

Architecture:
    Pixels → Encoder → s_t → Dynamics(s_t, a) → s_after → Chance(s_after) → s_{t+1}
                                    ↑                           ↑
                              DETERMINISTIC               STOCHASTIC
                              (rule core)                 (chance node)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PixelModelConfig:
    """Configuration for pixel-based world model."""
    # Image input
    in_channels: int = 1  # grayscale
    img_size: int = 64    # 64x64 like Atari preprocessing
    
    # Latent dimensions
    state_dim: int = 256       # causal state dimension
    afterstate_dim: int = 256  # afterstate dimension (can differ)
    hidden_dim: int = 512      # MLP hidden dimension
    
    # Action space
    n_actions: int = 4  # e.g., 4 directions for 2048
    
    # Chance modeling
    n_chance_outcomes: int = 32   # discretized chance outcomes
    chance_embedding_dim: int = 64
    
    # Architecture
    encoder_channels: List[int] = None  # CNN channels
    use_residual: bool = True
    
    # Training
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [32, 64, 128, 256]


class PixelEncoder(nn.Module):
    """
    CNN encoder: pixels → latent state
    
    Similar to ObsEncoder in ale_rssm_causal_stochastic_v2.py
    but outputs a structured state for dynamics.
    """
    
    def __init__(self, cfg: PixelModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Build CNN
        layers = []
        in_ch = cfg.in_channels
        for out_ch in cfg.encoder_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        
        # Compute flattened size
        # After 4 conv layers with stride 2: 64 → 32 → 16 → 8 → 4
        final_size = cfg.img_size // (2 ** len(cfg.encoder_channels))
        flat_dim = cfg.encoder_channels[-1] * final_size * final_size
        
        # Project to state
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.state_dim),
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, C, H, W] pixel observations in [0, 1]
        Returns:
            state: [B, state_dim] latent state
        """
        h = self.conv(obs)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


class DeterministicDynamics(nn.Module):
    """
    Deterministic dynamics: (state, action) → afterstate
    
    This is the RULE CORE. For a well-trained model on a game with
    deterministic rules, this should have very low output entropy
    (i.e., the afterstate is predictable given state and action).
    
    Key: We don't model uncertainty here - that's for the chance node.
    """
    
    def __init__(self, cfg: PixelModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Action embedding
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.hidden_dim)
        
        # Dynamics MLP
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.afterstate_dim),
        )
        
        # Optional residual connection if dimensions match
        self.use_residual = cfg.use_residual and (cfg.state_dim == cfg.afterstate_dim)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
            action: [B] action indices
        Returns:
            afterstate: [B, afterstate_dim]
        """
        a_emb = self.action_embed(action)
        h = torch.cat([state, a_emb], dim=-1)
        afterstate = self.net(h)
        
        if self.use_residual:
            afterstate = afterstate + state
            
        return afterstate


class ChanceOutcomeModel(nn.Module):
    """
    Chance outcome model: afterstate → distribution over next states
    
    This models the STOCHASTIC part of transitions (e.g., tile spawns in 2048,
    opponent responses, etc.)
    
    Two approaches:
    1. Categorical: predict discrete outcome, then embed
    2. Gaussian: predict mean/var of next state directly
    
    We use categorical for interpretability (can measure entropy directly).
    """
    
    def __init__(self, cfg: PixelModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Predict distribution over discrete chance outcomes
        self.outcome_logits = nn.Sequential(
            nn.Linear(cfg.afterstate_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.n_chance_outcomes),
        )
        
        # Embed chance outcome to state delta
        self.outcome_embed = nn.Embedding(cfg.n_chance_outcomes, cfg.state_dim)
        
        # Combine afterstate + outcome → next state
        self.state_combine = nn.Sequential(
            nn.Linear(cfg.afterstate_dim + cfg.state_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.state_dim),
        )
        
    def forward(
        self, 
        afterstate: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            afterstate: [B, afterstate_dim]
            sample: if True, sample outcome; if False, use argmax
            temperature: sampling temperature
            
        Returns:
            next_state: [B, state_dim]
            outcome_probs: [B, n_outcomes] probability distribution
            outcome_idx: [B] sampled/chosen outcome index
        """
        logits = self.outcome_logits(afterstate)
        probs = F.softmax(logits / temperature, dim=-1)
        
        if sample:
            outcome_idx = torch.multinomial(probs, 1).squeeze(-1)
        else:
            outcome_idx = probs.argmax(dim=-1)
            
        # Get outcome embedding
        outcome_emb = self.outcome_embed(outcome_idx)
        
        # Combine to get next state
        combined = torch.cat([afterstate, outcome_emb], dim=-1)
        next_state = self.state_combine(combined)
        
        return next_state, probs, outcome_idx
    
    def entropy(self, afterstate: torch.Tensor) -> torch.Tensor:
        """Compute entropy of chance distribution (bits)."""
        logits = self.outcome_logits(afterstate)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1) / math.log(2)  # convert to bits
        return entropy


class PixelDecoder(nn.Module):
    """
    Decode latent state back to pixels (auxiliary loss for representation learning).
    """
    
    def __init__(self, cfg: PixelModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Compute spatial size
        final_size = cfg.img_size // (2 ** len(cfg.encoder_channels))
        self.final_size = final_size
        self.init_channels = cfg.encoder_channels[-1]
        
        # Project state to spatial
        self.fc = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, self.init_channels * final_size * final_size),
            nn.ReLU(inplace=True),
        )
        
        # Transposed convolutions
        layers = []
        channels = list(reversed(cfg.encoder_channels))
        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
        # Final layer to image channels
        layers.append(nn.ConvTranspose2d(channels[-1], cfg.in_channels, 4, stride=2, padding=1))
        layers.append(nn.Sigmoid())  # output in [0, 1]
        
        self.deconv = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
        Returns:
            reconstruction: [B, C, H, W] in [0, 1]
        """
        h = self.fc(state)
        h = h.reshape(-1, self.init_channels, self.final_size, self.final_size)
        return self.deconv(h)


class PixelWorldModel(nn.Module):
    """
    Complete pixel-based world model with afterstate/chance separation.
    
    This is the unified architecture that:
    1. Encodes pixels to latent state (perception)
    2. Applies deterministic dynamics to get afterstate (rule core)
    3. Samples chance outcome to get next state (stochastic envelope)
    4. Can decode back to pixels (auxiliary supervision)
    
    The key diagnostic is comparing:
    - Dynamics entropy (should be ~0 for deterministic games)
    - Chance entropy (should match oracle for stochastic games)
    """
    
    def __init__(self, cfg: PixelModelConfig):
        super().__init__()
        self.cfg = cfg
        
        self.encoder = PixelEncoder(cfg)
        self.dynamics = DeterministicDynamics(cfg)
        self.chance = ChanceOutcomeModel(cfg)
        self.decoder = PixelDecoder(cfg)
        
        # Optional: policy and value heads for RL
        self.policy_head = nn.Linear(cfg.state_dim, cfg.n_actions)
        self.value_head = nn.Linear(cfg.state_dim, 1)
        
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode pixels to latent state."""
        return self.encoder(obs)
    
    def step(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        sample_chance: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        One step of world model dynamics.
        
        Args:
            state: [B, state_dim] current latent state
            action: [B] action indices
            sample_chance: whether to sample chance outcome
            
        Returns:
            dict with:
                - afterstate: [B, afterstate_dim]
                - next_state: [B, state_dim]
                - chance_probs: [B, n_outcomes]
                - chance_idx: [B]
                - chance_entropy: [B] entropy in bits
        """
        # Deterministic dynamics → afterstate
        afterstate = self.dynamics(state, action)
        
        # Chance model → next state
        next_state, chance_probs, chance_idx = self.chance(
            afterstate, sample=sample_chance
        )
        
        # Compute entropy
        chance_entropy = self.chance.entropy(afterstate)
        
        return {
            'afterstate': afterstate,
            'next_state': next_state,
            'chance_probs': chance_probs,
            'chance_idx': chance_idx,
            'chance_entropy': chance_entropy,
        }
    
    def decode(self, state: torch.Tensor) -> torch.Tensor:
        """Decode latent state to pixels."""
        return self.decoder(state)
    
    def policy(self, state: torch.Tensor) -> torch.Tensor:
        """Policy logits from state."""
        return self.policy_head(state)
    
    def value(self, state: torch.Tensor) -> torch.Tensor:
        """Value estimate from state."""
        return self.value_head(state).squeeze(-1)
    
    def imagine_trajectory(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        sample_chance: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine a trajectory in latent space.
        
        Args:
            initial_state: [B, state_dim]
            actions: [B, T] sequence of actions
            
        Returns:
            dict with sequences of states, afterstates, entropies, etc.
        """
        B, T = actions.shape
        
        states = [initial_state]
        afterstates = []
        chance_entropies = []
        chance_probs_list = []
        
        state = initial_state
        for t in range(T):
            result = self.step(state, actions[:, t], sample_chance=sample_chance)
            afterstates.append(result['afterstate'])
            states.append(result['next_state'])
            chance_entropies.append(result['chance_entropy'])
            chance_probs_list.append(result['chance_probs'])
            state = result['next_state']
            
        return {
            'states': torch.stack(states, dim=1),           # [B, T+1, state_dim]
            'afterstates': torch.stack(afterstates, dim=1), # [B, T, afterstate_dim]
            'chance_entropies': torch.stack(chance_entropies, dim=1),  # [B, T]
            'chance_probs': torch.stack(chance_probs_list, dim=1),     # [B, T, n_outcomes]
        }


# =============================================================================
# Loss Functions
# =============================================================================

def reconstruction_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """MSE reconstruction loss for pixel predictions."""
    return F.mse_loss(pred, target, reduction=reduction)


def dynamics_consistency_loss(
    model: PixelWorldModel,
    obs_sequence: torch.Tensor,
    action_sequence: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute losses for training the world model on observed trajectories.
    
    Args:
        model: PixelWorldModel
        obs_sequence: [B, T+1, C, H, W] observed pixel sequence
        action_sequence: [B, T] actions taken
        
    Returns:
        dict of losses:
            - recon_loss: reconstruction of all observed frames
            - consistency_loss: predicted next state matches encoded next obs
            - total_loss: weighted sum
    """
    B, Tp1, C, H, W = obs_sequence.shape
    T = Tp1 - 1
    
    # Encode all observations
    obs_flat = obs_sequence.reshape(B * Tp1, C, H, W)
    states_encoded = model.encode(obs_flat).reshape(B, Tp1, -1)
    
    # Reconstruction loss (can decode all states)
    recon = model.decode(states_encoded.reshape(B * Tp1, -1))
    recon_loss = reconstruction_loss(recon, obs_flat)
    
    # Dynamics consistency: predicted next state should match encoded next obs
    consistency_loss = 0.0
    chance_entropy_sum = 0.0
    
    for t in range(T):
        state_t = states_encoded[:, t]
        action_t = action_sequence[:, t]
        
        result = model.step(state_t, action_t, sample_chance=False)
        predicted_next = result['next_state']
        encoded_next = states_encoded[:, t + 1]
        
        # L2 consistency in latent space
        consistency_loss = consistency_loss + F.mse_loss(predicted_next, encoded_next)
        chance_entropy_sum = chance_entropy_sum + result['chance_entropy'].mean()
    
    consistency_loss = consistency_loss / T
    avg_chance_entropy = chance_entropy_sum / T
    
    return {
        'recon_loss': recon_loss,
        'consistency_loss': consistency_loss,
        'avg_chance_entropy': avg_chance_entropy,
        'total_loss': recon_loss + consistency_loss,
    }


# =============================================================================
# Entropy Analysis (for rule detection)
# =============================================================================

@torch.no_grad()
def analyze_entropy_distribution(
    model: PixelWorldModel,
    obs_sequence: torch.Tensor,
    action_sequence: torch.Tensor,
) -> Dict[str, float]:
    """
    Analyze the entropy distribution of the chance model across a trajectory.
    
    For a game with deterministic rules:
    - Most transitions should have LOW entropy (rule applies deterministically)
    - Only genuine chance events should have HIGH entropy
    
    For fully deterministic games (e.g., Othello):
    - ALL transitions should have near-zero entropy
    
    Returns statistics about the entropy distribution.
    """
    model.eval()
    
    B, Tp1, C, H, W = obs_sequence.shape
    T = Tp1 - 1
    
    # Encode observations
    obs_flat = obs_sequence.reshape(B * Tp1, C, H, W)
    states = model.encode(obs_flat).reshape(B, Tp1, -1)
    
    # Collect entropies
    entropies = []
    for t in range(T):
        state_t = states[:, t]
        action_t = action_sequence[:, t]
        result = model.step(state_t, action_t, sample_chance=False)
        entropies.append(result['chance_entropy'])
    
    entropies = torch.cat(entropies, dim=0).cpu().numpy()
    
    return {
        'mean_entropy': float(entropies.mean()),
        'std_entropy': float(entropies.std()),
        'min_entropy': float(entropies.min()),
        'max_entropy': float(entropies.max()),
        'median_entropy': float(sorted(entropies)[len(entropies)//2]),
        'p10_entropy': float(sorted(entropies)[len(entropies)//10]),
        'p90_entropy': float(sorted(entropies)[9*len(entropies)//10]),
        'frac_below_0.1': float((entropies < 0.1).mean()),
        'frac_below_0.5': float((entropies < 0.5).mean()),
        'frac_below_1.0': float((entropies < 1.0).mean()),
    }


def is_transition_deterministic(
    model: PixelWorldModel,
    state: torch.Tensor,
    action: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Check if a transition is deterministic (low entropy).
    
    Args:
        model: trained world model
        state: [B, state_dim]
        action: [B]
        threshold: entropy threshold in bits
        
    Returns:
        [B] boolean tensor, True if deterministic
    """
    result = model.step(state, action, sample_chance=False)
    return result['chance_entropy'] < threshold
