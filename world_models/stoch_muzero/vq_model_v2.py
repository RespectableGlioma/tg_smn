"""
VQ-VAE World Model v2 - with improved codebook utilization

Key improvements over v1:
1. Dead code reset - reinitialize unused codes from encoder outputs  
2. Codebook usage tracking and reporting
3. Lower EMA decay (0.95 vs 0.99) for faster adaptation
4. Separate deterministic dynamics from stochastic chance (explicit split)

The core insight remains: discrete codes + categorical distributions = learnable uncertainty.
No oracle needed - the model discovers stochasticity from data.

Usage:
    from world_models.stoch_muzero.vq_model_v2 import VQWorldModel, VQWorldModelConfig
    cfg = VQWorldModelConfig(n_actions=4, codebook_size=512)
    model = VQWorldModel(cfg)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQWorldModelConfig:
    """Configuration for VQ-VAE World Model."""
    # Image
    img_size: int = 64
    img_channels: int = 1
    
    # VQ-VAE
    codebook_size: int = 512
    code_dim: int = 64
    commitment_cost: float = 0.25
    
    # Encoder/Decoder
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)
    
    # Dynamics
    n_actions: int = 4
    hidden_dim: int = 256
    n_transformer_layers: int = 2
    n_heads: int = 4
    
    # Codebook management
    ema_decay: float = 0.99
    dead_code_threshold: int = 2      # Reset codes used fewer than N times
    reset_every: int = 100            # Check for dead codes every N steps


class VectorQuantizerV2(nn.Module):
    """
    Vector Quantization with:
    - EMA codebook updates
    - Dead code reset (reinitialize unused codes)
    - Usage tracking
    """
    
    def __init__(
        self, 
        n_codes: int, 
        code_dim: int, 
        commitment_cost: float = 0.25, 
        ema_decay: float = 0.99,
        dead_code_threshold: int = 2,
    ):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        
        # Codebook
        self.embedding = nn.Embedding(n_codes, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        
        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(n_codes))
        self.register_buffer('ema_w', torch.zeros(n_codes, code_dim))
        
        # Usage tracking (for dead code detection)
        self.register_buffer('code_usage', torch.zeros(n_codes))
        self.register_buffer('total_steps', torch.tensor(0))
        
    def forward(self, z: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [B, N, D] continuous latents
            
        Returns:
            z_q, indices, vq_loss, perplexity, codebook_usage
        """
        B, N, D = z.shape
        
        # Flatten for distance computation
        z_flat = z.reshape(-1, D)
        
        # Compute distances to all codes
        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True) +
            self.embedding.weight.pow(2).sum(dim=1) -
            2 * z_flat @ self.embedding.weight.t()
        )
        
        # Find nearest codes
        indices = d.argmin(dim=-1)
        
        # Get quantized vectors
        z_q_flat = self.embedding(indices)
        z_q = z_q_flat.reshape(B, N, D)
        indices = indices.reshape(B, N)
        
        # Track usage
        if training:
            with torch.no_grad():
                flat_indices = indices.flatten()
                self.code_usage.scatter_add_(
                    0, flat_indices, 
                    torch.ones_like(flat_indices, dtype=torch.float)
                )
                self.total_steps += 1
        
        # Compute losses
        if training:
            # One-hot encoding
            encodings = F.one_hot(indices.reshape(-1), self.n_codes).float()
            
            # EMA update cluster sizes
            cluster_size = encodings.sum(0)
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size, alpha=1 - self.ema_decay
            )
            
            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size.add_(1e-5).div_(
                n + self.n_codes * 1e-5
            ).mul_(n)
            
            # EMA update embeddings
            dw = encodings.t() @ z_flat
            self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
            
            self.embedding.weight.data.copy_(
                self.ema_w / self.ema_cluster_size.unsqueeze(1).clamp(min=1e-5)
            )
            
            # Losses
            commitment_loss = F.mse_loss(z, z_q.detach())
            codebook_loss = F.mse_loss(z_q, z.detach())
            vq_loss = commitment_loss * self.commitment_cost + codebook_loss
        else:
            vq_loss = torch.tensor(0.0, device=z.device)
        
        # Straight-through
        z_q = z + (z_q - z).detach()
        
        # Perplexity
        with torch.no_grad():
            encodings = F.one_hot(indices.reshape(-1), self.n_codes).float()
            avg_probs = encodings.mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
            # Codebook usage stats
            unique_codes = (self.code_usage > 0).sum().float()
            usage_ratio = unique_codes / self.n_codes
        
        return {
            'z_q': z_q,
            'indices': indices,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'codebook_usage': usage_ratio,
            'unique_codes': unique_codes,
        }
    
    def reset_dead_codes(self, z_samples: torch.Tensor) -> int:
        """
        Reset codes that have been used fewer than threshold times.
        
        Args:
            z_samples: [M, D] encoder outputs to use for reinitialization
            
        Returns:
            Number of codes reset
        """
        with torch.no_grad():
            dead_mask = self.code_usage < self.dead_code_threshold
            dead_indices = dead_mask.nonzero().squeeze(-1)
            
            if len(dead_indices) == 0:
                return 0
            
            n_dead = len(dead_indices)
            n_samples = len(z_samples)
            
            if n_samples == 0:
                return 0
            
            # Sample random encoder outputs to reinitialize dead codes
            # Add small noise for diversity
            perm = torch.randperm(n_samples, device=z_samples.device)
            n_reset = min(n_dead, n_samples)
            
            reset_values = z_samples[perm[:n_reset]]
            reset_values = reset_values + 0.01 * torch.randn_like(reset_values)
            
            self.embedding.weight.data[dead_indices[:n_reset]] = reset_values
            
            # Also reset EMA trackers for these codes
            self.ema_cluster_size[dead_indices[:n_reset]] = 1.0
            self.ema_w[dead_indices[:n_reset]] = reset_values
            
            # Reset usage counter
            self.code_usage.zero_()
            
            return n_reset
    
    def get_usage_stats(self) -> Dict[str, float]:
        """Get detailed codebook usage statistics."""
        with torch.no_grad():
            usage = self.code_usage.cpu().numpy()
            active = (usage > 0).sum()
            
            return {
                'active_codes': int(active),
                'total_codes': self.n_codes,
                'usage_ratio': float(active / self.n_codes),
                'max_usage': float(usage.max()),
                'mean_usage_active': float(usage[usage > 0].mean()) if active > 0 else 0,
            }


class VQEncoder(nn.Module):
    """CNN encoder: pixels → continuous latents for quantization."""
    
    def __init__(self, cfg: VQWorldModelConfig):
        super().__init__()
        
        channels = [cfg.img_channels] + list(cfg.encoder_channels)
        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
        self.conv = nn.Sequential(*layers)
        
        final_size = cfg.img_size // (2 ** len(cfg.encoder_channels))
        self.n_spatial = final_size * final_size
        self.proj = nn.Linear(cfg.encoder_channels[-1], cfg.code_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.conv(x)
        h = h.permute(0, 2, 3, 1)
        h = h.reshape(B, -1, h.shape[-1])
        z = self.proj(h)
        return z


class VQDecoder(nn.Module):
    """CNN decoder: quantized codes → pixels."""
    
    def __init__(self, cfg: VQWorldModelConfig):
        super().__init__()
        
        final_size = cfg.img_size // (2 ** len(cfg.encoder_channels))
        self.final_size = final_size
        self.init_channels = cfg.encoder_channels[-1]
        
        self.proj = nn.Linear(cfg.code_dim, self.init_channels)
        
        channels = list(reversed(cfg.encoder_channels)) + [cfg.img_channels]
        layers = []
        for i in range(len(channels) - 1):
            if i < len(channels) - 2:
                layers.extend([
                    nn.ConvTranspose2d(channels[i], channels[i+1], 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ])
            else:
                layers.append(
                    nn.ConvTranspose2d(channels[i], channels[i+1], 4, stride=2, padding=1)
                )
        self.deconv = nn.Sequential(*layers)
        
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        B, N, D = z_q.shape
        h = self.proj(z_q)
        h = h.reshape(B, self.final_size, self.final_size, -1)
        h = h.permute(0, 3, 1, 2)
        x_recon = self.deconv(h)
        return torch.sigmoid(x_recon)


class TransformerBlock(nn.Module):
    """Transformer block for code interactions."""
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class DeterministicDynamics(nn.Module):
    """
    Deterministic dynamics: (codes, action) → afterstate codes
    
    This learns THE RULES - purely deterministic transformations.
    """
    
    def __init__(self, cfg: VQWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.code_dim)
        
        self.transformer = nn.ModuleList([
            TransformerBlock(cfg.code_dim, cfg.n_heads)
            for _ in range(cfg.n_transformer_layers)
        ])
        
        self.out_proj = nn.Sequential(
            nn.Linear(cfg.code_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.code_dim),
        )
        
    def forward(self, z_q: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: [B, N, D] current codes
            action: [B] action indices
            
        Returns:
            afterstate: [B, N, D] deterministic result (pre-chance)
        """
        a_emb = self.action_embed(action)
        h = z_q + a_emb.unsqueeze(1)
        
        for block in self.transformer:
            h = block(h)
        
        delta = self.out_proj(h)
        afterstate = z_q + delta
        
        return afterstate


class StochasticChance(nn.Module):
    """
    Stochastic chance: afterstate → P(next codes)
    
    This models RANDOMNESS - outputs categorical distribution.
    Entropy emerges from data!
    """
    
    def __init__(self, cfg: VQWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        
        self.transformer = nn.ModuleList([
            TransformerBlock(cfg.code_dim, cfg.n_heads)
            for _ in range(cfg.n_transformer_layers)
        ])
        
        self.logits = nn.Linear(cfg.code_dim, cfg.codebook_size)
        
    def forward(self, afterstate: torch.Tensor, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Args:
            afterstate: [B, N, D] deterministic afterstate
            
        Returns:
            logits, probs, entropy (per-position and mean)
        """
        h = afterstate
        for block in self.transformer:
            h = block(h)
        
        logits = self.logits(h)
        logits_scaled = logits / temperature
        probs = F.softmax(logits_scaled, dim=-1)
        
        # Entropy per position
        log_probs = F.log_softmax(logits_scaled, dim=-1)
        entropy_per_pos = -(probs * log_probs).sum(dim=-1) / math.log(2)  # [B, N] in bits
        entropy_mean = entropy_per_pos.mean(dim=-1)  # [B] average entropy
        
        return {
            'logits': logits,
            'probs': probs,
            'entropy_per_position': entropy_per_pos,
            'entropy': entropy_mean,
        }
    
    def sample(self, afterstate: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample next code indices."""
        result = self.forward(afterstate, temperature)
        probs = result['probs']
        
        B, N, K = probs.shape
        indices = torch.multinomial(probs.reshape(B*N, K), 1).reshape(B, N)
        
        return indices, result['entropy']


class VQWorldModelV2(nn.Module):
    """
    VQ-VAE World Model with:
    - Explicit deterministic/stochastic factorization
    - Dead code reset for better codebook utilization
    - Entropy learned from data (no oracle)
    
    Architecture:
        Pixels → Encoder → z_continuous → VQ → z_discrete
                                                    ↓
                         (z_discrete, action) → Dynamics → afterstate
                                                              ↓
                                                    Chance → P(next_codes)
                                                              ↓
                                                    Entropy is DATA-DRIVEN!
    """
    
    def __init__(self, cfg: VQWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.step_count = 0
        
        self.encoder = VQEncoder(cfg)
        self.quantizer = VectorQuantizerV2(
            cfg.codebook_size, 
            cfg.code_dim,
            cfg.commitment_cost,
            cfg.ema_decay,
            cfg.dead_code_threshold,
        )
        self.decoder = VQDecoder(cfg)
        
        # Explicit factorization
        self.dynamics = DeterministicDynamics(cfg)  # RULES (deterministic)
        self.chance = StochasticChance(cfg)          # RANDOMNESS (stochastic)
        
    def encode(self, x: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """Encode image to discrete codes."""
        z = self.encoder(x)
        vq_result = self.quantizer(z, training=training)
        vq_result['z_continuous'] = z  # Keep for dead code reset
        return vq_result
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode codes to image."""
        return self.decoder(z_q)
    
    def step(
        self, 
        z_q: torch.Tensor, 
        action: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        World model step with explicit factorization.
        
        Returns:
            afterstate: deterministic result of action (THE RULES)
            next_z_q: final state after chance
            entropy: uncertainty (learned from data!)
            entropy_per_position: per-code-position entropy
        """
        # Deterministic dynamics (rules)
        afterstate = self.dynamics(z_q, action)
        
        # Stochastic chance
        chance_result = self.chance(afterstate, temperature)
        
        if sample:
            next_indices, entropy = self.chance.sample(afterstate, temperature)
        else:
            next_indices = chance_result['logits'].argmax(dim=-1)
            entropy = chance_result['entropy']
        
        next_z_q = self.quantizer.embedding(next_indices)
        
        return {
            'afterstate': afterstate,
            'next_z_q': next_z_q,
            'next_indices': next_indices,
            'entropy': entropy,
            'entropy_per_position': chance_result['entropy_per_position'],
            'logits': chance_result['logits'],
            'probs': chance_result['probs'],
        }
    
    def compute_loss(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
        unroll_steps: int = 5,
        w_recon: float = 1.0,
        w_vq: float = 1.0,
        w_transition: float = 1.0,
        reset_dead_codes: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with optional dead code reset.
        """
        B, Tp1, C, H, W = obs_batch.shape
        T = Tp1 - 1
        unroll = min(unroll_steps, T)
        
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        total_transition_loss = 0.0
        total_entropy = 0.0
        
        # Encode all observations
        obs_flat = obs_batch.reshape(B * Tp1, C, H, W)
        enc_result = self.encode(obs_flat, training=True)
        
        all_z_q = enc_result['z_q'].reshape(B, Tp1, -1, self.cfg.code_dim)
        all_indices = enc_result['indices'].reshape(B, Tp1, -1)
        z_continuous = enc_result['z_continuous']  # For dead code reset
        
        total_vq_loss = enc_result['vq_loss']
        
        # Reconstruction loss
        recon = self.decode(enc_result['z_q'])
        recon_loss = F.mse_loss(recon, obs_flat)
        total_recon_loss = recon_loss
        
        # Transition loss
        for t in range(unroll):
            z_q_t = all_z_q[:, t]
            action_t = action_batch[:, t]
            target_indices = all_indices[:, t + 1]
            
            step_result = self.step(z_q_t, action_t, sample=False)
            
            logits = step_result['logits']
            B_t, N_t, K = logits.shape
            
            transition_loss_t = F.cross_entropy(
                logits.reshape(B_t * N_t, K),
                target_indices.reshape(B_t * N_t),
            )
            
            total_transition_loss = total_transition_loss + transition_loss_t
            total_entropy = total_entropy + step_result['entropy'].mean()
        
        n_steps = float(unroll)
        avg_transition_loss = total_transition_loss / n_steps
        avg_entropy = total_entropy / n_steps
        
        # Dead code reset
        self.step_count += 1
        n_reset = 0
        if reset_dead_codes and self.step_count % self.cfg.reset_every == 0:
            z_flat = z_continuous.reshape(-1, self.cfg.code_dim)
            perm = torch.randperm(len(z_flat))[:256]  # Sample subset
            n_reset = self.quantizer.reset_dead_codes(z_flat[perm])
        
        total_loss = (
            w_recon * total_recon_loss +
            w_vq * total_vq_loss +
            w_transition * avg_transition_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': total_recon_loss,
            'vq_loss': total_vq_loss,
            'transition_loss': avg_transition_loss,
            'entropy': avg_entropy,
            'perplexity': enc_result['perplexity'],
            'codebook_usage': enc_result['codebook_usage'],
            'unique_codes': enc_result['unique_codes'],
            'codes_reset': torch.tensor(n_reset, dtype=torch.float),
        }
    
    def get_codebook_stats(self) -> Dict[str, float]:
        """Get detailed codebook statistics."""
        return self.quantizer.get_usage_stats()
    
    # =========================================================================
    # Rule extraction methods (for interpretability)
    # =========================================================================
    
    def extract_transition_table(
        self, 
        obs_batch: torch.Tensor, 
        action_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract empirical transition statistics.
        
        For each (code, action) pair, tracks:
        - What next codes occurred
        - How often each transition happened
        - Entropy of the transition distribution
        
        This reveals which transitions are deterministic (rules) vs stochastic (chance).
        """
        self.eval()
        
        # Transition counts: [n_codes, n_actions, n_codes]
        n_codes = self.cfg.codebook_size
        n_actions = self.cfg.n_actions
        transition_counts = torch.zeros(n_codes, n_actions, n_codes, device=obs_batch.device)
        
        B, Tp1, C, H, W = obs_batch.shape
        T = Tp1 - 1
        
        with torch.no_grad():
            # Encode all observations
            obs_flat = obs_batch.reshape(B * Tp1, C, H, W)
            enc_result = self.encode(obs_flat, training=False)
            all_indices = enc_result['indices'].reshape(B, Tp1, -1)  # [B, T+1, N]
            
            # Count transitions
            for t in range(T):
                curr_codes = all_indices[:, t]      # [B, N]
                next_codes = all_indices[:, t + 1]  # [B, N]
                actions = action_batch[:, t]        # [B]
                
                # For each spatial position
                for pos in range(curr_codes.shape[1]):
                    for b in range(B):
                        c_curr = curr_codes[b, pos].item()
                        c_next = next_codes[b, pos].item()
                        a = actions[b].item()
                        transition_counts[c_curr, a, c_next] += 1
        
        # Compute transition probabilities and entropy
        counts_sum = transition_counts.sum(dim=-1, keepdim=True).clamp(min=1)
        transition_probs = transition_counts / counts_sum
        
        # Entropy per (code, action) pair
        log_probs = torch.log(transition_probs + 1e-10)
        entropy = -(transition_probs * log_probs).sum(dim=-1) / math.log(2)  # bits
        
        # Identify deterministic transitions (low entropy)
        deterministic_mask = entropy < 0.1
        
        self.train()
        
        return {
            'transition_counts': transition_counts,
            'transition_probs': transition_probs,
            'transition_entropy': entropy,
            'deterministic_mask': deterministic_mask,
            'n_deterministic': deterministic_mask.sum().item(),
            'n_stochastic': (~deterministic_mask & (counts_sum.squeeze(-1) > 0)).sum().item(),
        }


def create_vq_world_model_v2(
    img_size: int = 64,
    n_actions: int = 4,
    codebook_size: int = 512,
    code_dim: int = 64,
) -> VQWorldModelV2:
    """Create a VQ-VAE world model with default settings."""
    cfg = VQWorldModelConfig(
        img_size=img_size,
        n_actions=n_actions,
        codebook_size=codebook_size,
        code_dim=code_dim,
    )
    return VQWorldModelV2(cfg)


# =============================================================================
# Testing
# =============================================================================

def _test_vq_world_model_v2():
    print("Testing VQWorldModelV2...")
    
    cfg = VQWorldModelConfig(
        img_size=64,
        n_actions=4,
        codebook_size=256,
        code_dim=32,
    )
    model = VQWorldModelV2(cfg)
    
    # Test encode
    x = torch.randn(2, 1, 64, 64)
    enc = model.encode(x)
    print(f"  Encoded: z_q={enc['z_q'].shape}, usage={enc['codebook_usage'].item():.2%}")
    
    # Test step with explicit factorization
    action = torch.randint(0, 4, (2,))
    step = model.step(enc['z_q'], action)
    print(f"  Step: afterstate={step['afterstate'].shape}")
    print(f"        entropy={step['entropy'].mean().item():.2f} bits")
    
    # Test loss
    obs_batch = torch.randn(2, 6, 1, 64, 64)
    action_batch = torch.randint(0, 4, (2, 5))
    losses = model.compute_loss(obs_batch, action_batch, unroll_steps=3)
    print(f"  Loss: {losses['total_loss'].item():.4f}")
    print(f"  Codebook usage: {losses['codebook_usage'].item():.2%}")
    
    print("VQWorldModelV2 test passed!")


if __name__ == '__main__':
    _test_vq_world_model_v2()
