"""
VQ-VAE World Model for Causal Structure Learning

This implements a discrete latent world model inspired by:
- VQ-VAE (van den Oord et al.)
- Dreamer v2/v3 (Hafner et al.) - categorical latents
- Stochastic MuZero - afterstate/chance separation

Key insight: Discrete codes make entropy LEARNABLE FROM DATA.
- Same (code, action) → same next_code = deterministic (entropy → 0)
- Same (code, action) → varied next_codes = stochastic (entropy > 0)

No oracle needed - the model discovers stochasticity from transition statistics.

Architecture:
    
    Pixels ─→ Encoder ─→ z_continuous ─→ Quantize ─→ z_q (codebook indices)
                                              │
                                              ↓
                         ┌────────────────────┴───────────────────┐
                         │                                        │
                    Dynamics(z_q, a)                          Decoder(z_q)
                         │                                        │
                         ↓                                        ↓
                   afterstate_codes                         reconstructed
                         │                                    pixels
                         ↓
                   Chance(afterstate)
                         │
                         ↓
              P(next_codes) ← CATEGORICAL DISTRIBUTION
                         │
              ┌──────────┴──────────┐
              │                     │
         Cross-entropy         Entropy is
         with actual z'        DATA-DRIVEN!
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
    codebook_size: int = 512      # Number of codes in codebook
    code_dim: int = 64            # Dimension of each code vector
    n_codes_per_image: int = 16   # Spatial codes (4x4 grid for 64x64 image)
    commitment_cost: float = 0.25 # VQ commitment loss weight
    
    # Encoder/Decoder
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)
    
    # Dynamics
    n_actions: int = 4
    hidden_dim: int = 256
    n_transformer_layers: int = 2  # For modeling code interactions
    n_heads: int = 4
    
    # Training
    ema_decay: float = 0.99       # EMA for codebook updates


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with EMA codebook updates.
    
    Maps continuous vectors to nearest codebook entries.
    Uses straight-through estimator for gradients.
    """
    
    def __init__(self, n_codes: int, code_dim: int, commitment_cost: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        
        # Codebook
        self.embedding = nn.Embedding(n_codes, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        
        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(n_codes))
        self.register_buffer('ema_w', torch.zeros(n_codes, code_dim))
        self.register_buffer('initialized', torch.tensor(False))
        
    def forward(self, z: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [B, N, D] continuous latents (N = spatial positions, D = code_dim)
            
        Returns:
            z_q: [B, N, D] quantized latents
            indices: [B, N] codebook indices
            vq_loss: scalar commitment + codebook loss
            perplexity: codebook usage metric
        """
        B, N, D = z.shape
        
        # Flatten for distance computation
        z_flat = z.reshape(-1, D)  # [B*N, D]
        
        # Compute distances to all codes
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True) +
            self.embedding.weight.pow(2).sum(dim=1) -
            2 * z_flat @ self.embedding.weight.t()
        )  # [B*N, n_codes]
        
        # Find nearest codes
        indices = d.argmin(dim=-1)  # [B*N]
        
        # Get quantized vectors
        z_q_flat = self.embedding(indices)  # [B*N, D]
        z_q = z_q_flat.reshape(B, N, D)
        indices = indices.reshape(B, N)
        
        # Compute losses
        if training:
            # EMA codebook update (no gradient needed)
            with torch.no_grad():
                # One-hot encoding of assignments
                encodings = F.one_hot(indices.reshape(-1), self.n_codes).float()  # [B*N, n_codes]
                
                # Update cluster sizes
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    encodings.sum(0), alpha=1 - self.ema_decay
                )
                
                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                self.ema_cluster_size.add_(1e-5).div_(
                    n + self.n_codes * 1e-5
                ).mul_(n)
                
                # Update codebook
                dw = encodings.t() @ z_flat  # [n_codes, D]
                self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
                
                self.embedding.weight.data.copy_(
                    self.ema_w / self.ema_cluster_size.unsqueeze(1)
                )
            
            # Commitment loss (encoder should commit to codes)
            commitment_loss = F.mse_loss(z, z_q.detach())
            
            # Codebook loss (EMA handles this, but keep for non-EMA option)
            codebook_loss = F.mse_loss(z_q, z.detach())
            
            vq_loss = commitment_loss * self.commitment_cost + codebook_loss
        else:
            vq_loss = torch.tensor(0.0, device=z.device)
        
        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()
        
        # Compute perplexity (codebook usage)
        with torch.no_grad():
            encodings = F.one_hot(indices.reshape(-1), self.n_codes).float()
            avg_probs = encodings.mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return {
            'z_q': z_q,
            'indices': indices,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
        }


class VQEncoder(nn.Module):
    """CNN encoder that outputs continuous vectors for quantization."""
    
    def __init__(self, cfg: VQWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Convolutional layers
        channels = [cfg.img_channels] + list(cfg.encoder_channels)
        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
        self.conv = nn.Sequential(*layers)
        
        # Final projection to code dimension
        # After 4 conv layers with stride 2: 64 → 4x4
        final_size = cfg.img_size // (2 ** len(cfg.encoder_channels))
        self.n_spatial = final_size * final_size
        
        self.proj = nn.Linear(cfg.encoder_channels[-1], cfg.code_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] images
            
        Returns:
            z: [B, N, D] continuous latents for quantization
        """
        B = x.shape[0]
        h = self.conv(x)  # [B, C_last, H', W']
        h = h.permute(0, 2, 3, 1)  # [B, H', W', C]
        h = h.reshape(B, -1, h.shape[-1])  # [B, N, C]
        z = self.proj(h)  # [B, N, code_dim]
        return z


class VQDecoder(nn.Module):
    """CNN decoder that reconstructs images from discrete codes."""
    
    def __init__(self, cfg: VQWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        
        final_size = cfg.img_size // (2 ** len(cfg.encoder_channels))
        self.final_size = final_size
        self.init_channels = cfg.encoder_channels[-1]
        
        # Project from codes to spatial features
        self.proj = nn.Linear(cfg.code_dim, self.init_channels)
        
        # Deconvolutional layers
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
        """
        Args:
            z_q: [B, N, D] quantized latents
            
        Returns:
            x_recon: [B, C, H, W] reconstructed images
        """
        B, N, D = z_q.shape
        h = self.proj(z_q)  # [B, N, C]
        h = h.reshape(B, self.final_size, self.final_size, -1)
        h = h.permute(0, 3, 1, 2)  # [B, C, H', W']
        x_recon = self.deconv(h)
        return torch.sigmoid(x_recon)


class TransformerBlock(nn.Module):
    """Transformer block for modeling interactions between codes."""
    
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
        # Self-attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        
        # Feed-forward
        x = x + self.ff(self.ln2(x))
        return x


class DynamicsNetwork(nn.Module):
    """
    Dynamics network: (codes, action) → afterstate codes
    
    This is the DETERMINISTIC part - learns the rules.
    Uses Transformer to model interactions between spatial codes.
    """
    
    def __init__(self, cfg: VQWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Action embedding
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.code_dim)
        
        # Transformer for code interactions
        self.transformer = nn.ModuleList([
            TransformerBlock(cfg.code_dim, cfg.n_heads)
            for _ in range(cfg.n_transformer_layers)
        ])
        
        # Output projection (residual style)
        self.out_proj = nn.Sequential(
            nn.Linear(cfg.code_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.code_dim),
        )
        
    def forward(self, z_q: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: [B, N, D] quantized codes
            action: [B] action indices
            
        Returns:
            afterstate: [B, N, D] afterstate codes (continuous, will be quantized)
        """
        B, N, D = z_q.shape
        
        # Add action embedding to all positions
        a_emb = self.action_embed(action)  # [B, D]
        h = z_q + a_emb.unsqueeze(1)  # [B, N, D]
        
        # Transformer layers
        for block in self.transformer:
            h = block(h)
        
        # Residual output
        delta = self.out_proj(h)
        afterstate = z_q + delta
        
        return afterstate


class ChanceNetwork(nn.Module):
    """
    Chance network: afterstate → P(next codes)
    
    This is the STOCHASTIC part - outputs a CATEGORICAL distribution
    over codebook indices for each spatial position.
    
    ENTROPY IS LEARNED FROM DATA:
    - If transitions are deterministic, the distribution will be peaked (low entropy)
    - If transitions are stochastic, the distribution will be spread (high entropy)
    """
    
    def __init__(self, cfg: VQWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Transformer for code interactions
        self.transformer = nn.ModuleList([
            TransformerBlock(cfg.code_dim, cfg.n_heads)
            for _ in range(cfg.n_transformer_layers)
        ])
        
        # Output logits over codebook
        self.logits = nn.Linear(cfg.code_dim, cfg.codebook_size)
        
    def forward(
        self, 
        afterstate: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            afterstate: [B, N, D] afterstate codes
            temperature: sampling temperature
            
        Returns:
            logits: [B, N, codebook_size] log-probabilities
            probs: [B, N, codebook_size] probabilities
            entropy: [B] entropy in bits (averaged over positions)
        """
        B, N, D = afterstate.shape
        
        h = afterstate
        for block in self.transformer:
            h = block(h)
        
        logits = self.logits(h)  # [B, N, codebook_size]
        
        # Apply temperature
        logits_scaled = logits / temperature
        probs = F.softmax(logits_scaled, dim=-1)
        
        # Compute entropy per position, then average
        log_probs = F.log_softmax(logits_scaled, dim=-1)
        entropy_per_pos = -(probs * log_probs).sum(dim=-1)  # [B, N] in nats
        entropy_bits = entropy_per_pos.mean(dim=-1) / math.log(2)  # [B] in bits
        
        return {
            'logits': logits,
            'probs': probs,
            'entropy': entropy_bits,
        }
    
    def sample(
        self, 
        afterstate: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample next code indices from the distribution."""
        result = self.forward(afterstate, temperature)
        probs = result['probs']  # [B, N, codebook_size]
        
        B, N, K = probs.shape
        indices = torch.multinomial(probs.reshape(B*N, K), 1).reshape(B, N)
        
        return indices, result['entropy']


class VQWorldModel(nn.Module):
    """
    VQ-VAE World Model with learned stochasticity.
    
    The key insight: by using discrete codes and categorical transition distributions,
    entropy emerges naturally from the data:
    
    - Deterministic transitions → peaked distribution → low entropy
    - Stochastic transitions → spread distribution → high entropy
    
    No oracle or hand-tuned entropy bonuses needed!
    """
    
    def __init__(self, cfg: VQWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        
        self.encoder = VQEncoder(cfg)
        self.quantizer = VectorQuantizer(
            cfg.codebook_size, 
            cfg.code_dim,
            cfg.commitment_cost,
            cfg.ema_decay,
        )
        self.decoder = VQDecoder(cfg)
        self.dynamics = DynamicsNetwork(cfg)
        self.chance = ChanceNetwork(cfg)
        
    def encode(self, x: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """Encode image to discrete codes."""
        z = self.encoder(x)  # [B, N, D] continuous
        vq_result = self.quantizer(z, training=training)
        return {
            'z': z,
            'z_q': vq_result['z_q'],
            'indices': vq_result['indices'],
            'vq_loss': vq_result['vq_loss'],
            'perplexity': vq_result['perplexity'],
        }
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized codes to image."""
        return self.decoder(z_q)
    
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode code indices directly to image."""
        z_q = self.quantizer.embedding(indices)  # [B, N, D]
        return self.decode(z_q)
    
    def step(
        self, 
        z_q: torch.Tensor, 
        action: torch.Tensor,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Take a world model step.
        
        Args:
            z_q: [B, N, D] current quantized codes
            action: [B] action indices
            sample: if True, sample next codes; if False, use argmax
            temperature: sampling temperature
            
        Returns:
            afterstate: [B, N, D] deterministic result of action
            next_z_q: [B, N, D] next state codes (after chance)
            next_indices: [B, N] next code indices
            entropy: [B] entropy of chance distribution (bits)
            logits: [B, N, codebook_size] transition logits
        """
        # Deterministic dynamics
        afterstate = self.dynamics(z_q, action)  # [B, N, D]
        
        # Stochastic chance
        chance_result = self.chance(afterstate, temperature)
        
        if sample:
            next_indices, entropy = self.chance.sample(afterstate, temperature)
        else:
            next_indices = chance_result['logits'].argmax(dim=-1)
            entropy = chance_result['entropy']
        
        # Get quantized codes for next state
        next_z_q = self.quantizer.embedding(next_indices)
        
        return {
            'afterstate': afterstate,
            'next_z_q': next_z_q,
            'next_indices': next_indices,
            'entropy': entropy,
            'logits': chance_result['logits'],
            'probs': chance_result['probs'],
        }
    
    def compute_loss(
        self,
        obs_batch: torch.Tensor,      # [B, T+1, C, H, W]
        action_batch: torch.Tensor,   # [B, T]
        unroll_steps: int = 5,
        w_recon: float = 1.0,
        w_vq: float = 1.0,
        w_transition: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Losses:
        1. Reconstruction: decoder should reconstruct images from codes
        2. VQ: commitment loss for encoder
        3. Transition: cross-entropy between predicted and actual next codes
        
        The transition loss AUTOMATICALLY learns calibrated uncertainty:
        - If same (code, action) always leads to same next_code, CE is minimized by peaked distribution
        - If same (code, action) leads to varied next_codes, CE is minimized by spread distribution
        """
        B, Tp1, C, H, W = obs_batch.shape
        T = Tp1 - 1
        unroll = min(unroll_steps, T)
        
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        total_transition_loss = 0.0
        total_entropy = 0.0
        total_perplexity = 0.0
        
        # Encode all observations
        obs_flat = obs_batch.reshape(B * Tp1, C, H, W)
        enc_result = self.encode(obs_flat, training=True)
        
        all_z_q = enc_result['z_q'].reshape(B, Tp1, -1, self.cfg.code_dim)  # [B, T+1, N, D]
        all_indices = enc_result['indices'].reshape(B, Tp1, -1)  # [B, T+1, N]
        
        total_vq_loss = enc_result['vq_loss']
        total_perplexity = enc_result['perplexity']
        
        # Reconstruction loss
        recon = self.decode(enc_result['z_q'])  # [B*(T+1), C, H, W]
        recon_loss = F.mse_loss(recon, obs_flat)
        total_recon_loss = recon_loss
        
        # Transition loss: predict next codes from current codes + action
        for t in range(unroll):
            z_q_t = all_z_q[:, t]  # [B, N, D]
            action_t = action_batch[:, t]  # [B]
            target_indices = all_indices[:, t + 1]  # [B, N]
            
            step_result = self.step(z_q_t, action_t, sample=False)
            
            # Cross-entropy loss for each position
            logits = step_result['logits']  # [B, N, codebook_size]
            B_t, N_t, K = logits.shape
            
            transition_loss_t = F.cross_entropy(
                logits.reshape(B_t * N_t, K),
                target_indices.reshape(B_t * N_t),
            )
            
            total_transition_loss = total_transition_loss + transition_loss_t
            total_entropy = total_entropy + step_result['entropy'].mean()
        
        # Average over steps
        n_steps = float(unroll)
        avg_transition_loss = total_transition_loss / n_steps
        avg_entropy = total_entropy / n_steps
        
        # Total loss
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
            'perplexity': total_perplexity,
        }


def create_vq_world_model(
    img_size: int = 64,
    n_actions: int = 4,
    codebook_size: int = 512,
    code_dim: int = 64,
) -> VQWorldModel:
    """Create a VQ-VAE world model with default settings."""
    cfg = VQWorldModelConfig(
        img_size=img_size,
        n_actions=n_actions,
        codebook_size=codebook_size,
        code_dim=code_dim,
    )
    return VQWorldModel(cfg)


# =============================================================================
# Testing
# =============================================================================

def _test_vq_world_model():
    """Quick test that the model runs."""
    print("Testing VQWorldModel...")
    
    cfg = VQWorldModelConfig(
        img_size=64,
        n_actions=4,
        codebook_size=256,
        code_dim=32,
    )
    model = VQWorldModel(cfg)
    
    # Test encode
    x = torch.randn(2, 1, 64, 64)
    enc = model.encode(x)
    print(f"  Encoded: z_q={enc['z_q'].shape}, indices={enc['indices'].shape}")
    print(f"  VQ loss: {enc['vq_loss'].item():.4f}, perplexity: {enc['perplexity'].item():.1f}")
    
    # Test decode
    recon = model.decode(enc['z_q'])
    print(f"  Decoded: {recon.shape}")
    
    # Test step
    action = torch.randint(0, 4, (2,))
    step = model.step(enc['z_q'], action)
    print(f"  Step: next_z_q={step['next_z_q'].shape}, entropy={step['entropy'].mean().item():.2f} bits")
    
    # Test loss computation
    obs_batch = torch.randn(2, 6, 1, 64, 64)  # 5 transitions
    action_batch = torch.randint(0, 4, (2, 5))
    losses = model.compute_loss(obs_batch, action_batch, unroll_steps=3)
    print(f"  Loss: total={losses['total_loss'].item():.4f}")
    print(f"    recon={losses['recon_loss'].item():.4f}")
    print(f"    vq={losses['vq_loss'].item():.4f}")  
    print(f"    transition={losses['transition_loss'].item():.4f}")
    print(f"    entropy={losses['entropy'].item():.2f} bits")
    
    print("VQWorldModel test passed!")


if __name__ == '__main__':
    _test_vq_world_model()
