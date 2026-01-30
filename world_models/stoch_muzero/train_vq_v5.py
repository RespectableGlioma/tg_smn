"""
VQ-VAE World Model v5 - JEPA-inspired (Predictive Latent Learning)

Key insight from user: Pixel reconstruction encourages codes to capture 
visual appearance, not predictively useful structure. Cell-wise encoding
is a hack that assumes we know the grid structure.

Solution: Train primarily on PREDICTION in latent space, not reconstruction.

Architecture:
  - Encoder: image → continuous latent z
  - VQ: z → discrete codes
  - Dynamics: (codes_t, action) → predicted codes_{t+1}
  - Target encoder (EMA): provides stable prediction targets

Training signals:
  1. Transition prediction (MAIN): predict next state's codes
  2. Code diversity: entropy regularization on code usage
  3. Variance preservation: prevent encoder collapse (VICReg-style)
  4. Light reconstruction (optional, heavily downweighted)

NO cell-wise hack - full image encoding with learned spatial structure.

This is closer to how JEPA/I-JEPA work:
  - Don't reconstruct pixels
  - Predict in latent space
  - Use EMA target encoder for stable targets
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional
from collections import defaultdict
import copy


# ============================================================
# VQ Layer with entropy regularization
# ============================================================

class VectorQuantizerWithEntropy(nn.Module):
    """VQ with explicit entropy regularization to encourage code diversity."""
    
    def __init__(self, num_codes: int, code_dim: int, 
                 commitment_cost: float = 0.25,
                 ema_decay: float = 0.99):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        
        # Initialize codebook
        self.register_buffer('embed', torch.randn(num_codes, code_dim))
        self.register_buffer('cluster_size', torch.ones(num_codes))
        self.register_buffer('embed_avg', self.embed.clone())
        
        # For tracking
        self.register_buffer('code_count', torch.zeros(num_codes))
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z: [..., D] continuous latents
        Returns:
            z_q: [..., D] quantized
            codes: [...] indices
            info: dict with losses and stats
        """
        orig_shape = z.shape
        flat_z = z.reshape(-1, self.code_dim)
        
        # Compute distances
        dist = (flat_z.pow(2).sum(1, keepdim=True) 
                - 2 * flat_z @ self.embed.t()
                + self.embed.pow(2).sum(1, keepdim=True).t())
        
        # Get nearest codes
        _, codes_flat = dist.min(1)
        
        # Quantize
        z_q_flat = F.embedding(codes_flat, self.embed)
        
        # EMA updates
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(codes_flat, self.num_codes).float()
                
                self.cluster_size.mul_(self.ema_decay).add_(
                    encodings.sum(0), alpha=1 - self.ema_decay
                )
                
                embed_sum = encodings.t() @ flat_z
                self.embed_avg.mul_(self.ema_decay).add_(
                    embed_sum, alpha=1 - self.ema_decay
                )
                
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + 1e-5)
                    / (n + self.num_codes * 1e-5) * n
                )
                
                self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))
                
                # Track usage
                self.code_count.add_(encodings.sum(0))
        
        # Commitment loss
        commitment_loss = F.mse_loss(flat_z, z_q_flat.detach())
        
        # Straight-through
        z_q_flat = flat_z + (z_q_flat - flat_z).detach()
        
        # Code entropy (for regularization)
        # Higher entropy = more diverse code usage = better
        with torch.no_grad():
            encodings = F.one_hot(codes_flat, self.num_codes).float()
            avg_probs = encodings.mean(0)
            code_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
            max_entropy = np.log(self.num_codes)
            normalized_entropy = code_entropy / max_entropy
        
        # Reshape outputs
        z_q = z_q_flat.reshape(orig_shape)
        codes = codes_flat.reshape(orig_shape[:-1])
        
        info = {
            'commitment_loss': commitment_loss,
            'code_entropy': code_entropy,
            'normalized_entropy': normalized_entropy,
            'avg_probs': avg_probs,
        }
        
        return z_q, codes, info
    
    def get_usage_stats(self) -> dict:
        used = (self.code_count > 0).sum().item()
        total_assigned = self.code_count.sum().item()
        
        if total_assigned > 0:
            probs = self.code_count / total_assigned
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        else:
            entropy = 0
            
        return {
            'active_codes': used,
            'total_codes': self.num_codes,
            'usage_ratio': used / self.num_codes,
            'entropy_bits': entropy / np.log(2),
        }
    
    def reset_counts(self):
        self.code_count.zero_()


# ============================================================
# Encoder / Decoder
# ============================================================

class Encoder(nn.Module):
    """CNN encoder: image -> spatial latent grid."""
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, 
                 latent_dim: int = 64, output_grid: int = 4):
        super().__init__()
        self.output_grid = output_grid
        
        # Compute number of downsampling steps needed
        # 64 -> 32 -> 16 -> 8 -> 4 (4 steps for grid=4)
        # 64 -> 32 -> 16 -> 8 (3 steps for grid=8)
        
        layers = []
        in_ch = in_channels
        out_ch = hidden_dim
        
        # Progressive downsampling
        n_downsample = int(np.log2(64 / output_grid))
        for i in range(n_downsample):
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
            if i < n_downsample - 1:
                out_ch = min(out_ch * 2, 256)
        
        # Final projection to latent dim
        layers.append(nn.Conv2d(in_ch, latent_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, H, W] -> [B, latent_dim, grid, grid]"""
        return self.net(x)


class Decoder(nn.Module):
    """CNN decoder: spatial latent grid -> image."""
    def __init__(self, out_channels: int = 1, hidden_dim: int = 64,
                 latent_dim: int = 64, input_grid: int = 4):
        super().__init__()
        
        n_upsample = int(np.log2(64 / input_grid))
        
        layers = [nn.Conv2d(latent_dim, hidden_dim * (2 ** (n_upsample-1)), 1)]
        
        in_ch = hidden_dim * (2 ** (n_upsample-1))
        for i in range(n_upsample):
            out_ch = in_ch // 2 if i < n_upsample - 1 else hidden_dim
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        
        layers.append(nn.Conv2d(hidden_dim, out_channels, 3, 1, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim, grid, grid] -> [B, 1, H, W]"""
        return self.net(z)


# ============================================================
# Dynamics Model
# ============================================================

class DynamicsPredictor(nn.Module):
    """Predict next latent codes from current codes + action.
    
    Unlike v3/v4, this operates on the continuous latent BEFORE quantization,
    then we compare against quantized target.
    """
    
    def __init__(self, latent_dim: int, num_actions: int, 
                 grid_size: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        
        # Process spatial latents with action conditioning
        self.action_embed = nn.Embedding(num_actions, latent_dim)
        
        # Transformer-style processing (action as extra token)
        self.proj_in = nn.Linear(latent_dim, hidden_dim)
        
        # Simple MLP for now (could use attention)
        n_positions = grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * n_positions + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, latent_dim * n_positions),
        )
        
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim, H, W] current latent
            action: [B] action indices
        Returns:
            z_pred: [B, latent_dim, H, W] predicted next latent
        """
        B, D, H, W = z.shape
        
        # Flatten spatial
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H*W, D)  # [B, HW, D]
        z_proj = self.proj_in(z_flat)  # [B, HW, hidden]
        z_proj = z_proj.reshape(B, -1)  # [B, HW*hidden]
        
        # Action embedding
        a_emb = self.action_embed(action)  # [B, D]
        a_proj = self.proj_in(a_emb)  # [B, hidden]
        
        # Predict
        x = torch.cat([z_proj, a_proj], dim=-1)
        z_pred = self.net(x)  # [B, D*HW]
        z_pred = z_pred.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        
        return z_pred


# ============================================================
# Full Model
# ============================================================

class JEPAWorldModel(nn.Module):
    """
    JEPA-inspired world model:
    - Online encoder: learns representations
    - Target encoder (EMA): provides stable prediction targets
    - Dynamics: predicts target latents from online latents + action
    """
    
    def __init__(self, num_codes: int = 512, num_actions: int = 4,
                 latent_dim: int = 64, hidden_dim: int = 64,
                 grid_size: int = 4, ema_decay: float = 0.99):
        super().__init__()
        
        self.num_codes = num_codes
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        
        # Online encoder
        self.encoder = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim,
                              output_grid=grid_size)
        
        # Target encoder (EMA of online encoder)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        
        self.ema_decay = ema_decay
        
        # VQ layer
        self.vq = VectorQuantizerWithEntropy(num_codes, latent_dim)
        
        # Dynamics predictor
        self.dynamics = DynamicsPredictor(latent_dim, num_actions, grid_size)
        
        # Optional decoder (for visualization, not training)
        self.decoder = Decoder(hidden_dim=hidden_dim, latent_dim=latent_dim,
                              input_grid=grid_size)
        
    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder."""
        for online_p, target_p in zip(self.encoder.parameters(), 
                                       self.target_encoder.parameters()):
            target_p.data.mul_(self.ema_decay).add_(online_p.data, alpha=1 - self.ema_decay)
    
    def encode(self, x: torch.Tensor, use_target: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Encode image to latents and codes.
        
        Returns:
            z: [B, D, H, W] continuous latent (pre-VQ)
            z_q: [B, D, H, W] quantized latent
            codes: [B, H, W] code indices
            vq_info: dict
        """
        encoder = self.target_encoder if use_target else self.encoder
        z = encoder(x)  # [B, D, H, W]
        
        # Reshape for VQ: [B, H, W, D]
        z_for_vq = z.permute(0, 2, 3, 1)
        z_q_hwc, codes, vq_info = self.vq(z_for_vq)
        z_q = z_q_hwc.permute(0, 3, 1, 2)  # [B, D, H, W]
        
        return z, z_q, codes, vq_info
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent to image."""
        return self.decoder(z_q)
    
    def predict_next_latent(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent from current latent + action."""
        return self.dynamics(z, action)
    
    def forward(self, x: torch.Tensor, action: Optional[torch.Tensor] = None,
                x_next: Optional[torch.Tensor] = None):
        """Forward pass."""
        # Encode current (online encoder)
        z, z_q, codes, vq_info = self.encode(x, use_target=False)
        
        results = {
            'z': z,
            'z_q': z_q,
            'codes': codes,
            'vq_info': vq_info,
        }
        
        if action is not None and x_next is not None:
            # Predict next latent
            z_pred = self.predict_next_latent(z_q, action)  # Use quantized as input
            results['z_pred'] = z_pred
            
            # Get target latent (from target encoder, no gradient)
            with torch.no_grad():
                z_target, z_q_target, codes_target, _ = self.encode(x_next, use_target=True)
            
            results['z_target'] = z_target
            results['z_q_target'] = z_q_target
            results['codes_target'] = codes_target
        
        return results


# ============================================================
# Losses
# ============================================================

def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Encourage variance in latent dimensions (prevent collapse).
    
    From VICReg: variance should be above some threshold.
    """
    # z: [B, D, H, W] or [B, D]
    if z.dim() == 4:
        z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])  # [B*H*W, D]
    
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1 - std))  # Hinge loss: penalize if std < 1


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Encourage decorrelation between latent dimensions.
    
    From VICReg/Barlow Twins.
    """
    if z.dim() == 4:
        z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    
    z = z - z.mean(dim=0)
    N, D = z.shape
    cov = (z.T @ z) / (N - 1)
    
    # Off-diagonal elements should be zero
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / D


def code_entropy_loss(avg_probs: torch.Tensor) -> torch.Tensor:
    """Encourage uniform code usage (maximize entropy).
    
    Returns negative entropy (to minimize = maximize entropy).
    """
    entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
    max_entropy = np.log(len(avg_probs))
    return -entropy / max_entropy  # Negative because we minimize loss


# ============================================================
# Data Generation
# ============================================================

def generate_2048_trajectories(n_trajectories: int, max_steps: int, img_size: int = 64):
    """Generate 2048 gameplay trajectories."""
    try:
        from ..grid.game_2048 import Game2048
        from ..grid.renderer import GameRenderer
    except ImportError:
        from world_models.grid.game_2048 import Game2048
        from world_models.grid.renderer import GameRenderer
    
    renderer = GameRenderer(img_size)
    all_obs = []
    all_actions = []
    
    for _ in tqdm(range(n_trajectories), desc="Generating 2048"):
        game = Game2048()
        obs_list = [renderer.render_2048(game.board)]
        action_list = []
        
        for _ in range(max_steps):
            action = np.random.randint(4)
            obs, reward, done, info = game.step(action)
            obs_list.append(renderer.render_2048(game.board))
            action_list.append(action)
            
            if done:
                game.reset()
        
        all_obs.append(np.stack(obs_list))
        all_actions.append(np.array(action_list))
    
    obs = np.stack(all_obs)[:, :max_steps+1]
    actions = np.stack(all_actions)[:, :max_steps]
    obs = obs[:, :, np.newaxis, :, :]
    return obs.astype(np.float32), actions.astype(np.int64)


def generate_othello_trajectories(n_trajectories: int, max_steps: int, img_size: int = 64):
    """Generate Othello gameplay trajectories."""
    try:
        from ..grid.othello import OthelloGame
        from ..grid.renderer import GameRenderer
    except ImportError:
        from world_models.grid.othello import OthelloGame
        from world_models.grid.renderer import GameRenderer
    
    renderer = GameRenderer(img_size)
    all_obs = []
    all_actions = []
    
    for _ in tqdm(range(n_trajectories), desc="Generating Othello"):
        game = OthelloGame()
        obs_list = [renderer.render_othello(game.board)]
        action_list = []
        
        for _ in range(max_steps):
            valid = game.get_valid_moves()
            if not valid or game.done:
                game.reset()
                valid = game.get_valid_moves()
            
            action = np.random.choice(valid) if valid else 0
            obs, reward, done, info = game.step(action)
            obs_list.append(renderer.render_othello(game.board))
            action_list.append(action)
        
        all_obs.append(np.stack(obs_list))
        all_actions.append(np.array(action_list))
    
    obs = np.stack(all_obs)[:, :max_steps+1]
    actions = np.stack(all_actions)[:, :max_steps]
    obs = obs[:, :, np.newaxis, :, :]
    return obs.astype(np.float32), actions.astype(np.int64)


# ============================================================
# Training
# ============================================================

def train(game: str, n_steps: int, device: str, output_dir: str,
          n_trajectories: int = 2000, traj_len: int = 50,
          batch_size: int = 64, num_codes: int = 512,
          # Loss weights
          w_pred: float = 1.0,      # Prediction in latent space
          w_commit: float = 0.25,   # VQ commitment
          w_entropy: float = 0.1,   # Code diversity
          w_variance: float = 0.1,  # Prevent collapse
          w_recon: float = 0.0,     # Reconstruction (0 = pure JEPA)
          ):
    """Train JEPA-style world model."""
    
    print("=" * 60)
    print("VQ World Model v5 - JEPA-inspired")
    print(f"Game: {game} | Steps: {n_steps} | Device: {device}")
    print("=" * 60)
    print("\nKey ideas:")
    print("  - Predict in LATENT space, not pixel space")
    print("  - EMA target encoder for stable targets")
    print("  - Entropy regularization for code diversity")
    print("  - Variance regularization to prevent collapse")
    print(f"\nLoss weights: pred={w_pred}, commit={w_commit}, " 
          f"entropy={w_entropy}, var={w_variance}, recon={w_recon}\n")
    
    # Generate data
    print("Generating trajectories...")
    if game == '2048':
        obs, actions = generate_2048_trajectories(n_trajectories, traj_len)
        num_actions = 4
        grid_size = 4
    else:
        obs, actions = generate_othello_trajectories(n_trajectories, traj_len)
        num_actions = 64
        grid_size = 8
    
    print(f"Data: {obs.shape[0]} trajectories, {obs.shape[1]-1} steps each")
    
    # Create model
    model = JEPAWorldModel(
        num_codes=num_codes,
        num_actions=num_actions,
        latent_dim=64,
        hidden_dim=64,
        grid_size=grid_size,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")
    
    # Move data to device
    obs_t = torch.from_numpy(obs).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    
    # Training loop
    pbar = tqdm(range(n_steps))
    metrics = defaultdict(list)
    
    for step in pbar:
        model.train()
        
        # Sample batch
        batch_idx = torch.randint(0, obs_t.shape[0], (batch_size,))
        time_idx = torch.randint(0, obs_t.shape[1] - 1, (batch_size,))
        
        x = obs_t[batch_idx, time_idx]
        x_next = obs_t[batch_idx, time_idx + 1]
        a = actions_t[batch_idx, time_idx]
        
        # Forward
        results = model(x, a, x_next)
        
        # === Losses ===
        
        # 1. Prediction loss in latent space
        # Predict where target encoder puts next state
        pred_loss = F.mse_loss(results['z_pred'], results['z_target'])
        
        # 2. VQ commitment
        commit_loss = results['vq_info']['commitment_loss']
        
        # 3. Code entropy (maximize diversity)
        entropy_loss = code_entropy_loss(results['vq_info']['avg_probs'])
        
        # 4. Variance (prevent collapse)
        var_loss = variance_loss(results['z'])
        
        # 5. Optional reconstruction
        if w_recon > 0:
            recon = model.decode(results['z_q'])
            recon_loss = F.mse_loss(recon, x)
        else:
            recon_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        loss = (w_pred * pred_loss + 
                w_commit * commit_loss + 
                w_entropy * entropy_loss +
                w_variance * var_loss +
                w_recon * recon_loss)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update target encoder (EMA)
        model.update_target_encoder()
        
        # Compute accuracy metrics
        with torch.no_grad():
            # Quantize prediction and compare to target codes
            z_pred_for_vq = results['z_pred'].permute(0, 2, 3, 1)
            _, pred_codes, _ = model.vq(z_pred_for_vq)
            
            target_codes = results['codes_target']
            curr_codes = results['codes']
            
            # Overall accuracy
            accuracy = (pred_codes == target_codes).float().mean()
            
            # Split by changed/unchanged
            changed_mask = (curr_codes != target_codes)
            if changed_mask.any():
                acc_changed = (pred_codes[changed_mask] == target_codes[changed_mask]).float().mean()
                n_changed = changed_mask.sum().item()
            else:
                acc_changed = torch.tensor(0.0)
                n_changed = 0
            
            if (~changed_mask).any():
                acc_stayed = (pred_codes[~changed_mask] == target_codes[~changed_mask]).float().mean()
                n_stayed = (~changed_mask).sum().item()
            else:
                acc_stayed = torch.tensor(0.0)
                n_stayed = 0
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'pred': f"{pred_loss.item():.4f}",
            'acc': f"{accuracy.item():.1%}",
            'ent': f"{-entropy_loss.item():.2f}",  # Show positive entropy
        })
        
        # Log
        if step % 100 == 0:
            metrics['step'].append(step)
            metrics['loss'].append(loss.item())
            metrics['pred_loss'].append(pred_loss.item())
            metrics['accuracy'].append(accuracy.item())
            metrics['code_entropy'].append(-entropy_loss.item())
        
        # Periodic detailed logging
        if (step + 1) % 2000 == 0:
            stats = model.vq.get_usage_stats()
            print(f"\n[Step {step+1}]")
            print(f"  Prediction loss: {pred_loss.item():.4f}")
            print(f"  Codebook: {stats['active_codes']}/{stats['total_codes']} "
                  f"({100*stats['usage_ratio']:.1f}%), entropy={stats['entropy_bits']:.2f} bits")
            print(f"  Accuracy: {accuracy.item():.1%}")
            print(f"    Changed:   {acc_changed.item() if isinstance(acc_changed, torch.Tensor) else acc_changed:.1%} (n={n_changed})")
            print(f"    Unchanged: {acc_stayed.item() if isinstance(acc_stayed, torch.Tensor) else acc_stayed:.1%} (n={n_stayed})")
            
            model.vq.reset_counts()
    
    # Final analysis
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - Final Analysis")
    print("=" * 60)
    
    model.eval()
    model.vq.reset_counts()
    
    # Analyze on larger sample
    print("\nAnalyzing learned representation...")
    
    all_entropies = []
    all_changed = []
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_start in range(0, min(500, obs_t.shape[0]), 32):
            batch_end = min(batch_start + 32, obs_t.shape[0])
            
            for t in range(min(20, obs_t.shape[1] - 1)):  # Limit time steps
                x = obs_t[batch_start:batch_end, t]
                x_next = obs_t[batch_start:batch_end, t + 1]
                a = actions_t[batch_start:batch_end, t]
                
                results = model(x, a, x_next)
                
                # Get codes
                curr_codes = results['codes']
                target_codes = results['codes_target']
                
                # Predict and quantize
                z_pred_for_vq = results['z_pred'].permute(0, 2, 3, 1)
                _, pred_codes, _ = model.vq(z_pred_for_vq)
                
                # Compute per-position "entropy" via prediction uncertainty
                # Use distance to nearest code as proxy for uncertainty
                z_pred_flat = results['z_pred'].permute(0, 2, 3, 1).reshape(-1, model.latent_dim)
                dist = (z_pred_flat.pow(2).sum(1, keepdim=True) 
                        - 2 * z_pred_flat @ model.vq.embed.t()
                        + model.vq.embed.pow(2).sum(1, keepdim=True).t())
                
                # Softmax over negative distances = probability of each code
                probs = F.softmax(-dist / 0.1, dim=-1)  # Temperature for sharper distribution
                entropy = -(probs * (probs + 1e-10).log()).sum(-1)
                entropy = entropy.reshape(curr_codes.shape)
                
                # Track
                changed = (curr_codes != target_codes)
                correct = (pred_codes == target_codes)
                
                total_correct += correct.sum().item()
                total_samples += correct.numel()
                
                all_entropies.append(entropy.cpu().numpy().flatten())
                all_changed.append(changed.cpu().numpy().flatten())
    
    all_entropies = np.concatenate(all_entropies)
    all_changed = np.concatenate(all_changed)
    
    # Convert to bits
    all_entropies = all_entropies / np.log(2)
    
    stats = model.vq.get_usage_stats()
    print(f"\nCodebook utilization: {stats['active_codes']}/{stats['total_codes']} "
          f"({100*stats['usage_ratio']:.1f}%)")
    print(f"Code entropy: {stats['entropy_bits']:.2f} bits")
    
    print(f"\nFinal prediction accuracy: {100*total_correct/total_samples:.1f}%")
    
    print(f"\nPrediction uncertainty (entropy proxy):")
    print(f"  Mean: {all_entropies.mean():.4f} bits")
    print(f"  Std:  {all_entropies.std():.4f}")
    
    ent_changed = all_entropies[all_changed]
    ent_stayed = all_entropies[~all_changed]
    
    print(f"\nEntropy by position type:")
    if len(ent_changed) > 0:
        print(f"  CHANGED ({len(ent_changed)} samples, {100*len(ent_changed)/len(all_entropies):.1f}%):")
        print(f"    Mean: {ent_changed.mean():.4f} bits")
        print(f"    Median: {np.median(ent_changed):.4f} bits")
    else:
        print(f"  CHANGED: 0 samples")
    
    print(f"  STAYED ({len(ent_stayed)} samples, {100*len(ent_stayed)/len(all_entropies):.1f}%):")
    print(f"    Mean: {ent_stayed.mean():.4f} bits")
    print(f"    Median: {np.median(ent_stayed):.4f} bits")
    
    if len(ent_changed) > 0 and ent_stayed.mean() > 0.001:
        ratio = ent_changed.mean() / ent_stayed.mean()
        print(f"\n  ** ENTROPY RATIO: {ratio:.1f}x for changed vs stayed **")
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state': model.state_dict(),
        'metrics': dict(metrics),
        'config': {
            'game': game,
            'num_codes': num_codes,
            'n_steps': n_steps,
            'loss_weights': {
                'pred': w_pred, 'commit': w_commit, 
                'entropy': w_entropy, 'variance': w_variance, 'recon': w_recon
            }
        }
    }, output_path / f'{game}_vq_v5.pt')
    
    print(f"\nModel saved to {output_path / f'{game}_vq_v5.pt'}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='2048', choices=['2048', 'othello'])
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='outputs_vq_v5')
    parser.add_argument('--trajectories', type=int, default=2000)
    parser.add_argument('--num_codes', type=int, default=512)
    
    # Loss weights
    parser.add_argument('--w_pred', type=float, default=1.0)
    parser.add_argument('--w_entropy', type=float, default=0.1)
    parser.add_argument('--w_variance', type=float, default=0.1)
    parser.add_argument('--w_recon', type=float, default=0.0, 
                        help='Reconstruction weight (0=pure JEPA)')
    
    args = parser.parse_args()
    
    train(
        game=args.game,
        n_steps=args.steps,
        device=args.device,
        output_dir=args.output,
        n_trajectories=args.trajectories,
        num_codes=args.num_codes,
        w_pred=args.w_pred,
        w_entropy=args.w_entropy,
        w_variance=args.w_variance,
        w_recon=args.w_recon,
    )


if __name__ == '__main__':
    main()
