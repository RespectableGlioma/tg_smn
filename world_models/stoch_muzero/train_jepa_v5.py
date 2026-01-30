"""
VQ-VAE World Model v5 - JEPA-style Predictive Learning

Key insight from user: Reconstruction loss learns features for PIXELS, 
not for DYNAMICS. We need representations driven by PREDICTABILITY.

Approach:
1. Encode observations to latent codes
2. Predict NEXT LATENT from (current latent, action) 
3. Loss is in LATENT SPACE, not pixel space
4. Reconstruction is optional/auxiliary (for visualization only)

This is similar to JEPA (Joint Embedding Predictive Architecture):
- Learn representations where future states are predictable
- No pixel decoder needed for training
- Codes emerge from what's useful for prediction

Architecture:
- Encoder: obs -> continuous latent z
- VQ: z -> discrete codes (for interpretability)
- Predictor: (z_t, action) -> predicted z_{t+1}
- Loss: ||predicted_z - actual_z_{t+1}||
- Diversity: entropy regularization to prevent collapse

Key differences from v1-v4:
- NO reconstruction loss driving the encoder
- Encoder learns to produce PREDICTABLE latents
- Codes represent "what matters for dynamics"
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional, Dict
from collections import defaultdict
import copy


# ============================================================
# VQ Layer with entropy regularization
# ============================================================

class VectorQuantizerEMA(nn.Module):
    """VQ with EMA updates."""
    
    def __init__(self, num_codes: int, code_dim: int, 
                 ema_decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        
        embed = torch.randn(num_codes, code_dim) * 0.1
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.ones(num_codes))
        self.register_buffer('embed_avg', embed.clone())
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            z: [..., D] continuous latents
        Returns:
            z_q: [..., D] quantized
            codes: [...] indices  
            info: dict with distances for soft assignment
        """
        orig_shape = z.shape
        flat_z = z.reshape(-1, self.code_dim)
        
        # Distances to codes
        dist = (flat_z.pow(2).sum(1, keepdim=True) 
                - 2 * flat_z @ self.embed.t()
                + self.embed.pow(2).sum(1, keepdim=True).t())
        
        # Hard assignment
        _, codes = dist.min(1)
        
        # Quantize
        z_q = F.embedding(codes, self.embed)
        
        # Soft assignment probabilities (for entropy computation)
        soft_assign = F.softmax(-dist / 0.1, dim=-1)  # temperature=0.1
        
        # EMA updates
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(codes, self.num_codes).float()
                
                self.cluster_size.mul_(self.ema_decay).add_(
                    encodings.sum(0), alpha=1 - self.ema_decay
                )
                
                embed_sum = encodings.t() @ flat_z
                self.embed_avg.mul_(self.ema_decay).add_(
                    embed_sum, alpha=1 - self.ema_decay
                )
                
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.epsilon)
                    / (n + self.num_codes * self.epsilon) * n
                )
                
                self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))
        
        # Commitment loss (anchor z to codes)
        commitment_loss = F.mse_loss(flat_z, z_q.detach())
        
        # Straight-through
        z_q = flat_z + (z_q - flat_z).detach()
        
        # Reshape
        z_q = z_q.view(orig_shape)
        codes = codes.view(orig_shape[:-1])
        
        # Perplexity (effective codebook usage)
        avg_probs = F.one_hot(codes.view(-1), self.num_codes).float().mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Code entropy (for regularization)
        code_entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
        
        info = {
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
            'code_entropy': code_entropy,
            'soft_assign': soft_assign.view(*orig_shape[:-1], self.num_codes),
        }
        
        return z_q, codes, info
    
    def get_usage_stats(self) -> Dict:
        used = (self.cluster_size > 1).sum().item()
        return {
            'active_codes': int(used),
            'total_codes': self.num_codes,
            'usage_ratio': used / self.num_codes,
        }


# ============================================================
# Model Components
# ============================================================

class Encoder(nn.Module):
    """Image -> latent grid. Standard CNN."""
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, 
                 latent_dim: int = 64, grid_size: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        
        # CNN that outputs grid_size x grid_size spatial
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1),      # 64->32
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),       # 32->16
            nn.ReLU(), 
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),   # 16->8
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, latent_dim, 4, 2, 1),   # 8->4
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, H, W] -> [B, D, grid, grid]"""
        return self.net(x)


class LatentPredictor(nn.Module):
    """Predict next latent from current latent + action.
    
    This is the core of JEPA-style learning:
    - Input: current latent z_t [B, D, H, W] and action
    - Output: predicted next latent z_{t+1} [B, D, H, W]
    """
    def __init__(self, latent_dim: int, num_actions: int, 
                 grid_size: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.num_positions = grid_size * grid_size
        
        # Action embedding
        self.action_embed = nn.Embedding(num_actions, hidden_dim)
        
        # Process spatial latent + action -> next latent
        # Flatten spatial, concat action, predict
        input_dim = self.num_positions * latent_dim + hidden_dim
        output_dim = self.num_positions * latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )
        
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, D, H, W] current latent
            action: [B] action indices
        Returns:
            z_pred: [B, D, H, W] predicted next latent
        """
        B = z.shape[0]
        
        # Flatten spatial
        z_flat = z.reshape(B, -1)  # [B, D*H*W]
        
        # Action embedding
        a_emb = self.action_embed(action)  # [B, hidden]
        
        # Predict
        x = torch.cat([z_flat, a_emb], dim=-1)
        z_pred_flat = self.net(x)
        
        # Reshape
        z_pred = z_pred_flat.reshape(B, self.latent_dim, self.grid_size, self.grid_size)
        
        return z_pred


class CodePredictor(nn.Module):
    """Predict next codes from current codes + action.
    
    For interpretability: discrete code predictions with entropy.
    """
    def __init__(self, num_codes: int, num_actions: int,
                 grid_size: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.num_codes = num_codes
        self.grid_size = grid_size
        self.num_positions = grid_size * grid_size
        
        self.code_embed = nn.Embedding(num_codes, hidden_dim // 4)
        self.action_embed = nn.Embedding(num_actions, hidden_dim // 4)
        
        input_dim = self.num_positions * (hidden_dim // 4) + hidden_dim // 4
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_positions * num_codes),
        )
        
    def forward(self, codes: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: [B, H, W] current code indices
            action: [B] action indices
        Returns:
            logits: [B, H, W, num_codes] next code logits
        """
        B = codes.shape[0]
        
        code_emb = self.code_embed(codes)  # [B, H, W, D]
        code_emb_flat = code_emb.reshape(B, -1)
        
        action_emb = self.action_embed(action)
        
        x = torch.cat([code_emb_flat, action_emb], dim=-1)
        logits = self.net(x)
        
        return logits.reshape(B, self.grid_size, self.grid_size, self.num_codes)


class Decoder(nn.Module):
    """Optional: latent -> image for visualization."""
    def __init__(self, out_channels: int = 1, hidden_dim: int = 64, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim * 2, 4, 2, 1),  # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),      # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1),    # 32->64
            nn.Sigmoid(),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ============================================================
# Full Model
# ============================================================

class JEPAWorldModel(nn.Module):
    """
    JEPA-style World Model:
    - Encoder produces latents
    - VQ discretizes for interpretability  
    - Predictor learns dynamics in latent space
    - No reconstruction needed for training
    """
    
    def __init__(self, num_codes: int = 512, num_actions: int = 4,
                 latent_dim: int = 64, hidden_dim: int = 64,
                 grid_size: int = 4):
        super().__init__()
        self.num_codes = num_codes
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        
        # Core components
        self.encoder = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.vq = VectorQuantizerEMA(num_codes, latent_dim)
        
        # Predictors (both latent and code level)
        self.latent_predictor = LatentPredictor(latent_dim, num_actions, grid_size)
        self.code_predictor = CodePredictor(num_codes, num_actions, grid_size)
        
        # Optional decoder (for visualization)
        self.decoder = Decoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        
        # Target encoder (EMA of main encoder for stable targets)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.99):
        """EMA update of target encoder."""
        for p, p_target in zip(self.encoder.parameters(), 
                               self.target_encoder.parameters()):
            p_target.data.mul_(momentum).add_(p.data, alpha=1 - momentum)
    
    def encode(self, x: torch.Tensor, use_target: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Encode image to latent and codes.
        
        Returns:
            z: [B, D, H, W] continuous latent
            z_q: [B, D, H, W] quantized latent
            codes: [B, H, W] code indices
            vq_info: dict
        """
        encoder = self.target_encoder if use_target else self.encoder
        z = encoder(x)  # [B, D, H, W]
        
        # Reshape for VQ: [B, H, W, D]
        z_for_vq = z.permute(0, 2, 3, 1)
        z_q, codes, vq_info = self.vq(z_for_vq)
        
        # Back to [B, D, H, W]
        z_q = z_q.permute(0, 3, 1, 2)
        
        return z, z_q, codes, vq_info
    
    def forward(self, x: torch.Tensor, action: torch.Tensor, 
                x_next: torch.Tensor) -> Dict:
        """
        Full forward pass.
        
        JEPA-style: predict target latent, not pixels.
        """
        # Encode current (online encoder)
        z_t, z_q_t, codes_t, vq_info = self.encode(x, use_target=False)
        
        # Encode next (target encoder - EMA, no gradient)
        with torch.no_grad():
            z_t_next, z_q_t_next, codes_t_next, _ = self.encode(x_next, use_target=True)
        
        # === LATENT PREDICTION (main JEPA loss) ===
        # Predict next latent from current latent + action
        z_pred = self.latent_predictor(z_t, action)
        
        # Loss: predict target encoder's output
        # Using z (continuous) not z_q (quantized) for smoother gradients
        latent_pred_loss = F.mse_loss(z_pred, z_t_next)
        
        # === CODE PREDICTION (for interpretability) ===
        code_logits = self.code_predictor(codes_t, action)
        code_pred_loss = F.cross_entropy(
            code_logits.reshape(-1, self.num_codes),
            codes_t_next.reshape(-1)
        )
        
        # Code prediction entropy (uncertainty measure)
        code_probs = F.softmax(code_logits, dim=-1)
        code_entropy = -(code_probs * (code_probs + 1e-10).log()).sum(-1)  # [B, H, W]
        
        # Accuracy
        pred_codes = code_logits.argmax(dim=-1)
        accuracy = (pred_codes == codes_t_next).float().mean()
        
        # Changed vs stayed analysis
        changed_mask = (codes_t != codes_t_next)
        
        return {
            # Losses
            'latent_pred_loss': latent_pred_loss,
            'code_pred_loss': code_pred_loss,
            'commitment_loss': vq_info['commitment_loss'],
            
            # Monitoring
            'perplexity': vq_info['perplexity'],
            'code_entropy': vq_info['code_entropy'],
            'accuracy': accuracy,
            'pred_entropy': code_entropy,
            'changed_mask': changed_mask,
            'codes_t': codes_t,
            'codes_t_next': codes_t_next,
            'pred_codes': pred_codes,
            
            # For visualization
            'z_q_t': z_q_t,
        }


# ============================================================
# Data Generation (same as before)
# ============================================================

def generate_2048_trajectories(n_trajectories: int, max_steps: int, img_size: int = 64):
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
            game.step(action)
            obs_list.append(renderer.render_2048(game.board))
            action_list.append(action)
            
            if game.done:
                game.reset()
        
        all_obs.append(np.stack(obs_list))
        all_actions.append(np.array(action_list))
    
    obs = np.stack(all_obs)[:, :max_steps+1]
    actions = np.stack(all_actions)[:, :max_steps]
    obs = obs[:, :, np.newaxis, :, :]
    return obs.astype(np.float32), actions.astype(np.int64)


def generate_othello_trajectories(n_trajectories: int, max_steps: int, img_size: int = 64):
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
            game.step(action)
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
          batch_size: int = 64, num_codes: int = 256,
          latent_pred_weight: float = 1.0,
          code_pred_weight: float = 0.5,
          commitment_weight: float = 0.1,
          target_momentum: float = 0.99):
    """
    Train JEPA-style world model.
    
    Key: NO reconstruction loss. Learn representations for PREDICTION.
    """
    
    print("=" * 60)
    print("JEPA-style World Model v5")
    print(f"Game: {game} | Steps: {n_steps} | Device: {device}")
    print("=" * 60)
    print("\nKey differences from v1-v4:")
    print("  - NO pixel reconstruction loss")
    print("  - Learn representations for PREDICTION, not reconstruction")
    print("  - Target encoder (EMA) for stable prediction targets")
    print("  - Codes emerge from predictive utility\n")
    
    # Generate data
    print("Generating trajectories...")
    if game == '2048':
        obs, actions = generate_2048_trajectories(n_trajectories, traj_len)
        num_actions = 4
    else:
        obs, actions = generate_othello_trajectories(n_trajectories, traj_len)
        num_actions = 64
    
    print(f"Data: {obs.shape[0]} trajectories, {obs.shape[1]-1} steps each\n")
    
    # Create model
    model = JEPAWorldModel(
        num_codes=num_codes,
        num_actions=num_actions,
        latent_dim=64,
        hidden_dim=64,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")
    
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
        
        # Combined loss (NO reconstruction!)
        loss = (latent_pred_weight * results['latent_pred_loss'] +
                code_pred_weight * results['code_pred_loss'] +
                commitment_weight * results['commitment_loss'])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update target encoder (EMA)
        model.update_target_encoder(momentum=target_momentum)
        
        # Logging
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lat': f"{results['latent_pred_loss'].item():.4f}",
            'code': f"{results['code_pred_loss'].item():.4f}",
            'acc': f"{results['accuracy'].item():.1%}",
            'perp': f"{results['perplexity'].item():.1f}",
        })
        
        if step % 100 == 0:
            for k in ['latent_pred_loss', 'code_pred_loss', 'accuracy', 'perplexity']:
                metrics[k].append(results[k].item())
            metrics['step'].append(step)
        
        # Detailed logging
        if (step + 1) % 2000 == 0:
            stats = model.vq.get_usage_stats()
            
            # Compute accuracy split by changed/stayed
            changed = results['changed_mask']
            correct = (results['pred_codes'] == results['codes_t_next'])
            
            acc_changed = correct[changed].float().mean().item() if changed.any() else 0
            acc_stayed = correct[~changed].float().mean().item() if (~changed).any() else 0
            n_changed = changed.sum().item()
            n_total = changed.numel()
            
            # Entropy split
            pred_ent = results['pred_entropy']
            ent_changed = pred_ent[changed].mean().item() / np.log(2) if changed.any() else 0
            ent_stayed = pred_ent[~changed].mean().item() / np.log(2) if (~changed).any() else 0
            
            print(f"\n[Step {step+1}]")
            print(f"  Codebook: {stats['active_codes']}/{stats['total_codes']} ({100*stats['usage_ratio']:.1f}%)")
            print(f"  Perplexity: {results['perplexity'].item():.1f}")
            print(f"  Latent pred loss: {results['latent_pred_loss'].item():.4f}")
            print(f"  Code pred accuracy: {results['accuracy'].item():.1%}")
            print(f"    Changed: {acc_changed:.1%} (n={n_changed}, {100*n_changed/n_total:.1f}%)")
            print(f"    Stayed:  {acc_stayed:.1%}")
            print(f"  Prediction entropy (bits):")
            print(f"    Changed: {ent_changed:.3f}")
            print(f"    Stayed:  {ent_stayed:.3f}")
            if ent_stayed > 0.001:
                print(f"    RATIO:   {ent_changed/ent_stayed:.1f}x")
    
    # ============================================================
    # Final Analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - Final Analysis")
    print("=" * 60)
    
    model.eval()
    
    all_entropies = []
    all_changed = []
    all_correct = []
    
    with torch.no_grad():
        for batch_start in range(0, min(500, obs_t.shape[0]), 32):
            batch_end = min(batch_start + 32, obs_t.shape[0])
            
            for t in range(obs_t.shape[1] - 1):
                x = obs_t[batch_start:batch_end, t]
                x_next = obs_t[batch_start:batch_end, t + 1]
                a = actions_t[batch_start:batch_end, t]
                
                results = model(x, a, x_next)
                
                # Entropy per position
                entropy = results['pred_entropy'].cpu().numpy().flatten()
                changed = results['changed_mask'].cpu().numpy().flatten()
                correct = (results['pred_codes'] == results['codes_t_next']).cpu().numpy().flatten()
                
                all_entropies.append(entropy)
                all_changed.append(changed)
                all_correct.append(correct)
    
    all_entropies = np.concatenate(all_entropies) / np.log(2)  # to bits
    all_changed = np.concatenate(all_changed)
    all_correct = np.concatenate(all_correct)
    
    stats = model.vq.get_usage_stats()
    
    print(f"\nCodebook utilization: {stats['active_codes']}/{stats['total_codes']} ({100*stats['usage_ratio']:.1f}%)")
    print(f"Final accuracy: {100*all_correct.mean():.1f}%")
    
    print(f"\nPosition statistics:")
    print(f"  Changed: {all_changed.sum()} ({100*all_changed.mean():.1f}%)")
    print(f"  Stayed:  {(~all_changed).sum()} ({100*(1-all_changed.mean()):.1f}%)")
    
    ent_changed = all_entropies[all_changed]
    ent_stayed = all_entropies[~all_changed]
    
    print(f"\nPrediction entropy (bits):")
    print(f"  Overall mean: {all_entropies.mean():.4f}")
    print(f"  Overall std:  {all_entropies.std():.4f}")
    
    if len(ent_changed) > 0:
        print(f"\n  CHANGED positions ({len(ent_changed)} samples):")
        print(f"    Mean: {ent_changed.mean():.4f} bits")
        print(f"    Median: {np.median(ent_changed):.4f} bits")
    
    if len(ent_stayed) > 0:
        print(f"\n  STAYED positions ({len(ent_stayed)} samples):")
        print(f"    Mean: {ent_stayed.mean():.4f} bits")
        print(f"    Median: {np.median(ent_stayed):.4f} bits")
    
    if len(ent_changed) > 0 and len(ent_stayed) > 0 and ent_stayed.mean() > 0.001:
        ratio = ent_changed.mean() / ent_stayed.mean()
        print(f"\n  ** ENTROPY RATIO: {ratio:.1f}x for changed vs stayed **")
    
    # Classification
    det_threshold = 0.1
    n_det = (all_entropies < det_threshold).sum()
    print(f"\nTransition classification (threshold={det_threshold} bits):")
    print(f"  Deterministic: {n_det} ({100*n_det/len(all_entropies):.1f}%)")
    print(f"  Stochastic:    {len(all_entropies)-n_det} ({100*(len(all_entropies)-n_det)/len(all_entropies):.1f}%)")
    
    # Accuracy by position type
    if len(all_correct[all_changed]) > 0:
        print(f"\nAccuracy breakdown:")
        print(f"  Changed positions: {100*all_correct[all_changed].mean():.1f}%")
        print(f"  Stayed positions:  {100*all_correct[~all_changed].mean():.1f}%")
    
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
        }
    }, output_path / f'{game}_jepa_v5.pt')
    
    print(f"\nModel saved to {output_path / f'{game}_jepa_v5.pt'}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='2048', choices=['2048', 'othello'])
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='outputs_jepa_v5')
    parser.add_argument('--trajectories', type=int, default=2000)
    parser.add_argument('--num_codes', type=int, default=256)
    args = parser.parse_args()
    
    train(
        game=args.game,
        n_steps=args.steps,
        device=args.device,
        output_dir=args.output,
        n_trajectories=args.trajectories,
        num_codes=args.num_codes,
    )


if __name__ == '__main__':
    main()
