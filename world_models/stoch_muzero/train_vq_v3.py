"""
VQ-VAE World Model v3 Training

Key changes from v2:
1. Warmup phase: Train reconstruction only first, let codebook stabilize
2. Then add dynamics loss after codebook is stable
3. No dead code reset during dynamics phase
4. Track transition accuracy explicitly
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional
from collections import defaultdict


# ============================================================
# Simple VQ Layer (no aggressive reset)
# ============================================================

class VectorQuantizer(nn.Module):
    """Simple VQ with EMA updates but NO aggressive reset."""
    
    def __init__(self, num_codes: int, code_dim: int, 
                 commitment_cost: float = 0.25,
                 ema_decay: float = 0.99):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        
        # Codebook
        self.register_buffer('codebook', torch.randn(num_codes, code_dim))
        self.register_buffer('ema_count', torch.zeros(num_codes))
        self.register_buffer('ema_weight', torch.randn(num_codes, code_dim))
        
        # Track usage for logging
        self.register_buffer('usage_count', torch.zeros(num_codes))
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z: [B, D] continuous latents
        Returns:
            z_q: [B, D] quantized
            codes: [B] code indices
            info: dict with losses and stats
        """
        B, D = z.shape
        
        # Find nearest codes
        dists = torch.cdist(z, self.codebook)  # [B, num_codes]
        codes = dists.argmin(dim=-1)  # [B]
        z_q = self.codebook[codes]  # [B, D]
        
        # Losses
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z.detach(), z_q)
        
        # Straight-through
        z_q = z + (z_q - z).detach()
        
        # EMA update (training only)
        if self.training:
            with torch.no_grad():
                # Count usage
                one_hot = F.one_hot(codes, self.num_codes).float()  # [B, num_codes]
                counts = one_hot.sum(0)  # [num_codes]
                
                # EMA updates
                self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1-self.ema_decay)
                
                # Sum of assigned vectors
                assigned_sum = one_hot.T @ z  # [num_codes, D]
                self.ema_weight.mul_(self.ema_decay).add_(assigned_sum, alpha=1-self.ema_decay)
                
                # Update codebook
                n = self.ema_count.unsqueeze(1)
                self.codebook.copy_(self.ema_weight / (n + 1e-5))
                
                # Track usage
                self.usage_count.add_(counts)
        
        info = {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'vq_loss': codebook_loss + self.commitment_cost * commitment_loss,
        }
        
        return z_q, codes, info
    
    def get_usage_stats(self) -> dict:
        """Get codebook usage statistics."""
        used = (self.usage_count > 0).sum().item()
        total = self.num_codes
        return {
            'active_codes': used,
            'total_codes': total,
            'usage_ratio': used / total,
        }
    
    def reset_usage_tracking(self):
        """Reset usage counter for new tracking period."""
        self.usage_count.zero_()


# ============================================================
# Model Components
# ============================================================

class Encoder(nn.Module):
    """Image -> latent grid."""
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),   # 32->16
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),   # 16->8
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, 4, 2, 1),   # 8->4
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, latent_dim, 4, 4]


class Decoder(nn.Module):
    """Latent grid -> image."""
    def __init__(self, out_channels: int = 1, hidden_dim: int = 64, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim, 4, 2, 1),  # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),  # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1), # 32->64
            nn.Sigmoid(),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DynamicsModel(nn.Module):
    """Predict next codes from current codes + action."""
    def __init__(self, num_codes: int, num_actions: int, 
                 grid_size: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.num_codes = num_codes
        self.grid_size = grid_size
        self.num_positions = grid_size * grid_size
        
        # Embed codes and actions
        self.code_embed = nn.Embedding(num_codes, hidden_dim // 2)
        self.action_embed = nn.Embedding(num_actions, hidden_dim // 2)
        
        # Process grid
        self.net = nn.Sequential(
            nn.Linear(self.num_positions * (hidden_dim // 2) + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_positions * num_codes),
        )
        
    def forward(self, codes: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: [B, grid_size, grid_size] current code indices
            action: [B] action indices
        Returns:
            logits: [B, grid_size, grid_size, num_codes] next code logits
        """
        B = codes.shape[0]
        
        # Embed
        code_emb = self.code_embed(codes)  # [B, H, W, D/2]
        code_emb = code_emb.reshape(B, -1)  # [B, H*W*D/2]
        action_emb = self.action_embed(action)  # [B, D/2]
        
        # Concat and predict
        x = torch.cat([code_emb, action_emb], dim=-1)
        logits = self.net(x)  # [B, H*W*num_codes]
        logits = logits.reshape(B, self.grid_size, self.grid_size, self.num_codes)
        
        return logits


# ============================================================
# Full Model
# ============================================================

class VQWorldModel(nn.Module):
    def __init__(self, num_codes: int = 512, num_actions: int = 4,
                 latent_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.num_codes = num_codes
        self.grid_size = 4
        
        self.encoder = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.vq = VectorQuantizer(num_codes, latent_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.dynamics = DynamicsModel(num_codes, num_actions, self.grid_size)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Encode image to quantized codes."""
        z = self.encoder(x)  # [B, D, H, W]
        B, D, H, W = z.shape
        
        # Reshape for VQ: [B*H*W, D]
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        z_q_flat, codes_flat, vq_info = self.vq(z_flat)
        
        # Reshape back
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)
        codes = codes_flat.reshape(B, H, W)
        
        return z_q, codes, vq_info
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to image."""
        return self.decoder(z_q)
    
    def predict_next_codes(self, codes: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict logits for next codes."""
        return self.dynamics(codes, action)
    
    def forward(self, x: torch.Tensor, action: Optional[torch.Tensor] = None,
                x_next: Optional[torch.Tensor] = None):
        """Full forward pass."""
        # Encode current
        z_q, codes, vq_info = self.encode(x)
        recon = self.decode(z_q)
        
        results = {
            'recon': recon,
            'codes': codes,
            'vq_info': vq_info,
        }
        
        # Dynamics (if action provided)
        if action is not None:
            next_logits = self.predict_next_codes(codes, action)
            results['next_logits'] = next_logits
            
            # If next obs provided, get target codes
            if x_next is not None:
                with torch.no_grad():
                    _, next_codes, _ = self.encode(x_next)
                results['next_codes_target'] = next_codes
        
        return results


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
    
    obs = np.stack(all_obs)[:, :max_steps+1]  # [N, T+1, H, W]
    actions = np.stack(all_actions)[:, :max_steps]  # [N, T]
    
    obs = obs[:, :, np.newaxis, :, :]  # Add channel dim
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
          warmup_steps: int = 5000, batch_size: int = 64):
    """
    Train with warmup phase:
    1. First warmup_steps: reconstruction only (stabilize codebook)
    2. After warmup: add dynamics loss
    """
    
    print("=" * 60)
    print(f"VQ-VAE World Model v3 Training")
    print(f"Game: {game} | Steps: {n_steps} | Warmup: {warmup_steps} | Device: {device}")
    print("=" * 60)
    
    # Generate data
    print("\nGenerating trajectories...")
    if game == '2048':
        obs, actions = generate_2048_trajectories(n_trajectories, traj_len)
        num_actions = 4
    else:
        obs, actions = generate_othello_trajectories(n_trajectories, traj_len)
        num_actions = 64
    
    print(f"Data: {obs.shape[0]} trajectories, {obs.shape[1]-1} steps each")
    print(f"Obs shape: {obs.shape}, Actions shape: {actions.shape}")
    
    # Create model
    model = VQWorldModel(num_codes=512, num_actions=num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Prepare data
    obs_t = torch.from_numpy(obs).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    
    # Training loop
    pbar = tqdm(range(n_steps))
    
    # Track metrics
    metrics = defaultdict(list)
    
    for step in pbar:
        model.train()
        
        # Sample batch
        batch_idx = torch.randint(0, obs_t.shape[0], (batch_size,))
        time_idx = torch.randint(0, obs_t.shape[1] - 1, (batch_size,))
        
        x = obs_t[batch_idx, time_idx]  # [B, 1, H, W]
        x_next = obs_t[batch_idx, time_idx + 1]
        a = actions_t[batch_idx, time_idx]
        
        # Forward
        if step < warmup_steps:
            # Warmup: reconstruction only
            z_q, codes, vq_info = model.encode(x)
            recon = model.decode(z_q)
            
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + vq_info['vq_loss']
            
            trans_loss = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
            accuracy = 0.0
            phase = "warmup"
        else:
            # Full training with dynamics
            results = model(x, a, x_next)
            
            recon_loss = F.mse_loss(results['recon'], x)
            vq_loss = results['vq_info']['vq_loss']
            
            # Transition loss
            next_logits = results['next_logits']  # [B, H, W, num_codes]
            next_target = results['next_codes_target']  # [B, H, W]
            
            trans_loss = F.cross_entropy(
                next_logits.reshape(-1, model.num_codes),
                next_target.reshape(-1)
            )
            
            # Compute entropy of predictions
            probs = F.softmax(next_logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(-1).mean()
            
            # Compute accuracy
            pred_codes = next_logits.argmax(dim=-1)
            accuracy = (pred_codes == next_target).float().mean().item()
            
            loss = recon_loss + vq_loss + trans_loss
            phase = "full"
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log
        if step % 100 == 0:
            stats = model.vq.get_usage_stats()
            
            metrics['step'].append(step)
            metrics['loss'].append(loss.item())
            metrics['recon'].append(recon_loss.item())
            metrics['trans'].append(trans_loss.item() if isinstance(trans_loss, torch.Tensor) else trans_loss)
            metrics['entropy'].append(entropy.item() if isinstance(entropy, torch.Tensor) else entropy)
            metrics['accuracy'].append(accuracy)
            metrics['codes_used'].append(stats['active_codes'])
        
        # Update progress bar
        pbar.set_postfix({
            'phase': phase,
            'loss': f"{loss.item():.4f}",
            'recon': f"{recon_loss.item():.4f}",
            'trans': f"{trans_loss.item():.4f}" if isinstance(trans_loss, torch.Tensor) else "N/A",
            'acc': f"{accuracy:.1%}" if accuracy > 0 else "N/A",
            'ent': f"{entropy.item():.3f}" if isinstance(entropy, torch.Tensor) else "N/A",
        })
        
        # Periodic logging
        if (step + 1) % 2000 == 0:
            model.vq.reset_usage_tracking()
            
            # Evaluate on full dataset sample
            model.eval()
            with torch.no_grad():
                eval_idx = torch.randint(0, obs_t.shape[0], (256,))
                eval_time = torch.randint(0, obs_t.shape[1] - 1, (256,))
                
                x_eval = obs_t[eval_idx, eval_time]
                x_next_eval = obs_t[eval_idx, eval_time + 1]
                a_eval = actions_t[eval_idx, eval_time]
                
                results = model(x_eval, a_eval, x_next_eval)
                
                if 'next_logits' in results:
                    pred = results['next_logits'].argmax(dim=-1)
                    target = results['next_codes_target']
                    
                    # Per-position accuracy
                    correct = (pred == target).float()
                    
                    # Check which positions changed
                    _, curr_codes, _ = model.encode(x_eval)
                    changed = (curr_codes != target)
                    stayed = ~changed
                    
                    acc_changed = correct[changed].mean().item() if changed.any() else 0
                    acc_stayed = correct[stayed].mean().item() if stayed.any() else 0
                    
                    print(f"\n[Step {step+1}] Eval accuracy:")
                    print(f"  Changed positions: {acc_changed:.1%}")
                    print(f"  Stayed positions:  {acc_stayed:.1%}")
                    print(f"  Overall:           {correct.mean().item():.1%}")
    
    # Final analysis
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - Final Analysis")
    print("=" * 60)
    
    model.eval()
    model.vq.reset_usage_tracking()
    
    # Analyze entropy distribution
    print("\nAnalyzing entropy distribution...")
    
    all_entropies = []
    all_changed = []
    
    with torch.no_grad():
        for batch_start in range(0, min(1000, obs_t.shape[0]), 32):
            batch_end = min(batch_start + 32, obs_t.shape[0])
            
            for t in range(obs_t.shape[1] - 1):
                x = obs_t[batch_start:batch_end, t]
                x_next = obs_t[batch_start:batch_end, t + 1]
                a = actions_t[batch_start:batch_end, t]
                
                results = model(x, a, x_next)
                
                # Get current and target codes
                _, curr_codes, _ = model.encode(x)
                next_target = results['next_codes_target']
                
                # Compute entropy per position
                probs = F.softmax(results['next_logits'], dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(-1)  # [B, H, W]
                
                # Track which changed
                changed = (curr_codes != next_target)
                
                all_entropies.append(entropy.cpu().numpy().flatten())
                all_changed.append(changed.cpu().numpy().flatten())
    
    all_entropies = np.concatenate(all_entropies)
    all_changed = np.concatenate(all_changed)
    
    # Convert to bits
    all_entropies = all_entropies / np.log(2)
    
    print(f"\nOverall entropy (per-position):")
    print(f"  Mean: {all_entropies.mean():.4f} bits")
    print(f"  Std:  {all_entropies.std():.4f}")
    
    ent_changed = all_entropies[all_changed]
    ent_stayed = all_entropies[~all_changed]
    
    print(f"\nEntropy vs actual code changes:")
    print(f"  Positions that CHANGED ({len(ent_changed)} samples):")
    print(f"    Mean entropy: {ent_changed.mean():.4f} bits")
    print(f"    Median:       {np.median(ent_changed):.4f} bits")
    print(f"  Positions that STAYED SAME ({len(ent_stayed)} samples):")
    print(f"    Mean entropy: {ent_stayed.mean():.4f} bits")
    print(f"    Median:       {np.median(ent_stayed):.4f} bits")
    
    if ent_stayed.mean() > 0:
        ratio = ent_changed.mean() / ent_stayed.mean()
        print(f"\n  ** RATIO: {ratio:.1f}x entropy for changed positions **")
    
    # Count deterministic vs stochastic
    det_threshold = 0.1  # bits
    n_deterministic = (all_entropies < det_threshold).sum()
    n_stochastic = (all_entropies >= det_threshold).sum()
    
    print(f"\nTransition classification:")
    print(f"  Deterministic (<{det_threshold} bits): {n_deterministic} ({100*n_deterministic/len(all_entropies):.1f}%)")
    print(f"  Stochastic (>={det_threshold} bits):   {n_stochastic} ({100*n_stochastic/len(all_entropies):.1f}%)")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state': model.state_dict(),
        'metrics': dict(metrics),
        'config': {
            'game': game,
            'n_steps': n_steps,
            'warmup_steps': warmup_steps,
        }
    }, output_path / f'{game}_vq_v3.pt')
    
    print(f"\nModel saved to {output_path / f'{game}_vq_v3.pt'}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='2048', choices=['2048', 'othello'])
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--warmup', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='outputs_vq_v3')
    parser.add_argument('--trajectories', type=int, default=2000)
    args = parser.parse_args()
    
    train(
        game=args.game,
        n_steps=args.steps,
        warmup_steps=args.warmup,
        device=args.device,
        output_dir=args.output,
        n_trajectories=args.trajectories,
    )


if __name__ == '__main__':
    main()
