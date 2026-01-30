"""
VQ World Model v6 - Back to Basics + Proper Diversity

Key realization: V1 WORKED! We got 14x entropy ratio for 2048.
The only problem was low codebook utilization (2%).

What went wrong in v2-v5:
- v2: Aggressive resets destabilized learning  
- v3: Warmup didn't help, still collapsed
- v4: Cell-wise was a structural hack
- v5: Pure JEPA collapsed without reconstruction anchor

Solution: Return to reconstruction-based training (it works!) 
but add proper diversity regularization instead of aggressive resets.

Architecture (same as v1):
- Encoder: image -> spatial latents
- VQ: latents -> discrete codes
- Decoder: codes -> reconstructed image
- Dynamics: predict next codes from (codes, action)

Key changes for diversity:
1. Entropy bonus: reward uniform code usage in loss
2. Soft quantization option: Gumbel-softmax for smoother gradients
3. NO aggressive resets - they destabilized v2
4. Larger codebook with temperature annealing
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict, Optional
from collections import defaultdict


# ============================================================
# VQ Layer with Entropy Regularization (no aggressive resets!)
# ============================================================

class VectorQuantizerWithDiversity(nn.Module):
    """VQ with explicit diversity encouragement, no destabilizing resets."""
    
    def __init__(self, num_codes: int, code_dim: int, 
                 commitment_cost: float = 0.25,
                 ema_decay: float = 0.99,
                 diversity_weight: float = 0.1):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.diversity_weight = diversity_weight
        
        # Initialize with more spread
        embed = torch.randn(num_codes, code_dim) * 0.5
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.ones(num_codes))
        self.register_buffer('embed_avg', embed.clone())
        
        # Track usage for monitoring (not for aggressive resets!)
        self.register_buffer('usage_count', torch.zeros(num_codes))
        
    def forward(self, z: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            z: [..., D] continuous latents
            temperature: for soft assignment (lower = harder)
        """
        orig_shape = z.shape
        flat_z = z.reshape(-1, self.code_dim)
        B = flat_z.shape[0]
        
        # Distances to all codes
        dist = (flat_z.pow(2).sum(1, keepdim=True) 
                - 2 * flat_z @ self.embed.t()
                + self.embed.pow(2).sum(1, keepdim=True).t())
        
        # Soft assignment probabilities (for diversity loss)
        soft_assign = F.softmax(-dist / max(temperature, 0.01), dim=-1)  # [B, num_codes]
        
        # Hard assignment
        _, codes = dist.min(1)
        
        # Quantize
        z_q = F.embedding(codes, self.embed)
        
        # EMA updates (standard, not aggressive)
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
                
                # Update embeddings
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + 1e-5)
                    / (n + self.num_codes * 1e-5) * n
                )
                self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))
                
                # Track usage (for monitoring only)
                self.usage_count.add_(encodings.sum(0))
        
        # === LOSSES ===
        
        # Commitment loss (standard VQ-VAE)
        commitment_loss = F.mse_loss(flat_z, z_q.detach())
        
        # Codebook loss (move codes toward inputs)
        codebook_loss = F.mse_loss(z_q, flat_z.detach())
        
        # DIVERSITY LOSS: encourage uniform code usage
        # Use soft assignments averaged over batch
        avg_soft = soft_assign.mean(0)  # [num_codes]
        # Entropy of this distribution (higher = more diverse)
        usage_entropy = -(avg_soft * (avg_soft + 1e-10).log()).sum()
        max_entropy = np.log(self.num_codes)
        # Loss is negative entropy (minimize to maximize entropy)
        diversity_loss = -usage_entropy / max_entropy
        
        # Straight-through
        z_q = flat_z + (z_q - flat_z).detach()
        
        # Reshape
        z_q = z_q.view(orig_shape)
        codes = codes.view(orig_shape[:-1])
        
        # Monitoring stats
        with torch.no_grad():
            hard_avg = F.one_hot(codes.view(-1), self.num_codes).float().mean(0)
            perplexity = torch.exp(-(hard_avg * (hard_avg + 1e-10).log()).sum())
            active = (self.cluster_size > 1).sum()
        
        info = {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'diversity_loss': diversity_loss,
            'usage_entropy': usage_entropy,
            'perplexity': perplexity,
            'active_codes': active,
        }
        
        return z_q, codes, info
    
    def get_usage_stats(self) -> Dict:
        total = self.usage_count.sum().item()
        if total > 0:
            probs = self.usage_count / total
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
        else:
            entropy = 0
        
        used = (self.usage_count > 0).sum().item()
        return {
            'active_codes': int(used),
            'total_codes': self.num_codes,
            'usage_ratio': used / self.num_codes,
            'entropy_bits': entropy / np.log(2),
        }
    
    def reset_usage_stats(self):
        self.usage_count.zero_()


# ============================================================
# Encoder / Decoder
# ============================================================

class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, 
                 latent_dim: int = 64, grid_size: int = 4):
        super().__init__()
        self.grid_size = grid_size
        
        # 64 -> 32 -> 16 -> 8 -> 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, latent_dim, 4, 2, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 1, hidden_dim: int = 64, 
                 latent_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ============================================================
# Dynamics Model
# ============================================================

class DynamicsPredictor(nn.Module):
    """Predict distribution over next codes."""
    
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
        """Returns logits [B, H, W, num_codes]."""
        B = codes.shape[0]
        
        code_emb = self.code_embed(codes).reshape(B, -1)
        action_emb = self.action_embed(action)
        
        x = torch.cat([code_emb, action_emb], dim=-1)
        logits = self.net(x)
        
        return logits.reshape(B, self.grid_size, self.grid_size, self.num_codes)


# ============================================================
# Full Model
# ============================================================

class VQWorldModel(nn.Module):
    def __init__(self, num_codes: int = 256, num_actions: int = 4,
                 latent_dim: int = 64, hidden_dim: int = 64,
                 grid_size: int = 4, diversity_weight: float = 0.1):
        super().__init__()
        self.num_codes = num_codes
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.vq = VectorQuantizerWithDiversity(
            num_codes, latent_dim, diversity_weight=diversity_weight
        )
        self.decoder = Decoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.dynamics = DynamicsPredictor(num_codes, num_actions, grid_size)
        
    def encode(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        z = self.encoder(x)  # [B, D, H, W]
        z_for_vq = z.permute(0, 2, 3, 1)  # [B, H, W, D]
        z_q, codes, vq_info = self.vq(z_for_vq, temperature)
        z_q = z_q.permute(0, 3, 1, 2)  # [B, D, H, W]
        return z_q, codes, vq_info
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)
    
    def forward(self, x: torch.Tensor, action: Optional[torch.Tensor] = None,
                x_next: Optional[torch.Tensor] = None, temperature: float = 1.0) -> Dict:
        # Encode current
        z_q, codes, vq_info = self.encode(x, temperature)
        
        # Reconstruct
        recon = self.decode(z_q)
        recon_loss = F.mse_loss(recon, x)
        
        results = {
            'recon': recon,
            'recon_loss': recon_loss,
            'codes': codes,
            'z_q': z_q,
            **vq_info,
        }
        
        if action is not None and x_next is not None:
            # Get target codes
            with torch.no_grad():
                _, codes_next, _ = self.encode(x_next, temperature)
            
            # Predict next codes
            logits = self.dynamics(codes, action)
            trans_loss = F.cross_entropy(
                logits.reshape(-1, self.num_codes),
                codes_next.reshape(-1)
            )
            
            # Prediction entropy per position
            probs = F.softmax(logits, dim=-1)
            pred_entropy = -(probs * (probs + 1e-10).log()).sum(-1)
            
            # Accuracy
            pred_codes = logits.argmax(dim=-1)
            accuracy = (pred_codes == codes_next).float().mean()
            
            # Changed mask
            changed_mask = (codes != codes_next)
            
            results.update({
                'trans_loss': trans_loss,
                'pred_entropy': pred_entropy,
                'accuracy': accuracy,
                'codes_next': codes_next,
                'pred_codes': pred_codes,
                'changed_mask': changed_mask,
            })
        
        return results


# ============================================================
# Data Generation
# ============================================================

def generate_2048_trajectories(n_trajectories: int, max_steps: int, img_size: int = 64):
    try:
        from ..grid.game_2048 import Game2048
        from ..grid.renderer import GameRenderer
    except ImportError:
        from world_models.grid.game_2048 import Game2048
        from world_models.grid.renderer import GameRenderer
    
    renderer = GameRenderer(img_size)
    all_obs, all_actions = [], []
    
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
    
    obs = np.stack(all_obs)[:, :max_steps+1, np.newaxis, :, :]
    actions = np.stack(all_actions)[:, :max_steps]
    return obs.astype(np.float32), actions.astype(np.int64)


def generate_othello_trajectories(n_trajectories: int, max_steps: int, img_size: int = 64):
    try:
        from ..grid.othello import OthelloGame
        from ..grid.renderer import GameRenderer
    except ImportError:
        from world_models.grid.othello import OthelloGame
        from world_models.grid.renderer import GameRenderer
    
    renderer = GameRenderer(img_size)
    all_obs, all_actions = [], []
    
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
    
    obs = np.stack(all_obs)[:, :max_steps+1, np.newaxis, :, :]
    actions = np.stack(all_actions)[:, :max_steps]
    return obs.astype(np.float32), actions.astype(np.int64)


# ============================================================
# Training
# ============================================================

def train(game: str, n_steps: int, device: str, output_dir: str,
          n_trajectories: int = 2000, traj_len: int = 50,
          batch_size: int = 64, num_codes: int = 256,
          diversity_weight: float = 0.5,  # Higher than before!
          recon_weight: float = 1.0,
          trans_weight: float = 1.0,
          commitment_weight: float = 0.25):
    
    print("=" * 60)
    print("VQ World Model v6 - Back to Basics + Diversity")
    print(f"Game: {game} | Steps: {n_steps}")
    print("=" * 60)
    print("\nKey insight: V1 worked (14x entropy ratio)!")
    print("Problem was only codebook utilization (2%).")
    print("Fix: Proper diversity loss, NO aggressive resets.\n")
    print(f"Weights: recon={recon_weight}, trans={trans_weight}, "
          f"diversity={diversity_weight}, commit={commitment_weight}\n")
    
    # Generate data
    if game == '2048':
        obs, actions = generate_2048_trajectories(n_trajectories, traj_len)
        num_actions = 4
    else:
        obs, actions = generate_othello_trajectories(n_trajectories, traj_len)
        num_actions = 64
    
    print(f"Data: {obs.shape[0]} trajectories, {obs.shape[1]-1} steps\n")
    
    # Create model
    model = VQWorldModel(
        num_codes=num_codes,
        num_actions=num_actions,
        diversity_weight=diversity_weight,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    obs_t = torch.from_numpy(obs).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    
    # Temperature annealing: start soft, end hard
    temp_start, temp_end = 2.0, 0.5
    
    pbar = tqdm(range(n_steps))
    metrics = defaultdict(list)
    
    for step in pbar:
        model.train()
        
        # Temperature schedule
        progress = step / n_steps
        temperature = temp_start + (temp_end - temp_start) * progress
        
        # Sample batch
        batch_idx = torch.randint(0, obs_t.shape[0], (batch_size,))
        time_idx = torch.randint(0, obs_t.shape[1] - 1, (batch_size,))
        
        x = obs_t[batch_idx, time_idx]
        x_next = obs_t[batch_idx, time_idx + 1]
        a = actions_t[batch_idx, time_idx]
        
        # Forward
        results = model(x, a, x_next, temperature)
        
        # Combined loss
        loss = (recon_weight * results['recon_loss'] +
                trans_weight * results['trans_loss'] +
                commitment_weight * results['commitment_loss'] +
                diversity_weight * results['diversity_loss'])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Log
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'recon': f"{results['recon_loss'].item():.4f}",
            'trans': f"{results['trans_loss'].item():.4f}",
            'acc': f"{results['accuracy'].item():.1%}",
            'codes': f"{results['active_codes'].item():.0f}",
            'temp': f"{temperature:.2f}",
        })
        
        if step % 100 == 0:
            metrics['step'].append(step)
            for k in ['recon_loss', 'trans_loss', 'accuracy', 'diversity_loss']:
                metrics[k].append(results[k].item())
            metrics['active_codes'].append(results['active_codes'].item())
        
        # Detailed logging
        if (step + 1) % 2000 == 0:
            stats = model.vq.get_usage_stats()
            
            changed = results['changed_mask']
            correct = (results['pred_codes'] == results['codes_next'])
            
            acc_changed = correct[changed].float().mean().item() if changed.any() else 0
            acc_stayed = correct[~changed].float().mean().item() if (~changed).any() else 0
            n_changed = changed.sum().item()
            pct_changed = 100 * n_changed / changed.numel()
            
            # Entropy analysis
            pred_ent = results['pred_entropy']
            ent_changed = pred_ent[changed].mean().item() / np.log(2) if changed.any() else 0
            ent_stayed = pred_ent[~changed].mean().item() / np.log(2) if (~changed).any() else 0
            
            print(f"\n[Step {step+1}] temp={temperature:.2f}")
            print(f"  Codebook: {stats['active_codes']}/{stats['total_codes']} "
                  f"({100*stats['usage_ratio']:.1f}%), entropy={stats['entropy_bits']:.2f} bits")
            print(f"  Recon: {results['recon_loss'].item():.4f}, Trans: {results['trans_loss'].item():.4f}")
            print(f"  Accuracy: {results['accuracy'].item():.1%}")
            print(f"    Changed: {acc_changed:.1%} (n={n_changed}, {pct_changed:.1f}%)")
            print(f"    Stayed:  {acc_stayed:.1%}")
            print(f"  Prediction entropy (bits):")
            print(f"    Changed: {ent_changed:.3f}")
            print(f"    Stayed:  {ent_stayed:.3f}")
            if ent_stayed > 0.001:
                print(f"    RATIO:   {ent_changed/ent_stayed:.1f}x")
            
            model.vq.reset_usage_stats()
    
    # Final analysis
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - Final Analysis")
    print("=" * 60)
    
    model.eval()
    model.vq.reset_usage_stats()
    
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
                
                results = model(x, a, x_next, temperature=0.5)
                
                entropy = results['pred_entropy'].cpu().numpy().flatten()
                changed = results['changed_mask'].cpu().numpy().flatten()
                correct = (results['pred_codes'] == results['codes_next']).cpu().numpy().flatten()
                
                all_entropies.append(entropy)
                all_changed.append(changed)
                all_correct.append(correct)
    
    all_entropies = np.concatenate(all_entropies) / np.log(2)
    all_changed = np.concatenate(all_changed)
    all_correct = np.concatenate(all_correct)
    
    stats = model.vq.get_usage_stats()
    
    print(f"\nCodebook: {stats['active_codes']}/{stats['total_codes']} "
          f"({100*stats['usage_ratio']:.1f}%), entropy={stats['entropy_bits']:.2f} bits")
    print(f"Final accuracy: {100*all_correct.mean():.1f}%")
    
    print(f"\nPosition changes: {all_changed.sum()} ({100*all_changed.mean():.1f}%)")
    
    ent_changed = all_entropies[all_changed]
    ent_stayed = all_entropies[~all_changed]
    
    print(f"\nPrediction entropy (bits):")
    if len(ent_changed) > 0:
        print(f"  CHANGED ({len(ent_changed)} samples):")
        print(f"    Mean: {ent_changed.mean():.4f}, Median: {np.median(ent_changed):.4f}")
    if len(ent_stayed) > 0:
        print(f"  STAYED ({len(ent_stayed)} samples):")
        print(f"    Mean: {ent_stayed.mean():.4f}, Median: {np.median(ent_stayed):.4f}")
    
    if len(ent_changed) > 0 and len(ent_stayed) > 0 and ent_stayed.mean() > 0.001:
        ratio = ent_changed.mean() / ent_stayed.mean()
        print(f"\n  ** ENTROPY RATIO: {ratio:.1f}x (changed/stayed) **")
        if game == '2048':
            print(f"  (V1 got 14x - this measures stochasticity detection)")
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state': model.state_dict(),
        'metrics': dict(metrics),
        'final_stats': {
            'codebook_usage': stats,
            'accuracy': float(all_correct.mean()),
            'entropy_ratio': float(ent_changed.mean() / max(ent_stayed.mean(), 1e-6)) if len(ent_changed) > 0 else 0,
        }
    }, output_path / f'{game}_vq_v6.pt')
    
    print(f"\nSaved to {output_path / f'{game}_vq_v6.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='2048', choices=['2048', 'othello'])
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='outputs_vq_v6')
    parser.add_argument('--trajectories', type=int, default=2000)
    parser.add_argument('--num_codes', type=int, default=256)
    parser.add_argument('--diversity_weight', type=float, default=0.5)
    args = parser.parse_args()
    
    train(
        game=args.game,
        n_steps=args.steps,
        device=args.device,
        output_dir=args.output,
        n_trajectories=args.trajectories,
        num_codes=args.num_codes,
        diversity_weight=args.diversity_weight,
    )


if __name__ == '__main__':
    main()
