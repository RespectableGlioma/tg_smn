"""
Rule Extraction and Visualization for VQ-VAE World Models

This module extracts the learned rules from a trained VQ world model and
visualizes them to understand what the model has learned.

Key visualizations:
1. Codebook gallery - what does each code represent visually?
2. Transition graph - directed graph showing (code, action) → next_code
3. Entropy heatmap - which transitions are deterministic vs stochastic?
4. Rule summary - list of discovered deterministic rules

Usage:
    from world_models.stoch_muzero.rule_extraction import RuleExtractor
    
    extractor = RuleExtractor(model, device)
    extractor.analyze_codebook(obs_samples)
    extractor.extract_transitions(obs, actions)
    extractor.visualize_all(save_dir='rule_analysis/')
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TransitionRule:
    """A discovered transition rule."""
    from_code: int
    action: int
    to_code: int
    count: int
    entropy: float
    is_deterministic: bool
    
    def __repr__(self):
        det_str = "RULE" if self.is_deterministic else "CHANCE"
        return f"{det_str}: code_{self.from_code} + action_{self.action} → code_{self.to_code} (n={self.count}, H={self.entropy:.3f})"


class RuleExtractor:
    """
    Extract and visualize rules from a trained VQ world model.
    
    The key insight: deterministic transitions (rules) have LOW entropy,
    stochastic transitions (chance) have HIGH entropy.
    """
    
    def __init__(
        self, 
        model, 
        device: torch.device,
        entropy_threshold: float = 0.1,  # bits
    ):
        self.model = model
        self.device = device
        self.entropy_threshold = entropy_threshold
        self.cfg = model.cfg
        
        # Will be populated by analysis
        self.codebook_images = None  # [n_codes, H, W] representative images
        self.codebook_usage = None   # [n_codes] usage counts
        self.transition_counts = None  # [n_codes, n_actions, n_codes]
        self.transition_entropy = None  # [n_codes, n_actions]
        self.rules = []
        
    def analyze_codebook(
        self, 
        obs_samples: torch.Tensor,
        n_samples_per_code: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Analyze what each codebook entry represents.
        
        For each code, find observations that map to it and compute:
        - Representative image (mean of observations using this code)
        - Usage frequency
        - Variance (how consistent is this code?)
        
        Args:
            obs_samples: [N, C, H, W] observation samples
            
        Returns:
            Dictionary with codebook analysis
        """
        self.model.eval()
        
        n_codes = self.cfg.codebook_size
        code_dim = self.cfg.code_dim
        
        # Track which observations map to each code
        code_to_obs = defaultdict(list)
        code_to_positions = defaultdict(list)
        
        with torch.no_grad():
            for i in range(0, len(obs_samples), 32):
                batch = obs_samples[i:i+32].to(self.device)
                enc = self.model.encode(batch, training=False)
                indices = enc['indices'].cpu().numpy()  # [B, N]
                
                B, N = indices.shape
                H = W = int(math.sqrt(N))
                
                for b in range(B):
                    obs_np = batch[b, 0].cpu().numpy()  # [H, W]
                    
                    for pos in range(N):
                        code_idx = indices[b, pos]
                        
                        # Extract the patch this code represents
                        py, px = pos // W, pos % W
                        patch_h = obs_np.shape[0] // H
                        patch_w = obs_np.shape[1] // W
                        
                        y0, x0 = py * patch_h, px * patch_w
                        patch = obs_np[y0:y0+patch_h, x0:x0+patch_w]
                        
                        code_to_obs[code_idx].append(patch)
                        code_to_positions[code_idx].append((b, pos))
        
        # Compute representative images
        img_h = obs_samples.shape[2] // int(math.sqrt(enc['indices'].shape[1]))
        img_w = obs_samples.shape[3] // int(math.sqrt(enc['indices'].shape[1]))
        
        self.codebook_images = np.zeros((n_codes, img_h, img_w))
        self.codebook_usage = np.zeros(n_codes)
        self.codebook_variance = np.zeros(n_codes)
        
        for code_idx in range(n_codes):
            patches = code_to_obs[code_idx]
            self.codebook_usage[code_idx] = len(patches)
            
            if len(patches) > 0:
                patches_arr = np.stack(patches[:100])  # Limit for memory
                self.codebook_images[code_idx] = patches_arr.mean(axis=0)
                self.codebook_variance[code_idx] = patches_arr.var()
        
        self.model.train()
        
        return {
            'images': self.codebook_images,
            'usage': self.codebook_usage,
            'variance': self.codebook_variance,
            'n_active': int((self.codebook_usage > 0).sum()),
        }
    
    def extract_transitions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        max_samples: int = 5000,
    ) -> Dict[str, np.ndarray]:
        """
        Extract empirical transition statistics.
        
        For each (code, action) pair, track:
        - Distribution over next codes
        - Entropy of the distribution
        - Whether it's deterministic (low entropy) or stochastic
        
        Args:
            obs: [N, T+1, C, H, W] observation sequences
            actions: [N, T] action sequences
            
        Returns:
            Dictionary with transition analysis
        """
        self.model.eval()
        
        n_codes = self.cfg.codebook_size
        n_actions = self.cfg.n_actions
        
        # Count transitions: [from_code, action, to_code]
        self.transition_counts = np.zeros((n_codes, n_actions, n_codes), dtype=np.float32)
        
        # Also track model-predicted entropy
        predicted_entropy = []
        actual_changed = []
        
        n_processed = 0
        
        with torch.no_grad():
            for i in range(min(max_samples, len(obs))):
                obs_seq = obs[i:i+1].to(self.device)
                act_seq = actions[i:i+1].to(self.device)
                
                B, Tp1, C, H, W = obs_seq.shape
                T = Tp1 - 1
                
                # Encode all frames
                obs_flat = obs_seq.reshape(B * Tp1, C, H, W)
                enc = self.model.encode(obs_flat, training=False)
                all_indices = enc['indices'].reshape(B, Tp1, -1).cpu().numpy()  # [1, T+1, N]
                
                for t in range(T):
                    curr_codes = all_indices[0, t]      # [N]
                    next_codes = all_indices[0, t + 1]  # [N]
                    action = act_seq[0, t].item()
                    
                    # Get model's predicted entropy
                    z_q = self.model.quantizer.embedding(
                        torch.from_numpy(curr_codes).to(self.device).unsqueeze(0)
                    )
                    step_result = self.model.step(z_q, act_seq[:, t], sample=False)
                    ent_per_pos = step_result['logits']
                    probs = F.softmax(ent_per_pos, dim=-1)
                    log_probs = F.log_softmax(ent_per_pos, dim=-1)
                    ent = -(probs * log_probs).sum(dim=-1) / math.log(2)
                    ent = ent[0].cpu().numpy()  # [N]
                    
                    # Count transitions per position
                    for pos in range(len(curr_codes)):
                        c_from = curr_codes[pos]
                        c_to = next_codes[pos]
                        self.transition_counts[c_from, action, c_to] += 1
                        
                        predicted_entropy.append(ent[pos])
                        actual_changed.append(c_from != c_to)
                
                n_processed += 1
        
        # Compute transition probabilities and entropy
        counts_sum = self.transition_counts.sum(axis=-1, keepdims=True)
        counts_sum = np.maximum(counts_sum, 1)  # Avoid division by zero
        
        self.transition_probs = self.transition_counts / counts_sum
        
        # Entropy per (code, action) pair
        log_probs = np.log(self.transition_probs + 1e-10)
        self.transition_entropy = -(self.transition_probs * log_probs).sum(axis=-1) / math.log(2)
        
        # Extract rules
        self.rules = []
        for code_from in range(n_codes):
            for action in range(n_actions):
                count = self.transition_counts[code_from, action].sum()
                if count < 5:  # Need enough samples
                    continue
                
                entropy = self.transition_entropy[code_from, action]
                is_deterministic = entropy < self.entropy_threshold
                
                # Find most likely next code
                to_code = self.transition_probs[code_from, action].argmax()
                
                rule = TransitionRule(
                    from_code=code_from,
                    action=action,
                    to_code=to_code,
                    count=int(count),
                    entropy=float(entropy),
                    is_deterministic=is_deterministic,
                )
                self.rules.append(rule)
        
        self.model.train()
        
        return {
            'transition_counts': self.transition_counts,
            'transition_probs': self.transition_probs,
            'transition_entropy': self.transition_entropy,
            'n_rules': len([r for r in self.rules if r.is_deterministic]),
            'n_chance': len([r for r in self.rules if not r.is_deterministic]),
            'predicted_entropy': np.array(predicted_entropy),
            'actual_changed': np.array(actual_changed),
        }
    
    def get_rule_summary(self) -> str:
        """Get a text summary of discovered rules."""
        if not self.rules:
            return "No rules extracted yet. Call extract_transitions() first."
        
        deterministic = [r for r in self.rules if r.is_deterministic]
        stochastic = [r for r in self.rules if not r.is_deterministic]
        
        lines = [
            "=" * 60,
            "RULE EXTRACTION SUMMARY",
            "=" * 60,
            f"Total transitions analyzed: {len(self.rules)}",
            f"Deterministic (RULES): {len(deterministic)} ({100*len(deterministic)/len(self.rules):.1f}%)",
            f"Stochastic (CHANCE): {len(stochastic)} ({100*len(stochastic)/len(self.rules):.1f}%)",
            "",
            "Top 20 most common DETERMINISTIC rules:",
            "-" * 40,
        ]
        
        for rule in sorted(deterministic, key=lambda r: -r.count)[:20]:
            lines.append(f"  {rule}")
        
        lines.extend([
            "",
            "Top 10 most common STOCHASTIC transitions:",
            "-" * 40,
        ])
        
        for rule in sorted(stochastic, key=lambda r: -r.count)[:10]:
            lines.append(f"  {rule}")
        
        return "\n".join(lines)
    
    def visualize_codebook(
        self,
        save_path: Optional[str] = None,
        n_cols: int = 16,
        top_k: int = 64,
    ):
        """
        Visualize the codebook as a grid of representative patches.
        
        Shows the top_k most used codes.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        if self.codebook_images is None:
            print("Call analyze_codebook() first")
            return
        
        # Sort by usage
        sorted_indices = np.argsort(-self.codebook_usage)[:top_k]
        
        n_rows = (top_k + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(sorted_indices):
                code_idx = sorted_indices[idx]
                img = self.codebook_images[code_idx]
                usage = self.codebook_usage[code_idx]
                
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'{code_idx}\n({int(usage)})', fontsize=6)
            ax.axis('off')
        
        plt.suptitle(f'Top {top_k} Most Used Codes (code_id, usage_count)', fontsize=10)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def visualize_entropy_heatmap(
        self,
        save_path: Optional[str] = None,
        top_k_codes: int = 32,
    ):
        """
        Visualize transition entropy as a heatmap.
        
        Rows = codes, Columns = actions
        Color = entropy (blue=deterministic, red=stochastic)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        if self.transition_entropy is None:
            print("Call extract_transitions() first")
            return
        
        # Select top-k most used codes
        if self.codebook_usage is not None:
            sorted_codes = np.argsort(-self.codebook_usage)[:top_k_codes]
        else:
            # Use codes with most transitions
            code_counts = self.transition_counts.sum(axis=(1, 2))
            sorted_codes = np.argsort(-code_counts)[:top_k_codes]
        
        # Extract entropy for selected codes
        entropy_subset = self.transition_entropy[sorted_codes, :]
        
        fig, ax = plt.subplots(figsize=(8, 10))
        
        im = ax.imshow(entropy_subset, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=2)
        
        ax.set_xlabel('Action')
        ax.set_ylabel('Code ID')
        ax.set_yticks(range(len(sorted_codes)))
        ax.set_yticklabels(sorted_codes, fontsize=6)
        ax.set_xticks(range(self.cfg.n_actions))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Entropy (bits)')
        
        # Add threshold line
        ax.axhline(y=-0.5, color='white', linewidth=0.5)
        
        plt.title('Transition Entropy Heatmap\n(Blue=Deterministic/Rule, Red=Stochastic/Chance)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def visualize_entropy_distribution(
        self,
        save_path: Optional[str] = None,
    ):
        """
        Visualize the distribution of transition entropies.
        
        Ideally shows bimodal distribution for games with both
        deterministic rules and stochastic chance.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        if self.transition_entropy is None:
            print("Call extract_transitions() first")
            return
        
        # Flatten entropy for all (code, action) pairs with sufficient samples
        entropies = []
        for code in range(self.cfg.codebook_size):
            for action in range(self.cfg.n_actions):
                count = self.transition_counts[code, action].sum()
                if count >= 5:
                    entropies.append(self.transition_entropy[code, action])
        
        entropies = np.array(entropies)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        ax = axes[0]
        ax.hist(entropies, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=self.entropy_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({self.entropy_threshold} bits)')
        ax.set_xlabel('Entropy (bits)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Transition Entropies')
        ax.legend()
        
        # Pie chart of deterministic vs stochastic
        ax = axes[1]
        n_det = (entropies < self.entropy_threshold).sum()
        n_stoch = (entropies >= self.entropy_threshold).sum()
        ax.pie([n_det, n_stoch], labels=[f'Deterministic\n({n_det})', f'Stochastic\n({n_stoch})'],
               colors=['steelblue', 'coral'], autopct='%1.1f%%')
        ax.set_title('Rule vs Chance Transitions')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
        # Print stats
        print(f"\nEntropy Statistics:")
        print(f"  Mean: {entropies.mean():.3f} bits")
        print(f"  Std:  {entropies.std():.3f} bits")
        print(f"  Min:  {entropies.min():.3f} bits")
        print(f"  Max:  {entropies.max():.3f} bits")
        print(f"  Deterministic (<{self.entropy_threshold}): {100*n_det/len(entropies):.1f}%")
    
    def visualize_transition_graph(
        self,
        save_path: Optional[str] = None,
        top_k_codes: int = 20,
        min_count: int = 10,
    ):
        """
        Visualize transitions as a directed graph.
        
        Nodes = codes
        Edges = transitions (color by entropy)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        if self.transition_counts is None:
            print("Call extract_transitions() first")
            return
        
        # Get top-k codes by usage
        if self.codebook_usage is not None:
            top_codes = set(np.argsort(-self.codebook_usage)[:top_k_codes])
        else:
            code_counts = self.transition_counts.sum(axis=(1, 2))
            top_codes = set(np.argsort(-code_counts)[:top_k_codes])
        
        # Build edge list
        edges = []
        for rule in self.rules:
            if rule.from_code in top_codes and rule.to_code in top_codes:
                if rule.count >= min_count:
                    edges.append(rule)
        
        # Simple circular layout
        codes = list(top_codes)
        n = len(codes)
        positions = {}
        for i, code in enumerate(codes):
            angle = 2 * math.pi * i / n
            positions[code] = (math.cos(angle), math.sin(angle))
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Draw edges
        for rule in edges:
            x1, y1 = positions[rule.from_code]
            x2, y2 = positions[rule.to_code]
            
            # Color by entropy
            if rule.is_deterministic:
                color = 'steelblue'
                alpha = 0.6
            else:
                color = 'coral'
                alpha = 0.4
            
            # Skip self-loops for clarity
            if rule.from_code != rule.to_code:
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=color, alpha=alpha,
                                         connectionstyle='arc3,rad=0.1'))
        
        # Draw nodes
        for code in codes:
            x, y = positions[code]
            usage = self.codebook_usage[code] if self.codebook_usage is not None else 0
            size = 100 + usage / 10
            ax.scatter([x], [y], s=size, c='white', edgecolors='black', zorder=5)
            ax.text(x, y, str(code), ha='center', va='center', fontsize=8)
        
        # Legend
        blue_patch = mpatches.Patch(color='steelblue', label='Deterministic (Rule)')
        red_patch = mpatches.Patch(color='coral', label='Stochastic (Chance)')
        ax.legend(handles=[blue_patch, red_patch], loc='upper right')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Transition Graph (Top {top_k_codes} codes)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def visualize_all(
        self,
        save_dir: str = 'rule_analysis',
        game_name: str = 'game',
    ):
        """Generate all visualizations and save to directory."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print(f"RULE ANALYSIS FOR {game_name.upper()}")
        print("=" * 60)
        
        # Print rule summary
        print(self.get_rule_summary())
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        self.visualize_codebook(
            save_path=os.path.join(save_dir, f'{game_name}_codebook.png')
        )
        
        self.visualize_entropy_heatmap(
            save_path=os.path.join(save_dir, f'{game_name}_entropy_heatmap.png')
        )
        
        self.visualize_entropy_distribution(
            save_path=os.path.join(save_dir, f'{game_name}_entropy_dist.png')
        )
        
        self.visualize_transition_graph(
            save_path=os.path.join(save_dir, f'{game_name}_transition_graph.png')
        )
        
        # Save rule summary
        summary_path = os.path.join(save_dir, f'{game_name}_rules.txt')
        with open(summary_path, 'w') as f:
            f.write(self.get_rule_summary())
        print(f"\nSaved rule summary: {summary_path}")


def analyze_model(
    model,
    obs: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    game_name: str = 'game',
    save_dir: str = 'rule_analysis',
):
    """
    Convenience function to run full rule extraction analysis.
    
    Args:
        model: Trained VQWorldModel
        obs: [N, T+1, C, H, W] observation sequences
        actions: [N, T] action sequences
        device: torch device
        game_name: Name for saving files
        save_dir: Directory for outputs
    """
    print(f"\nAnalyzing model for {game_name}...")
    
    extractor = RuleExtractor(model, device)
    
    # Analyze codebook
    print("Analyzing codebook...")
    obs_flat = obs[:, 0].to(device)  # Use first frames
    codebook_stats = extractor.analyze_codebook(obs_flat)
    print(f"  Active codes: {codebook_stats['n_active']}/{model.cfg.codebook_size}")
    
    # Extract transitions
    print("Extracting transitions...")
    trans_stats = extractor.extract_transitions(obs, actions)
    print(f"  Deterministic rules: {trans_stats['n_rules']}")
    print(f"  Stochastic transitions: {trans_stats['n_chance']}")
    
    # Visualize
    extractor.visualize_all(save_dir=save_dir, game_name=game_name)
    
    return extractor


# Quick test
if __name__ == '__main__':
    print("Rule extraction module loaded successfully")
    print("Use analyze_model() for full analysis")
