from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .model import StochMuZeroNet, masked_softmax


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.5
    gamma: float = 0.997
    dirichlet_alpha: float = 0.25
    root_noise_frac: float = 0.25
    chance_entropy_threshold: float = 0.5
    chance_topk: int = 0         # 0 => sample
    max_depth: int = 32


class Node:
    __slots__ = (
        "s", "prior", "N", "W", "Q",
        "children",
        "expanded",
        "cached_after",
        "cached_chance_logits",
        "cached_cur_aux",
    )

    def __init__(self, s: torch.Tensor, action_size: int):
        # s: [1, s_dim]
        self.s = s
        self.prior = torch.full((action_size,), 1.0 / action_size, device=s.device)
        self.N = torch.zeros((action_size,), device=s.device)
        self.W = torch.zeros((action_size,), device=s.device)
        self.Q = torch.zeros((action_size,), device=s.device)
        self.children: Dict[Tuple[int, int], Node] = {}
        self.expanded = False
        self.cached_after: Dict[int, torch.Tensor] = {}
        self.cached_chance_logits: Dict[int, torch.Tensor] = {}
        self.cached_cur_aux: Optional[Dict[str, torch.Tensor]] = None


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(probs, 1e-8, 1.0)
    return -(p * torch.log(p)).sum()


class StochasticMCTS:
    """Lightweight MCTS over latent states, with chance nodes via sampling or top-k enumeration."""

    def __init__(self, model: StochMuZeroNet, cfg: MCTSConfig, action_mask: Optional[np.ndarray] = None):
        self.model = model
        self.cfg = cfg
        self.action_mask = action_mask  # optional bool mask over actions at root

    @torch.no_grad()
    def run(self, obs01: torch.Tensor, rng: np.random.RandomState) -> Tuple[int, np.ndarray]:
        """Run MCTS from an observation. Returns (action, visit_probs)."""
        device = obs01.device
        s, _u, policy_logits, value, _aux, _style_logits = self.model.initial_inference(obs01)
        root = Node(s, self.model.action_size)

        # Expand root
        self._expand(root, policy_logits, value, add_noise=True, rng=rng)

        for _ in range(self.cfg.num_simulations):
            self._simulate(root, depth=0, rng=rng)

        visits = root.N.detach().cpu().numpy().astype(np.float64)
        if visits.sum() <= 0:
            # fallback to prior
            p = root.prior.detach().cpu().numpy()
            a = int(np.argmax(p))
            return a, p
        probs = visits / visits.sum()
        a = int(np.argmax(visits))
        return a, probs

    @torch.no_grad()
    def _expand(self, node: Node, policy_logits: torch.Tensor, value: torch.Tensor, add_noise: bool, rng: np.random.RandomState):
        # policy_logits: [1,A]
        mask = None
        if (not node.expanded) and (self.action_mask is not None):
            # root mask only
            mask = torch.from_numpy(self.action_mask.astype(np.bool_)).to(policy_logits.device)
            # If all false, ignore
            if not bool(mask.any()):
                mask = None
        if mask is not None:
            probs = masked_softmax(policy_logits[0], mask, dim=-1)
        else:
            probs = F.softmax(policy_logits[0], dim=-1)

        # Dirichlet noise at root
        if add_noise and self.cfg.root_noise_frac > 0.0:
            alpha = self.cfg.dirichlet_alpha
            noise = rng.dirichlet([alpha] * probs.shape[0]).astype(np.float32)
            noise_t = torch.from_numpy(noise).to(probs.device)
            probs = (1 - self.cfg.root_noise_frac) * probs + self.cfg.root_noise_frac * noise_t

        node.prior = probs
        node.expanded = True

    @torch.no_grad()
    def _select_action(self, node: Node) -> int:
        total_N = torch.sum(node.N) + 1.0
        u = self.cfg.c_puct * node.prior * torch.sqrt(total_N) / (1.0 + node.N)
        scores = node.Q + u
        return int(torch.argmax(scores).item())

    @torch.no_grad()
    def _chance_mask_from_decodes(self, cur_grid: Optional[torch.Tensor], after_grid: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Heuristic mask for 2048 chance outcomes using decoded grids (exponent classes).
        If decodes missing, returns None (no mask).
        """
        if cur_grid is None or after_grid is None:
            return None
        # only supports 2048 encoding: grid key
        # cur_grid, after_grid: [1,4,4] int64
        # changed?
        changed = bool(torch.any(cur_grid != after_grid).item())
        empties = (after_grid == 0)
        if (not changed) or (not bool(empties.any().item())):
            mask = torch.zeros((self.model.chance_size,), dtype=torch.bool, device=after_grid.device)
            mask[0] = True
            return mask
        mask = torch.zeros((self.model.chance_size,), dtype=torch.bool, device=after_grid.device)
        # 0 = no_spawn invalid
        # 1..32: pos 0..15, valbit 0/1
        for pos in range(16):
            r = pos // 4
            c = pos % 4
            if bool(empties[0, r, c].item()):
                mask[1 + 2*pos + 0] = True  # val=2
                mask[1 + 2*pos + 1] = True  # val=4
        return mask

    @torch.no_grad()
    def _simulate(self, node: Node, depth: int, rng: np.random.RandomState) -> float:
        if depth >= self.cfg.max_depth:
            # leaf value
            _pl, v = self.model.predict_policy_value(node.s)
            return float(v.item())

        if not node.expanded:
            policy_logits, v = self.model.predict_policy_value(node.s)
            self._expand(node, policy_logits.unsqueeze(0), v.unsqueeze(0), add_noise=False, rng=rng)
            return float(v.item())

        a = self._select_action(node)

        # cache current aux decode (optional)
        if node.cached_cur_aux is None:
            node.cached_cur_aux = self.model.decode_aux(node.s)

        # afterstate + chance logits
        if a not in node.cached_after:
            aa = torch.tensor([a], device=node.s.device, dtype=torch.long)
            a_s = self.model.afterstate(node.s, aa)
            node.cached_after[a] = a_s
            node.cached_chance_logits[a] = self.model.predict_chance_logits(a_s)

        a_s = node.cached_after[a]
        chance_logits = node.cached_chance_logits[a]  # [1,C]
        # decode grids for mask if possible
        cur_grid = None
        after_grid = None
        if node.cached_cur_aux is not None and "grid" in node.cached_cur_aux:
            cur_grid = torch.argmax(node.cached_cur_aux["grid"], dim=1)  # [1,4,4]
        after_aux = self.model.decode_after_aux(a_s)
        if "grid" in after_aux:
            after_grid = torch.argmax(after_aux["grid"], dim=1)  # [1,4,4]

        mask = None
        if self.model.chance_size > 1:
            mask = self._chance_mask_from_decodes(cur_grid, after_grid)
            if mask is None:
                probs = F.softmax(chance_logits[0], dim=-1)
            else:
                probs = masked_softmax(chance_logits[0], mask, dim=-1)
        else:
            probs = torch.ones((1,), device=node.s.device)

        ent = float(_entropy(probs).item()) if probs.numel() > 1 else 0.0

        # select chance
        if self.model.chance_size <= 1:
            c = 0
        else:
            if ent < self.cfg.chance_entropy_threshold:
                c = int(torch.argmax(probs).item())
            else:
                # sample
                p_np = probs.detach().cpu().numpy().astype(np.float64)
                p_np = p_np / (p_np.sum() + 1e-12)
                c = int(rng.choice(np.arange(self.model.chance_size), p=p_np))

        key = (a, c)
        if key not in node.children:
            # transition
            aa = torch.tensor([a], device=node.s.device, dtype=torch.long)
            cc = torch.tensor([c], device=node.s.device, dtype=torch.long)
            r = self.model.predict_reward(a_s, cc)
            s_next = self.model.next_state(a_s, cc)
            child = Node(s_next, self.model.action_size)
            node.children[key] = child
            # leaf expansion on child (lazy)
            v_child = self._simulate(child, depth + 1, rng=rng)
            v = float(r.item() + self.cfg.gamma * v_child)
        else:
            child = node.children[key]
            aa = torch.tensor([a], device=node.s.device, dtype=torch.long)
            cc = torch.tensor([c], device=node.s.device, dtype=torch.long)
            r = self.model.predict_reward(a_s, cc)
            v_child = self._simulate(child, depth + 1, rng=rng)
            v = float(r.item() + self.cfg.gamma * v_child)

        # backup
        node.N[a] += 1.0
        node.W[a] += v
        node.Q[a] = node.W[a] / node.N[a]
        return v
