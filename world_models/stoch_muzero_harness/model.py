from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AuxSpec:
    key: str
    num_classes: int
    shape: Tuple[int, ...]  # e.g., (8,8) or () for scalar


class ConvEncoder(nn.Module):
    """Simple Dreamer/MuZero-ish conv encoder for grayscale 64x64."""
    def __init__(self, in_ch: int = 1, feat_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, stride=2, padding=1), nn.ReLU(),   # 64->32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),      # 32->16
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),     # 16->8
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),    # 8->4
        )
        self.fc = nn.Linear(256 * 4 * 4, feat_dim)

    def forward(self, obs01: torch.Tensor) -> torch.Tensor:
        # obs01: [B,1,H,W] in [0,1]
        h = self.conv(obs01).reshape(obs01.shape[0], -1)
        return F.relu(self.fc(h))


class AuxHead(nn.Module):
    def __init__(self, in_dim: int, spec: AuxSpec, hidden: int = 256):
        super().__init__()
        self.spec = spec
        out_dim = spec.num_classes
        for d in spec.shape:
            out_dim *= d
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        y = self.net(x)
        if len(self.spec.shape) == 0:
            return y.view(B, self.spec.num_classes)
        prod = 1
        for d in self.spec.shape:
            prod *= d
        y = y.view(B, self.spec.num_classes, prod)
        return y.view(B, self.spec.num_classes, *self.spec.shape)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class StochMuZeroNet(nn.Module):
    """Stochastic MuZero-style network with explicit afterstate (rule core), chance, and nuisance u.

    Components:
      - representation: (s,u) = f(o)
      - afterstate: a_s = phi(s,a)
      - chance: logits = sigma(a_s)
      - dynamics: s' = g(a_s, c)
      - reward: r = r(a_s, c)
      - policy/value: from s
      - auxiliary decoders: from s (and from a_s) to symbolic state
      - style decoder: from u to style_id (forces u to carry nuisance)
    """
    def __init__(
        self,
        obs_shape: Tuple[int, int],
        action_size: int,
        chance_size: int,
        num_styles: int,
        s_dim: int = 256,
        u_dim: int = 32,
        enc_feat_dim: int = 512,
        action_emb_dim: int = 64,
        chance_emb_dim: int = 64,
        aux_specs: Optional[Dict[str, AuxSpec]] = None,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.chance_size = chance_size
        self.num_styles = num_styles
        self.s_dim = s_dim
        self.u_dim = u_dim

        self.encoder = ConvEncoder(in_ch=1, feat_dim=enc_feat_dim)
        self.s_head = nn.Linear(enc_feat_dim, s_dim)
        self.u_head = nn.Linear(enc_feat_dim, u_dim)

        self.action_emb = nn.Embedding(action_size, action_emb_dim)
        self.chance_emb = nn.Embedding(max(1, chance_size), chance_emb_dim)

        self.afterstate_net = MLP(s_dim + action_emb_dim, s_dim, hidden=256, layers=2)
        self.chance_head = MLP(s_dim, chance_size, hidden=256, layers=2) if chance_size > 1 else None
        self.dynamics_net = MLP(s_dim + chance_emb_dim, s_dim, hidden=256, layers=2)
        self.reward_head = MLP(s_dim + chance_emb_dim, 1, hidden=256, layers=2)

        self.policy_head = MLP(s_dim, action_size, hidden=256, layers=2)
        self.value_head = MLP(s_dim, 1, hidden=256, layers=2)

        self.style_head = MLP(u_dim, num_styles, hidden=128, layers=2) if num_styles > 1 else None

        self.aux_specs = aux_specs or {}
        self.aux_heads = nn.ModuleDict()
        self.after_aux_heads = nn.ModuleDict()
        for k, spec in self.aux_specs.items():
            self.aux_heads[k] = AuxHead(s_dim, spec, hidden=256)
            self.after_aux_heads[k] = AuxHead(s_dim, spec, hidden=256)

    def encode(self, obs01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(obs01)
        s = torch.tanh(self.s_head(feat))
        u = torch.tanh(self.u_head(feat))
        return s, u

    def predict_policy_value(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.policy_head(s)
        value = self.value_head(s).squeeze(-1)
        return policy_logits, value

    def afterstate(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        ae = self.action_emb(action)
        return torch.tanh(self.afterstate_net(torch.cat([s, ae], dim=-1)))

    def predict_chance_logits(self, a_s: torch.Tensor) -> torch.Tensor:
        if self.chance_size <= 1:
            # deterministic
            return torch.zeros((a_s.shape[0], 1), device=a_s.device)
        return self.chance_head(a_s)

    def next_state(self, a_s: torch.Tensor, chance: torch.Tensor) -> torch.Tensor:
        ce = self.chance_emb(chance)
        return torch.tanh(self.dynamics_net(torch.cat([a_s, ce], dim=-1)))

    def predict_reward(self, a_s: torch.Tensor, chance: torch.Tensor) -> torch.Tensor:
        ce = self.chance_emb(chance)
        r = self.reward_head(torch.cat([a_s, ce], dim=-1)).squeeze(-1)
        return r

    def decode_aux(self, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        for k, head in self.aux_heads.items():
            out[k] = head(s)
        return out

    def decode_after_aux(self, a_s: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        for k, head in self.after_aux_heads.items():
            out[k] = head(a_s)
        return out

    def predict_style_logits(self, u: torch.Tensor) -> Optional[torch.Tensor]:
        if self.style_head is None:
            return None
        return self.style_head(u)

    # Convenience: MuZero-ish inference API
    @torch.no_grad()
    def initial_inference(self, obs01: torch.Tensor):
        s, u = self.encode(obs01)
        policy_logits, value = self.predict_policy_value(s)
        aux = self.decode_aux(s)
        style_logits = self.predict_style_logits(u)
        return s, u, policy_logits, value, aux, style_logits

    @torch.no_grad()
    def recurrent_inference(self, s: torch.Tensor, action: torch.Tensor, chance: torch.Tensor):
        a_s = self.afterstate(s, action)
        chance_logits = self.predict_chance_logits(a_s)
        r = self.predict_reward(a_s, chance)
        s_next = self.next_state(a_s, chance)
        policy_logits, value = self.predict_policy_value(s_next)
        aux = self.decode_aux(s_next)
        after_aux = self.decode_after_aux(a_s)
        return a_s, chance_logits, r, s_next, policy_logits, value, aux, after_aux


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log_softmax with boolean mask (False entries get -inf)."""
    # mask: same shape as logits (or broadcastable)
    very_neg = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.tensor(very_neg, device=logits.device, dtype=logits.dtype))
    return F.log_softmax(masked, dim=dim)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.exp(masked_log_softmax(logits, mask, dim=dim))
