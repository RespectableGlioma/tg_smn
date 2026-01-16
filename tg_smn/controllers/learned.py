from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from ..config import LearnedCtrlCfgLM


class ObsNorm:
    """Running mean/var normalization for controller observations."""

    def __init__(self, dim: int, eps: float = 1e-5, clip: float = 5.0, device: str = "cpu"):
        self.dim = dim
        self.eps = eps
        self.clip = clip
        self.device = device
        self.n = 0
        self.mean = torch.zeros(dim, device=device)
        self.M2 = torch.zeros(dim, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach()
        self.n += 1
        if self.n == 1:
            self.mean.copy_(x)
            self.M2.zero_()
        else:
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

    @torch.no_grad()
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.n < 2:
            return torch.clamp(x, -self.clip, self.clip)
        var = self.M2 / (self.n - 1)
        x = (x - self.mean) / torch.sqrt(var + self.eps)
        return torch.clamp(x, -self.clip, self.clip)


def build_obs(
    loss: float,
    ema_loss: float,
    delta_t: float,
    delta_rho: float,
    sqrt2kl: float,
    eta: float,
    k_prev: float,
    replay_prev: float,
    device: str,
) -> torch.Tensor:
    """Observation vector used by the learned controller."""
    eps = 1e-8
    x = torch.tensor(
        [
            math.log(max(loss, eps)),
            math.log(max(ema_loss, eps)),
            float(delta_t),
            math.log(max(delta_rho, eps)),
            math.log(max(sqrt2kl, eps)),
            float(eta),
            float(k_prev),
            float(replay_prev),
        ],
        device=device,
        dtype=torch.float32,
    )
    return x


class GRUController(nn.Module):
    """Recurrent policy + value network.

    Actions:
      - k: categorical over {k_min..k_max}
      - replay_ratio: Beta -> [0, replay_max]
      - local_router_noise: Beta -> [0, noise_max]
      - local_router_temp: Beta -> [temp_min, temp_max]
    """

    def __init__(self, cfg: LearnedCtrlCfgLM, obs_dim: int = 8, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.h = cfg.hidden_size
        self.device = device

        self.inp = nn.Linear(obs_dim, self.h)
        self.gru = nn.GRUCell(self.h, self.h)

        self.k_logits = nn.Linear(self.h, cfg.k_max - cfg.k_min + 1)
        self.cont_head = nn.Linear(self.h, 6)  # 3 actions * (alpha,beta)
        self.v_head = nn.Linear(self.h, 1)

        nn.init.zeros_(self.cont_head.weight)
        nn.init.zeros_(self.cont_head.bias)

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.h, device=self.device)

    def step(self, obs: torch.Tensor, h: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        z = torch.tanh(self.inp(obs))
        h2 = self.gru(z, h)

        logits_k = self.k_logits(h2)
        cont_raw = self.cont_head(h2)
        a1, b1, a2, b2, a3, b3 = cont_raw.chunk(6, dim=-1)

        alpha_replay = F.softplus(a1) + 1.0
        beta_replay = F.softplus(b1) + 1.0
        alpha_noise = F.softplus(a2) + 1.0
        beta_noise = F.softplus(b2) + 1.0
        alpha_temp = F.softplus(a3) + 1.0
        beta_temp = F.softplus(b3) + 1.0

        v = self.v_head(h2).squeeze(-1)

        dists = {
            "k": D.Categorical(logits=logits_k),
            "replay": D.Beta(alpha_replay, beta_replay),
            "noise": D.Beta(alpha_noise, beta_noise),
            "temp": D.Beta(alpha_temp, beta_temp),
        }
        return dists, v, h2


def select_action(cfg: LearnedCtrlCfgLM, dists, deterministic: bool = False):
    """Sample or take mean actions and return (action_dict, raw_action_tensors, logprob, entropy)."""
    if deterministic:
        k_idx = torch.argmax(dists["k"].probs, dim=-1)
    else:
        k_idx = dists["k"].sample()
    k = cfg.k_min + int(k_idx.item())

    def beta_action(name: str):
        if deterministic:
            u = dists[name].mean
        else:
            u = dists[name].sample()
        u = torch.clamp(u, 1e-4, 1 - 1e-4)
        lp = dists[name].log_prob(u).sum()
        ent = dists[name].entropy().sum()
        return u.detach(), lp.detach(), ent.detach()

    u_rep, lp_rep, ent_rep = beta_action("replay")
    u_noise, lp_noise, ent_noise = beta_action("noise")
    u_temp, lp_temp, ent_temp = beta_action("temp")

    replay_ratio = float((u_rep * cfg.replay_max).item())
    noise = float((u_noise * cfg.noise_max).item())
    temp = float((cfg.temp_min + u_temp * (cfg.temp_max - cfg.temp_min)).item())

    lp_k = dists["k"].log_prob(k_idx).sum().detach()
    ent_k = dists["k"].entropy().sum().detach()

    logprob = float((lp_k + lp_rep + lp_noise + lp_temp).item())
    entropy = float((ent_k + ent_rep + ent_noise + ent_temp).item())

    action = {"k": k, "replay_ratio": replay_ratio, "noise": noise, "temp": temp}
    raw_action = (k_idx.detach().squeeze(), u_rep.squeeze(), u_noise.squeeze(), u_temp.squeeze())
    return action, raw_action, logprob, entropy


class Rollout:
    def __init__(self):
        self.obs: List[torch.Tensor] = []
        self.actions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []

    def clear(self) -> None:
        self.__init__()


def controller_update(
    controller: GRUController,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    h0: torch.Tensor,
    obs_norm: ObsNorm,
    cfg: LearnedCtrlCfgLM,
) -> torch.Tensor:
    """A2C-style update over a rollout window with truncated BPTT."""

    if len(rollout.rewards) == 0:
        return h0.detach()

    rewards = torch.tensor(rollout.rewards, device=h0.device, dtype=torch.float32)
    dones = torch.tensor(rollout.dones, device=h0.device, dtype=torch.float32)

    h = h0
    logps = []
    ents = []
    vals = []

    for t in range(len(rollout.obs)):
        obs_t = rollout.obs[t]
        obs_n = obs_norm.normalize(obs_t)
        dists, v, h = controller.step(obs_n, h)

        k_idx, u_rep, u_noise, u_temp = rollout.actions[t]

        lp = dists["k"].log_prob(k_idx)
        lp = lp + dists["replay"].log_prob(u_rep).sum()
        lp = lp + dists["noise"].log_prob(u_noise).sum()
        lp = lp + dists["temp"].log_prob(u_temp).sum()

        ent = dists["k"].entropy().sum()
        ent = ent + dists["replay"].entropy().sum()
        ent = ent + dists["noise"].entropy().sum()
        ent = ent + dists["temp"].entropy().sum()

        logps.append(lp)
        ents.append(ent)
        vals.append(v.squeeze(0))

    logps = torch.stack(logps)
    ents = torch.stack(ents)
    vals = torch.stack(vals)

    with torch.no_grad():
        R = vals[-1]
        returns = []
        for t in reversed(range(len(rewards))):
            R = rewards[t] + cfg.gamma * R * (1.0 - dones[t])
            returns.append(R)
        returns = torch.stack(list(reversed(returns))).detach()

    adv = returns - vals

    policy_loss = -(logps * adv.detach()).mean()
    value_loss = F.mse_loss(vals, returns)
    entropy_loss = -ents.mean()

    loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.max_grad_norm)
    optimizer.step()

    return h.detach()
