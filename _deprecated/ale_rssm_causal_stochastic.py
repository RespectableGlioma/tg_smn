
"""
Full recurrent SSM (Dreamer-style RSSM) with explicit Causal Core vs Stochastic Envelope split.

What this implements
--------------------
Dreamer-style latent dynamics model:

  h_t     : deterministic recurrent state (GRU)
  z_t     : stochastic *causal* latent (temporal, used in transition)
  u_t     : stochastic *nuisance/appearance* latent (non-temporal, NOT used in transition)

Core transition (rule-carrying):
  (h_{t+1} = GRU(h_t, concat(z_t, a_t)))
  z_{t} ~ posterior q(z_t | h_t, e_t)
  z_{t} ~ prior     p(z_t | h_t)

Nuisance latent:
  u_t ~ posterior q(u_t | e_t)   with prior N(0,I) (i.i.d. across time)

Decoders:
  clean decoder:  o_t_clean  ≈ Dec_clean(h_t, z_t)      (should be invariant to augment nuisance)
  noisy decoder:  o_t_noisy  ≈ Dec_noisy(h_t, z_t, u_t) (should capture augmentation nuisance)

We create a *known nuisance* by applying random brightness scaling + gaussian noise augmentation
to the clean Atari-preprocessed frame. Dynamics are unchanged by this augmentation, so an ideal
representation would put nuisance in u_t and keep (h_t,z_t) invariant.

This is a world-model training harness (not the full Dreamer actor-critic). Once you like the
properties of (h,z) you can plug imagination rollouts into RL.

Dependencies
------------
pip install "gymnasium[atari]" ale-py torch opencv-python pillow tqdm

ROM note
--------
If you hit missing ROM errors, follow the ALE/Gymnasium instructions for your installed versions.

Run
---
python ale_rssm_causal_stochastic.py --env_id ALE/Pong-v5 --collect_steps 100000 --train_steps 20000

Outputs
-------
- outputs/rollout_clean_gt_vs_pred.png
- outputs/nuisance_samples.png

Tips for a compelling demo
--------------------------
1) Start deterministic: --repeat_action_probability 0.0  (no sticky actions)
2) Use a simple game: ALE/Pong-v5 or ALE/Breakout-v5
3) Verify: (h,z) invariance under augmentation AND good open-loop clean rollouts
4) Then turn on stochastic transitions: repeat_action_probability 0.25 to see what remains unpredictable
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def onehot(a: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(a, num_classes=n).float()

def augment_obs(clean01: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """
    clean01: float tensor in [0,1], shape [...,1,H,W]
    Apply random brightness + gaussian noise, clip to [0,1].
    """
    # brightness per-sample (broadcast to H,W)
    b = torch.empty((clean01.shape[0], 1, 1, 1), device=clean01.device).uniform_(0.6, 1.4)
    y = clean01 * b
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return torch.clamp(y, 0.0, 1.0)

def kl_diag_gaussian(mu_q, logstd_q, mu_p, logstd_p):
    """
    KL( N(mu_q, std_q^2) || N(mu_p, std_p^2) ), mean over batch.
    logstd_* shape [B,D]
    Returns: [B] per-sample KL (sum over dims)
    """
    std_q = torch.exp(logstd_q)
    std_p = torch.exp(logstd_p)
    var_q = std_q * std_q
    var_p = std_p * std_p
    kl = (logstd_p - logstd_q) + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p) - 0.5
    return torch.sum(kl, dim=-1)

def kl_std_normal(mu, logstd):
    # KL(q || N(0,I)) per-sample
    var = torch.exp(2.0 * logstd)
    kl = 0.5 * (var + mu**2 - 1.0 - torch.log(var + 1e-8))
    return torch.sum(kl, dim=-1)

def save_grid_png(path: Path, top: np.ndarray, bottom: np.ndarray, pad: int = 2):
    """
    Save a 2-row strip image (grayscale) comparing top vs bottom.
    top/bottom: [K,H,W] float in [0,1]
    """
    from PIL import Image
    assert top.shape == bottom.shape
    K, H, W = top.shape
    canvas = np.zeros((2*H + pad, K*W + (K-1)*pad), dtype=np.uint8)

    def u8(x): return (np.clip(x,0,1)*255).astype(np.uint8)

    for i in range(K):
        x0 = i*(W+pad)
        canvas[0:H, x0:x0+W] = u8(top[i])
        canvas[H+pad:H+pad+H, x0:x0+W] = u8(bottom[i])

    Image.fromarray(canvas, mode="L").save(str(path))


# -------------------------
# ALE env
# -------------------------
def make_atari_env(env_id: str, seed: int, frame_skip: int, repeat_action_prob: float,
                   screen_size: int, noop_max: int):
    import gymnasium as gym
    import ale_py

    gym.register_envs(ale_py)

    # Base env frameskip=1 (preprocessing wrapper will do skipping)
    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=repeat_action_prob,
    )

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=noop_max,            # random no-ops at reset for diversity (0 = none)
        frame_skip=frame_skip,        # common: 4
        screen_size=screen_size,      # Dreamer uses 64 for Atari-like settings
        grayscale_obs=True,
        grayscale_newaxis=False,      # return (H,W) uint8
        scale_obs=False,              # uint8 0..255 (we scale ourselves)
        terminal_on_life_loss=False,
    )

    env.reset(seed=seed)
    return env

def collect_dataset(env, collect_steps: int):
    """
    Collect a fixed dataset with random actions.

    Returns:
      obs  : [N+1,H,W] uint8
      act  : [N] int64
      done : [N] bool
      rew  : [N] float32
    """
    obs0, _ = env.reset()
    obs0 = np.asarray(obs0, dtype=np.uint8)
    H, W = obs0.shape
    obs = np.zeros((collect_steps + 1, H, W), dtype=np.uint8)
    act = np.zeros((collect_steps,), dtype=np.int64)
    done = np.zeros((collect_steps,), dtype=np.bool_)
    rew = np.zeros((collect_steps,), dtype=np.float32)

    obs[0] = obs0
    o = obs0
    for t in tqdm(range(collect_steps), desc="Collect"):
        a = env.action_space.sample()
        o2, r, terminated, truncated, _info = env.step(a)
        d = bool(terminated or truncated)
        o2 = np.asarray(o2, dtype=np.uint8)

        act[t] = a
        rew[t] = float(r)
        done[t] = d
        obs[t+1] = o2

        if d:
            o, _ = env.reset()
            o = np.asarray(o, dtype=np.uint8)
            obs[t+1] = o
        else:
            o = o2

    return obs, act, done, rew

def valid_starts_from_dones(done: np.ndarray, seq_len: int):
    """
    Return indices i such that done[i : i+seq_len] contains no True.
    done length = N (for actions), obs length = N+1
    """
    N = done.shape[0]
    if N < seq_len:
        return np.array([], dtype=np.int64)
    # cumulative sum trick
    d = done.astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(d)])  # length N+1
    # window sum for i..i+seq_len-1 is cs[i+seq_len] - cs[i]
    win = cs[seq_len:] - cs[:-seq_len]
    return np.where(win == 0)[0].astype(np.int64)


# -------------------------
# RSSM modules
# -------------------------
class ObsEncoder(nn.Module):
    """CNN encoder: o_t -> embedding e_t."""
    def __init__(self, in_ch: int = 1, embed_dim: int = 1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, stride=2, padding=1), nn.ReLU(),   # 64->32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),      # 32->16
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),     # 16->8
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),    # 8->4
        )
        self.fc = nn.Linear(256 * 4 * 4, embed_dim)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        h = self.conv(o).reshape(o.shape[0], -1)
        return F.relu(self.fc(h))

class GaussianHead(nn.Module):
    """Map features -> (mu, logstd) for diagonal Gaussian."""
    def __init__(self, in_dim: int, out_dim: int, min_logstd: float = -5.0, max_logstd: float = 2.0):
        super().__init__()
        self.mu = nn.Linear(in_dim, out_dim)
        self.logstd = nn.Linear(in_dim, out_dim)
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def forward(self, x):
        mu = self.mu(x)
        logstd = self.logstd(x)
        logstd = torch.clamp(logstd, self.min_logstd, self.max_logstd)
        return mu, logstd

class RSSM(nn.Module):
    """
    Dreamer-style RSSM:
      h_t: deterministic GRU state
      z_t: stochastic causal latent (used in transition)
    """
    def __init__(self, action_dim: int, h_dim: int, z_dim: int, embed_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.gru = nn.GRUCell(z_dim + action_dim, h_dim)
        self.prior_head = GaussianHead(h_dim, z_dim)
        self.post_mlp = nn.Sequential(
            nn.Linear(h_dim + embed_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )
        self.post_head = GaussianHead(512, z_dim)

    def init_state(self, batch: int, device: torch.device):
        h = torch.zeros((batch, self.h_dim), device=device)
        z = torch.zeros((batch, self.z_dim), device=device)
        return h, z

    def prior(self, h: torch.Tensor):
        return self.prior_head(h)

    def posterior(self, h: torch.Tensor, e: torch.Tensor):
        x = self.post_mlp(torch.cat([h, e], dim=-1))
        return self.post_head(x)

    def step(self, h: torch.Tensor, z: torch.Tensor, a_oh: torch.Tensor):
        """Deterministic transition update for h_{t+1}."""
        inp = torch.cat([z, a_oh], dim=-1)
        h_next = self.gru(inp, h)
        return h_next

class DecoderClean(nn.Module):
    """Decode clean frame from features [h,z]."""
    def __init__(self, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 256*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),  # 4->8
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),   # 8->16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),    # 16->32
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),    # 32->64
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(feat)).reshape(feat.shape[0], 256, 4, 4)
        return self.deconv(h)

class DecoderNoisy(nn.Module):
    """Decode noisy frame from features [h,z,u]."""
    def __init__(self, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 256*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(feat)).reshape(feat.shape[0], 256, 4, 4)
        return self.deconv(h)

class UPredictor(nn.Module):
    """Posterior for nuisance u_t from embedding e_t (i.i.d. over time)."""
    def __init__(self, embed_dim: int, u_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )
        self.head = GaussianHead(512, u_dim)

    def forward(self, e: torch.Tensor):
        x = self.mlp(e)
        return self.head(x)

def reparam(mu, logstd):
    std = torch.exp(logstd)
    eps = torch.randn_like(std)
    return mu + eps * std


# -------------------------
# Training harness
# -------------------------
@dataclass
class HParams:
    screen_size: int = 64
    h_dim: int = 200
    z_dim: int = 32
    u_dim: int = 16
    embed_dim: int = 1024
    seq_len: int = 50
    batch: int = 32

    lr: float = 2e-4
    grad_clip: float = 100.0

    # Loss weights
    beta_clean: float = 1.0
    beta_noisy: float = 0.5
    beta_kl_z: float = 1.0
    beta_kl_u: float = 0.2
    beta_inv: float = 1.0

    free_nats_z: float = 3.0
    free_nats_u: float = 1.0

    noise_std: float = 0.05


@torch.no_grad()
def eval_rollout(enc_obs, rssm, dec_clean, obs_seq_u8, act_seq, device, action_dim: int, K: int = 16):
    """
    Open-loop rollout of clean frames:
      - initialize with posterior on first frame
      - then roll prior + deterministic GRU using recorded actions
    Returns GT clean frames and predicted frames (float in [0,1]), both [K+1,H,W].
    """
    # Prepare tensors
    obs01 = torch.from_numpy(obs_seq_u8[:K+1]).to(device).float() / 255.0  # [K+1,H,W]
    obs01 = obs01.unsqueeze(1)  # [K+1,1,H,W]
    act = torch.from_numpy(act_seq[:K]).to(device).long()

    # Init h,z
    h, z = rssm.init_state(batch=1, device=device)

    # First step posterior
    o0_noisy = augment_obs(obs01[0:1], noise_std=0.0)  # just to match interface; no noise for eval init
    e0 = enc_obs(o0_noisy)
    mu_p, logstd_p = rssm.prior(h)
    mu_q, logstd_q = rssm.posterior(h, e0)
    z = mu_q  # deterministic eval

    preds = []
    # decode t=0
    feat = torch.cat([h, z], dim=-1)
    preds.append(dec_clean(feat)[0,0].detach().cpu().numpy())

    for t in range(K):
        a_oh = onehot(act[t:t+1], action_dim)
        h = rssm.step(h, z, a_oh)
        # prior for next z
        mu_p, logstd_p = rssm.prior(h)
        z = mu_p  # deterministic
        feat = torch.cat([h, z], dim=-1)
        preds.append(dec_clean(feat)[0,0].detach().cpu().numpy())

    gt = obs01[:,0].detach().cpu().numpy()
    pr = np.stack(preds, axis=0)
    return gt, pr

@torch.no_grad()
def eval_invariance(enc_obs, rssm, obs_frame_u8, device, n_augs: int = 16):
    """
    Measure how much posterior mean z changes under different augmentations.
    Uses h=0 for a single frame.
    """
    o = torch.from_numpy(obs_frame_u8).to(device).float().unsqueeze(0).unsqueeze(0) / 255.0  # [1,1,H,W]
    h0, _ = rssm.init_state(batch=1, device=device)

    mus = []
    for _ in range(n_augs):
        o_noisy = augment_obs(o, noise_std=0.05)
        e = enc_obs(o_noisy)
        mu_q, _ = rssm.posterior(h0, e)
        mus.append(mu_q[0].detach().cpu().numpy())
    mus = np.stack(mus, axis=0)  # [n,D]
    # average pairwise squared distance
    diffs = mus[:, None, :] - mus[None, :, :]
    return float(np.mean(np.sum(diffs*diffs, axis=-1)))

@torch.no_grad()
def save_nuisance_samples(out_path: Path, enc_obs, rssm, u_pred, dec_noisy,
                          obs_frame_u8, device, action_dim: int, num_samples: int = 10):
    """
    Fix (h,z) from posterior on the given frame, then sample multiple u ~ N(0,I),
    decode noisy frames. Save as 1-row strip (top row only).
    """
    from PIL import Image
    o = torch.from_numpy(obs_frame_u8).to(device).float().unsqueeze(0).unsqueeze(0) / 255.0  # [1,1,H,W]
    h0, _ = rssm.init_state(batch=1, device=device)

    # posterior z from one noisy view
    o_noisy = augment_obs(o, noise_std=0.05)
    e = enc_obs(o_noisy)
    mu_q, logstd_q = rssm.posterior(h0, e)
    z = mu_q
    h = h0

    # Make samples of u
    H, W = obs_frame_u8.shape
    pad = 2
    canvas = np.zeros((H, num_samples*W + (num_samples-1)*pad), dtype=np.uint8)

    for i in range(num_samples):
        u = torch.randn((1, u_pred.head.mu.out_features), device=device)
        feat = torch.cat([h, z, u], dim=-1)
        img = dec_noisy(feat)[0,0].detach().cpu().numpy()
        canvas[:, i*(W+pad):i*(W+pad)+W] = (np.clip(img,0,1)*255).astype(np.uint8)

    Image.fromarray(canvas, mode="L").save(str(out_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--collect_steps", type=int, default=100_000)
    p.add_argument("--frame_skip", type=int, default=4)
    p.add_argument("--repeat_action_probability", type=float, default=0.0)
    p.add_argument("--screen_size", type=int, default=64)
    p.add_argument("--noop_max", type=int, default=0)

    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--train_steps", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", type=str, default="outputs_rssm_causal_stochastic")
    p.add_argument("--eval_every", type=int, default=2000)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = make_atari_env(
        env_id=args.env_id,
        seed=args.seed,
        frame_skip=args.frame_skip,
        repeat_action_prob=args.repeat_action_probability,
        screen_size=args.screen_size,
        noop_max=args.noop_max,
    )
    action_dim = env.action_space.n
    print("Env:", args.env_id, "| action_dim:", action_dim, "| obs:", env.observation_space)

    obs_u8, act, done, rew = collect_dataset(env, args.collect_steps)
    starts = valid_starts_from_dones(done, args.seq_len)
    if len(starts) == 0:
        raise RuntimeError("No valid sequences found (seq_len too large for collected episodes).")

    # Hyperparams
    hp = HParams(
        screen_size=args.screen_size,
        seq_len=args.seq_len,
        batch=args.batch,
        lr=args.lr,
    )

    # Build model
    enc_obs = ObsEncoder(in_ch=1, embed_dim=hp.embed_dim).to(device)
    rssm = RSSM(action_dim=action_dim, h_dim=hp.h_dim, z_dim=hp.z_dim, embed_dim=hp.embed_dim).to(device)
    u_pred = UPredictor(embed_dim=hp.embed_dim, u_dim=hp.u_dim).to(device)
    dec_clean = DecoderClean(feat_dim=hp.h_dim + hp.z_dim).to(device)
    dec_noisy = DecoderNoisy(feat_dim=hp.h_dim + hp.z_dim + hp.u_dim).to(device)

    opt = torch.optim.Adam(
        list(enc_obs.parameters()) + list(rssm.parameters()) + list(u_pred.parameters()) +
        list(dec_clean.parameters()) + list(dec_noisy.parameters()),
        lr=hp.lr
    )

    H, W = obs_u8.shape[1], obs_u8.shape[2]

    def sample_batch():
        idx = np.random.choice(starts, size=hp.batch, replace=True)
        # obs window length seq_len+1
        obs_seq = np.stack([obs_u8[i:i+hp.seq_len+1] for i in idx], axis=0)  # [B,L+1,H,W]
        act_seq = np.stack([act[i:i+hp.seq_len] for i in idx], axis=0)       # [B,L]
        return obs_seq, act_seq

    # Pick a fixed eval sequence from held-out region
    eval_i = int(starts[len(starts)//2])
    eval_obs_seq = obs_u8[eval_i:eval_i+hp.seq_len+1]
    eval_act_seq = act[eval_i:eval_i+hp.seq_len]

    # Training loop
    enc_obs.train(); rssm.train(); u_pred.train(); dec_clean.train(); dec_noisy.train()

    for step in range(1, args.train_steps + 1):
        obs_seq_u8, act_seq_np = sample_batch()
        obs_seq = torch.from_numpy(obs_seq_u8).to(device).float() / 255.0  # [B,L+1,H,W]
        obs_seq = obs_seq.unsqueeze(2)  # [B,L+1,1,H,W]
        act_seq = torch.from_numpy(act_seq_np).to(device).long()           # [B,L]

        B, Lp1 = obs_seq.shape[0], obs_seq.shape[1]
        L = Lp1 - 1

        # Create noisy augmented version
        obs_flat = obs_seq.reshape(B*(L+1), 1, H, W)
        obs_noisy_flat = augment_obs(obs_flat, noise_std=hp.noise_std)
        obs_noisy = obs_noisy_flat.reshape(B, L+1, 1, H, W)

        # Encode embeddings for noisy views (posterior uses noisy)
        e = enc_obs(obs_noisy_flat).reshape(B, L+1, -1)  # [B,L+1,E]

        # Unroll RSSM
        h, z = rssm.init_state(B, device=device)

        # Accumulate losses across time
        recon_clean = 0.0
        recon_noisy = 0.0
        kl_z = 0.0
        kl_u = 0.0

        # Invariance: compare posterior mean z under two different augmentations at t=0 (h=0)
        # (cheap and usually sufficient to enforce nuisance invariance)
        obs0 = obs_seq[:, 0]  # [B,1,H,W]
        e1 = enc_obs(augment_obs(obs0, noise_std=hp.noise_std))
        e2 = enc_obs(augment_obs(obs0, noise_std=hp.noise_std))
        mu1, _ = rssm.posterior(h, e1)
        mu2, _ = rssm.posterior(h, e2)
        inv_loss = F.mse_loss(mu1, mu2)

        for t in range(L+1):
            # prior and posterior for z_t
            mu_p, logstd_p = rssm.prior(h)
            mu_q, logstd_q = rssm.posterior(h, e[:, t])

            z = reparam(mu_q, logstd_q)

            # nuisance u_t
            mu_u, logstd_u = u_pred(e[:, t])
            u = reparam(mu_u, logstd_u)

            # decode
            feat_clean = torch.cat([h, z], dim=-1)
            pred_clean = dec_clean(feat_clean)                 # [B,1,H,W]
            tgt_clean = obs_seq[:, t]                          # [B,1,H,W]
            recon_clean = recon_clean + F.mse_loss(pred_clean, tgt_clean)

            feat_noisy = torch.cat([h, z, u], dim=-1)
            pred_noisy = dec_noisy(feat_noisy)
            tgt_noisy = obs_noisy[:, t]
            recon_noisy = recon_noisy + F.mse_loss(pred_noisy, tgt_noisy)

            # KL terms
            klz_t = kl_diag_gaussian(mu_q, logstd_q, mu_p, logstd_p)  # [B]
            klu_t = kl_std_normal(mu_u, logstd_u)                     # [B]

            # free nats (clamp per-sample)
            klz_t = torch.clamp(klz_t, min=hp.free_nats_z)
            klu_t = torch.clamp(klu_t, min=hp.free_nats_u)

            kl_z = kl_z + torch.mean(klz_t)
            kl_u = kl_u + torch.mean(klu_t)

            # transition update (skip after last)
            if t < L:
                a_oh = onehot(act_seq[:, t], action_dim)
                h = rssm.step(h, z, a_oh)

        # average across time steps
        recon_clean = recon_clean / (L+1)
        recon_noisy = recon_noisy / (L+1)
        kl_z = kl_z / (L+1)
        kl_u = kl_u / (L+1)

        loss = (
            hp.beta_clean * recon_clean +
            hp.beta_noisy * recon_noisy +
            hp.beta_kl_z * kl_z +
            hp.beta_kl_u * kl_u +
            hp.beta_inv * inv_loss
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(enc_obs.parameters()) + list(rssm.parameters()) + list(u_pred.parameters()) +
            list(dec_clean.parameters()) + list(dec_noisy.parameters()),
            hp.grad_clip
        )
        opt.step()

        if step % 200 == 0:
            print(f"step {step:06d} | loss {loss.item():.4f} | recon_clean {recon_clean.item():.4f} | recon_noisy {recon_noisy.item():.4f} | kl_z {kl_z.item():.3f} | kl_u {kl_u.item():.3f} | inv {inv_loss.item():.4f}")

        if step % args.eval_every == 0:
            enc_obs.eval(); rssm.eval(); dec_clean.eval(); u_pred.eval(); dec_noisy.eval()
            # Rollout viz
            gt, pr = eval_rollout(enc_obs, rssm, dec_clean, eval_obs_seq, eval_act_seq, device, action_dim, K=16)
            save_grid_png(outdir / f"rollout_clean_gt_vs_pred_step{step}.png", gt, pr)
            # Invariance metric
            inv = eval_invariance(enc_obs, rssm, eval_obs_seq[0], device, n_augs=16)
            print(f"[EVAL step {step}] inv_pairwise_z_mean_sqdist = {inv:.4f} | saved rollout png")
            # Nuisance samples
            save_nuisance_samples(outdir / f"nuisance_samples_step{step}.png", enc_obs, rssm, u_pred, dec_noisy,
                                  eval_obs_seq[0], device, action_dim, num_samples=10)
            enc_obs.train(); rssm.train(); dec_clean.train(); u_pred.train(); dec_noisy.train()

    print("Done. Outputs in:", outdir)


if __name__ == "__main__":
    main()
