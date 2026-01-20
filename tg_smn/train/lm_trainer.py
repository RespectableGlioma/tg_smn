from __future__ import annotations

import math
import os
import random
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..config import DataCfg, FixedCtrlCfg, LearnedCtrlCfgLM, ModelCfgLM, TrainCfgLM
from ..utils import count_params, ensure_dir, save_json, set_seed
from ..envs.base import EnvData, make_task_loaders
from ..models.transformer_lm import TransformerLM, lm_loss
from .replay import ReplayBufferLM
from ..controllers.fixed import FixedController
from ..controllers.learned import (
    GRUController,
    ObsNorm,
    Rollout,
    build_obs,
    controller_update,
    select_action,
)


@contextmanager
def temporarily_eval(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def deterministic_logits(model: TransformerLM, xb: torch.Tensor) -> torch.Tensor:
    with temporarily_eval(model):
        logits, _ = model(xb, stochastic=False)
    return logits


@torch.no_grad()
def kl_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits_p, dim=-1)
    logq = F.log_softmax(logits_q, dim=-1)
    p = logp.exp()
    kl = (p * (logp - logq)).sum(dim=-1)
    return kl.mean()


@torch.no_grad()
def eval_ppl(model: TransformerLM, loader) -> float:
    with temporarily_eval(model):
        total_loss = 0.0
        total_tokens = 0
        for xb, yb in loader:
            xb, yb = xb.to(next(model.parameters()).device), yb.to(next(model.parameters()).device)
            logits, _ = model(xb, stochastic=False)
            loss = lm_loss(logits, yb)
            total_loss += loss.item() * xb.numel()
            total_tokens += xb.numel()
        mean_loss = total_loss / max(1, total_tokens)
        return float(math.exp(mean_loss))


@torch.no_grad()
def loss_stochastic_std_lm(model: TransformerLM, xb: torch.Tensor, yb: torch.Tensor, n_samples: int = 3) -> float:
    losses = []
    for _ in range(n_samples):
        logits, _ = model(xb, stochastic=True)
        losses.append(lm_loss(logits, yb).detach())
    losses = torch.stack(losses)
    return float(losses.std().item())


def _summary_from_eval(eval_rows: List[Dict[str, Any]], n_tasks: int) -> Dict[str, float]:
    # eval_rows contains entries with keys: task_trained, task_eval, ppl
    # best ppl per task (over time)
    best = [float("inf")] * n_tasks
    for r in eval_rows:
        t = int(r["task_eval"])
        best[t] = min(best[t], float(r["ppl"]))

    # final ppl per task (after last task)
    final_rows = [r for r in eval_rows if int(r["task_trained"]) == (n_tasks - 1)]
    final_ppl = [float("nan")] * n_tasks
    for r in final_rows:
        final_ppl[int(r["task_eval"])] = float(r["ppl"])

    forgetting = [final_ppl[t] - best[t] for t in range(n_tasks)]
    return {
        "avg_final_task_ppl": float(np.nanmean(final_ppl)),
        "avg_forgetting_ppl": float(np.nanmean(forgetting)),
    }


def run_lm_experiment(
    *,
    exp_name: str,
    variant: str,
    env: EnvData,
    data_cfg: DataCfg,
    model_cfg: ModelCfgLM,
    train_cfg: TrainCfgLM,
    out_dir: str,
    device: str,
    fixed_ctrl_cfg: Optional[FixedCtrlCfg] = None,
    learned_ctrl_cfg: Optional[LearnedCtrlCfgLM] = None,
    learned_deterministic_policy: bool = False,
    learned_fixed_k: Optional[int] = None,
    learned_fixed_replay: Optional[float] = None,
    learned_drop_obs_kl: bool = False,
    learned_drop_reward_kl: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate a model on a continual LM environment.

    Variants:
      - "dense_baseline"
      - "sparse_fixed"
      - "tg_smn_learned"

    Results are written to out_dir/exp_name.
    """

    set_seed(train_cfg.seed)
    exp_dir = ensure_dir(os.path.join(out_dir, exp_name))

    # Long sweeps on Colab (especially writing to Drive) can occasionally hit
    # transient filesystem issues. Use best-effort I/O so training doesn't
    # crash due to a missed directory creation or a temporary mount hiccup.
    def _safe_makedirs(path: str) -> None:
        try:
            ensure_dir(path)
        except Exception:
            # If ensure_dir fails, we don't want to crash training; later writes
            # will retry.
            pass

    def _safe_torch_save(obj: Any, path: str) -> None:
        parent = os.path.dirname(path) or exp_dir
        for _ in range(2):
            try:
                _safe_makedirs(parent)
                torch.save(obj, path)
                return
            except Exception:
                continue
        # If saving still fails, proceed without checkpointing.

    def _safe_to_csv(df: pd.DataFrame, path: str) -> None:
        parent = os.path.dirname(path) or exp_dir
        for _ in range(2):
            try:
                _safe_makedirs(parent)
                df.to_csv(path, index=False)
                return
            except Exception:
                continue

    def _safe_save_json(path: str, obj: Any) -> None:
        parent = os.path.dirname(path) or exp_dir
        for _ in range(2):
            try:
                _safe_makedirs(parent)
                save_json(path, obj)
                return
            except Exception:
                continue

    # Build loaders
    train_loaders, val_loaders, test_loaders = make_task_loaders(
        env, seq_len=data_cfg.seq_len, batch_size=data_cfg.batch_size, num_workers=data_cfg.num_workers
    )

    n_tasks = len(train_loaders)

    # Model
    if variant == "dense_baseline":
        sparse = False
    else:
        sparse = True

    model = TransformerLM(model_cfg, vocab_size=env.vocab_size, sparse=sparse).to(device)

    # Opt
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # Replay buffer
    buffer = ReplayBufferLM(max_seqs=train_cfg.replay_max_seqs)

    # Controllers
    fixed_ctrl = None
    ctrl = None
    ctrl_opt = None
    obs_norm = None
    rollout = None
    h_ctrl = None
    h0_rollout = None

    if variant == "sparse_fixed":
        fixed_ctrl = FixedController(fixed_ctrl_cfg or FixedCtrlCfg())
        a = fixed_ctrl.act()
        model.set_routing(k=a["k"], router_temp=a["temp"], router_noise=a["noise"])

    if variant == "tg_smn_learned":
        if learned_ctrl_cfg is None:
            learned_ctrl_cfg = LearnedCtrlCfgLM(k_max=model_cfg.max_k, k_min=1)
        ctrl = GRUController(learned_ctrl_cfg, obs_dim=8, device=device).to(device)
        ctrl_opt = torch.optim.Adam(ctrl.parameters(), lr=learned_ctrl_cfg.lr)
        obs_norm = ObsNorm(dim=8, device=device)
        rollout = Rollout()
        h_ctrl = ctrl.init_state(batch_size=1).detach()
        h0_rollout = h_ctrl.detach()
        model.set_routing(k=model_cfg.max_k, router_temp=1.0, router_noise=0.0)

    # Save config
    cfg_dump = {
        "exp_name": exp_name,
        "variant": variant,
        "env": {"name": env.name, "meta": env.meta},
        "model_cfg": asdict(model_cfg),
        "data_cfg": asdict(data_cfg),
        "train_cfg": asdict(train_cfg),
        "fixed_ctrl_cfg": asdict(fixed_ctrl_cfg) if fixed_ctrl_cfg is not None else None,
        "learned_ctrl_cfg": asdict(learned_ctrl_cfg) if learned_ctrl_cfg is not None else None,
        "params": count_params(model),
        "device": device,
        "learned_ablations": {
            "fixed_k": learned_fixed_k,
            "fixed_replay": learned_fixed_replay,
            "drop_obs_kl": learned_drop_obs_kl,
            "drop_reward_kl": learned_drop_reward_kl,
        },
    }
    _safe_save_json(os.path.join(exp_dir, "config.json"), cfg_dump)

    metrics_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []

    # Book-keeping for forgetting
    best_ppl = [float("inf")] * n_tasks

    global_step = 0
    ema_loss: Optional[float] = None
    last_sqrt2kl: float = 1e-3

    # Learned controller uses prev-step obs
    prev_loss = 5.0
    prev_ema = 5.0
    prev_delta_t = 0.0
    prev_delta_rho = 1.0
    prev_eta = 0.0
    prev_k = float(model_cfg.max_k)
    prev_replay = 0.10

    # Progress bars: counts + elapsed only (no ETA).
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {rate_fmt}"

    task_iter = range(n_tasks)
    if train_cfg.show_progress:
        desc = f"{env.name}:{variant} E{model_cfg.n_experts} seed{train_cfg.seed}"
        task_iter = tqdm(task_iter, total=n_tasks, desc=desc, bar_format=bar_format)

    for task_id in task_iter:
        for epoch in range(train_cfg.epochs_per_task):
            step_loader = train_loaders[task_id]
            if train_cfg.show_progress and train_cfg.progress_steps:
                try:
                    total_steps = len(step_loader)
                except Exception:
                    total_steps = None
                if train_cfg.max_steps_per_task is not None and total_steps is not None:
                    total_steps = min(total_steps, int(train_cfg.max_steps_per_task) + 1)
                step_loader = tqdm(
                    step_loader,
                    total=total_steps,
                    desc=f"task{task_id} ep{epoch}",
                    leave=False,
                    bar_format=bar_format,
                )

            for step_in_task, (xb, yb) in enumerate(step_loader):
                xb, yb = xb.to(device), yb.to(device)

                # ----- choose action / routing -----
                if variant == "dense_baseline":
                    action = {"k": -1, "replay_ratio": 0.0, "noise": 0.0, "temp": 1.0}
                    raw_action = None

                elif variant == "sparse_fixed":
                    action = fixed_ctrl.act()  # type: ignore
                    raw_action = None
                    model.set_routing(k=action["k"], router_temp=action["temp"], router_noise=action["noise"])

                elif variant == "tg_smn_learned":
                    assert ctrl is not None and obs_norm is not None and rollout is not None and learned_ctrl_cfg is not None
                    obs = build_obs(
                        prev_loss,
                        prev_ema,
                        prev_delta_t,
                        prev_delta_rho,
                        1.0 if learned_drop_obs_kl else last_sqrt2kl,
                        prev_eta,
                        prev_k,
                        prev_replay,
                        device=device,
                    )
                    obs_norm.update(obs)
                    obs_n = obs_norm.normalize(obs)

                    with torch.no_grad():
                        dists, v, h_ctrl = ctrl.step(obs_n, h_ctrl)  # type: ignore
                        a_dict, raw_action, lp, ent = select_action(learned_ctrl_cfg, dists, deterministic=learned_deterministic_policy)

                    # Apply ablations
                    if learned_fixed_k is not None:
                        a_dict["k"] = int(learned_fixed_k)
                    if learned_fixed_replay is not None:
                        a_dict["replay_ratio"] = float(learned_fixed_replay)

                    action = a_dict
                    model.set_routing(k=action["k"], router_temp=action["temp"], router_noise=action["noise"])

                else:
                    raise ValueError(f"Unknown variant: {variant}")

                replay_ratio = float(action.get("replay_ratio", 0.0))

                # ----- choose online vs replay batch -----
                use_replay = (len(buffer) >= data_cfg.batch_size) and (random.random() < replay_ratio)
                if use_replay:
                    xb_train, yb_train = buffer.sample(data_cfg.batch_size, device)
                else:
                    xb_train, yb_train = xb, yb

                # Δρ L via stochastic forward samples (no grad)
                with torch.no_grad():
                    delta_rho_L = loss_stochastic_std_lm(model, xb_train, yb_train, n_samples=train_cfg.delta_rho_samples)

                # KL probe pre-step (deterministic) occasionally
                do_fisher = (global_step % train_cfg.fisher_every == 0)
                logits_pre = None
                if do_fisher:
                    logits_pre = deterministic_logits(model, xb_train)

                # ----- SGD step -----
                opt.zero_grad(set_to_none=True)
                logits, aux_all = model(xb_train, stochastic=True)
                loss = lm_loss(logits, yb_train)
                loss.backward()
                if train_cfg.grad_clip is not None and train_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                opt.step()

                # post-step KL
                if do_fisher:
                    assert logits_pre is not None
                    logits_post = deterministic_logits(model, xb_train)
                    kl = float(kl_logits(logits_pre, logits_post).item())
                    sqrt_2kl = math.sqrt(max(2.0 * kl, 1e-12))
                    last_sqrt2kl = float(sqrt_2kl)
                else:
                    kl = float("nan")
                    sqrt_2kl = float("nan")

                # EMA loss + Δt
                if ema_loss is None:
                    ema_loss = float(loss.item())
                ema_prev = float(ema_loss)
                ema_loss = 0.98 * ema_loss + 0.02 * float(loss.item())

                # Per-step improvement proxy
                delta_t = ema_prev - ema_loss

                denom_kl = 1.0 if learned_drop_reward_kl else last_sqrt2kl
                denom = (delta_rho_L * denom_kl) + 1e-8
                delta_pos = max(delta_t, 0.0)
                eta_step = (delta_pos / denom) if denom > 0 else 0.0

                # Reward for learned controller
                reward = float("nan")
                if variant == "tg_smn_learned":
                    # prefer efficient learning with low compute and low replay
                    reward = eta_step / (float(action["k"]) * (1.0 + float(replay_ratio)) + 1e-8)

                    # Store rollout transition
                    rollout.obs.append(obs.detach())
                    rollout.actions.append(raw_action)  # type: ignore
                    rollout.rewards.append(float(reward))
                    rollout.dones.append(0.0)

                    # Update controller every rollout window
                    if (global_step > 0) and (global_step % learned_ctrl_cfg.rollout_len == 0):
                        h_ctrl = controller_update(ctrl, ctrl_opt, rollout, h0_rollout, obs_norm, learned_ctrl_cfg)  # type: ignore
                        h0_rollout = h_ctrl.detach()
                        rollout.clear()

                # Logging
                if global_step % train_cfg.log_every == 0:
                    metrics_rows.append(
                        {
                            "step": global_step,
                            "task": task_id,
                            "loss": float(loss.item()),
                            "ema_loss": float(ema_loss),
                            "delta_t": float(delta_t),
                            "delta_rho_L": float(delta_rho_L),
                            "kl": float(kl),
                            "sqrt_2kl": float(sqrt_2kl),
                            "eta": float(eta_step),
                            "reward": float(reward),
                            "replay_ratio": float(replay_ratio),
                            "used_replay": int(use_replay),
                            "k": int(action.get("k", -1)),
                            "router_temp": float(action.get("temp", 1.0)),
                            "router_noise": float(action.get("noise", 0.0)),
                        }
                    )

                # Optional live postfix updates for the step progress bar (no ETA)
                if (
                    train_cfg.show_progress
                    and train_cfg.progress_steps
                    and hasattr(step_loader, "set_postfix")
                    and (step_in_task % max(1, int(train_cfg.progress_postfix_every))) == 0
                ):
                    try:
                        step_loader.set_postfix(
                            {
                                "loss": f"{float(loss.item()):.3f}",
                                "k": int(action.get("k", -1)),
                                "replay": f"{float(replay_ratio):.2f}",
                            }
                        )
                    except Exception:
                        pass

                # Add online batch to buffer
                buffer.add_batch(xb, yb)

                # Update prev metrics for next observation (learned controller)
                prev_loss = float(loss.item())
                prev_ema = float(ema_loss)
                prev_delta_t = float(delta_t)
                prev_delta_rho = float(delta_rho_L)
                prev_eta = float(eta_step)
                prev_k = float(action.get("k", model_cfg.max_k))
                prev_replay = float(replay_ratio)

                global_step += 1

                if train_cfg.max_steps_per_task is not None and step_in_task >= train_cfg.max_steps_per_task:
                    break

        # End of task evaluation on val
        for j in range(task_id + 1):
            ppl = eval_ppl(model, val_loaders[j])
            best_ppl[j] = min(best_ppl[j], ppl)
            eval_rows.append({"task_trained": task_id, "task_eval": j, "ppl": ppl})

        # checkpoint
        _safe_torch_save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "task": task_id},
            os.path.join(exp_dir, f"ckpt_task{task_id}.pt"),
        )

    # Final test evaluation (mean over tasks)
    final_test_ppls = [eval_ppl(model, test_loaders[t]) for t in range(n_tasks)]
    final_test_ppl = float(np.mean(final_test_ppls))

    summ = _summary_from_eval(eval_rows, n_tasks)
    summary = {
        "exp_name": exp_name,
        "avg_final_task_ppl": summ["avg_final_task_ppl"],
        "avg_forgetting_ppl": summ["avg_forgetting_ppl"],
        "final_val_ppl": float("nan"),
        "final_test_ppl": final_test_ppl,
        "params": count_params(model),
        "n_experts": model_cfg.n_experts,
    }

    # Write logs
    _safe_to_csv(pd.DataFrame(metrics_rows), os.path.join(exp_dir, "metrics.csv"))
    _safe_to_csv(pd.DataFrame(eval_rows), os.path.join(exp_dir, "eval.csv"))
    _safe_save_json(os.path.join(exp_dir, "summary.json"), summary)

    return summary
