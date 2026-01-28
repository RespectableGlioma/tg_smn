from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .games import Othello, Game2048
from .games.common import augment_obs_u8
from .model import StochMuZeroNet, AuxSpec, masked_log_softmax
from .replay import ReplayBuffer, Episode


# -------------------------
# Helpers: aux <-> state
# -------------------------
def _othello_state_from_aux(aux: Dict[str, np.ndarray]) -> Any:
    from .games.othello import OthelloState
    board_cls = aux["board"]
    player_cls = int(aux["player"]) if np.ndim(aux["player"]) == 0 else int(aux["player"].item())
    board = np.zeros((8, 8), dtype=np.int8)
    board[board_cls == 1] = 1
    board[board_cls == 2] = -1
    player = 1 if player_cls == 0 else -1
    return OthelloState(board=board, player=player)


def _game2048_state_from_aux(aux: Dict[str, np.ndarray]) -> Any:
    from .games.game2048 import Game2048State
    gexp = aux["grid"]
    grid = np.zeros((4, 4), dtype=np.int32)
    nz = gexp > 0
    if np.any(nz):
        grid[nz] = (2 ** gexp[nz]).astype(np.int32)
    return Game2048State(grid=grid)


def _save_compare_strip_png(path: Path, top_imgs: np.ndarray, bot_imgs: np.ndarray, pad: int = 2):
    """Save a 2-row strip image for quick qualitative inspection.
    top_imgs/bot_imgs: [K,H,W] uint8
    """
    from PIL import Image
    assert top_imgs.shape == bot_imgs.shape
    K, H, W = top_imgs.shape
    canvas = np.zeros((2 * H + pad, K * W + (K - 1) * pad), dtype=np.uint8)
    for i in range(K):
        x0 = i * (W + pad)
        canvas[0:H, x0:x0+W] = top_imgs[i]
        canvas[H+pad:H+pad+H, x0:x0+W] = bot_imgs[i]
    Image.fromarray(canvas, mode="L").save(str(path))


# -------------------------
# Data collection
# -------------------------
def collect_random_episodes(game, num_episodes: int, max_steps: int, rng: np.random.RandomState) -> ReplayBuffer:
    buf = ReplayBuffer(capacity_episodes=max(100, num_episodes * 2))
    H, W = game.obs_shape
    A = game.action_size

    for _ in tqdm(range(num_episodes), desc=f"Collect({game.name})"):
        state = game.reset(rng)

        obs_list = []
        style_list = []
        legal_list = []
        aux_lists: Dict[str, list] = {k: [] for k in game.encode_aux(state).keys()}

        act_list = []
        chance_list = []
        rew_list = []
        done_list = []
        after_aux_lists: Dict[str, list] = {k: [] for k in game.encode_aux(state).keys()}

        policy_list = []

        # initial obs
        style = int(rng.randint(0, game.num_styles))
        obs = game.render(state, style)
        aux0 = game.encode_aux(state)
        legal0 = game.legal_actions(state)

        obs_list.append(obs)
        style_list.append(style)
        legal_list.append(legal0)
        for k, v in aux0.items():
            aux_lists[k].append(v)

        for t in range(max_steps):
            legal = legal0
            legal_ids = np.where(legal)[0]
            if len(legal_ids) == 0:
                break
            a = int(rng.choice(legal_ids))
            pi = np.zeros((A,), dtype=np.float32)
            pi[a] = 1.0
            policy_list.append(pi)

            afterstate, r, valid, info = game.apply_action(state, a)
            # random policy should pick legal; still guard
            if not valid:
                # treat as pass / no-op (should be rare)
                afterstate = state
                r = 0.0
                info = {"changed": False}

            after_aux = game.encode_aux(afterstate)
            for k, v in after_aux.items():
                after_aux_lists[k].append(v)

            c = game.sample_chance(afterstate, info, rng)
            next_state = game.apply_chance(afterstate, c, info)
            done = bool(game.is_terminal(next_state))

            act_list.append(a)
            chance_list.append(int(c))
            rew_list.append(float(r))
            done_list.append(done)

            # next obs
            style = int(rng.randint(0, game.num_styles))
            obs = game.render(next_state, style)
            aux1 = game.encode_aux(next_state)
            legal1 = game.legal_actions(next_state)

            obs_list.append(obs)
            style_list.append(style)
            legal_list.append(legal1)
            for k, v in aux1.items():
                aux_lists[k].append(v)

            state = next_state
            aux0 = aux1
            legal0 = legal1

            if done:
                break

        # finalize episode arrays
        T = len(act_list)
        if T == 0:
            continue

        obs_u8 = np.stack(obs_list, axis=0).astype(np.uint8)
        style_arr = np.array(style_list, dtype=np.int64)
        legal_arr = np.stack(legal_list, axis=0).astype(np.bool_)

        aux_arr = {k: np.stack(v, axis=0) for k, v in aux_lists.items()}
        after_aux_arr = {k: np.stack(v, axis=0) for k, v in after_aux_lists.items()}

        actions = np.array(act_list, dtype=np.int64)
        chances = np.array(chance_list, dtype=np.int64)
        rewards = np.array(rew_list, dtype=np.float32)
        done = np.array(done_list, dtype=np.bool_)

        policy = np.stack(policy_list, axis=0).astype(np.float32)
        # policy targets are needed at T+1 positions; append a dummy uniform target at last state
        last_pi = legal_arr[-1].astype(np.float32)
        if last_pi.sum() > 0:
            last_pi = last_pi / last_pi.sum()
        else:
            last_pi = np.ones((A,), dtype=np.float32) / A
        policy = np.concatenate([policy, last_pi[None, :]], axis=0)

        # value targets
        if game.name == "othello":
            # compute winner from final board (black perspective)
            final_board = aux_arr["board"][T]
            black = int(np.sum(final_board == 1))
            white = int(np.sum(final_board == 2))
            if black > white:
                w = 1
            elif white > black:
                w = -1
            else:
                w = 0
            # value from perspective of player-to-move at each time
            p = aux_arr["player"].astype(np.int64)  # 0 black to move, 1 white to move
            # if player==0 => sign +1 else -1
            sign = np.where(p == 0, 1.0, -1.0).astype(np.float32)
            value = (float(w) * sign).astype(np.float32)
        else:
            # 2048: return = sum of future rewards (gamma=1)
            value = np.zeros((T + 1,), dtype=np.float32)
            G = 0.0
            for i in range(T - 1, -1, -1):
                G = float(rewards[i] + G)
                value[i] = G
            value[T] = 0.0

        ep = Episode(
            obs_u8=obs_u8,
            style=style_arr,
            aux=aux_arr,
            actions=actions,
            chances=chances,
            rewards=rewards,
            done=done,
            policy=policy,
            value=value,
            legal=legal_arr,
            after_aux=after_aux_arr,
        )
        buf.add(ep)

    return buf


# -------------------------
# Training utilities
# -------------------------
def aux_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # logits: [B,C,*S], target: [B,*S]
    if logits.dim() == 2:
        return F.cross_entropy(logits, target, reduction="mean")
    B, C = logits.shape[0], logits.shape[1]
    # flatten spatial
    logits_f = logits.view(B, C, -1)
    target_f = target.view(B, -1)
    # cross_entropy expects [B,C,*] with target [B,*]
    return F.cross_entropy(logits_f, target_f, reduction="mean")


def after_aux_loss_2048_weighted(
    logits: torch.Tensor,
    target: torch.Tensor,
    cur_grid: torch.Tensor,
    empty_weight: float = 0.2,
    changed_alpha: float = 3.0,
) -> torch.Tensor:
    """
    Weighted afterstate aux loss for 2048.

    Args:
        logits: [B, 16, 4, 4] - predicted logits (16 classes for tile exponents)
        target: [B, 4, 4] - ground truth afterstate grid
        cur_grid: [B, 4, 4] - current grid before action (to identify changed cells)
        empty_weight: weight for empty cells (class 0)
        changed_alpha: multiplicative boost for cells that changed due to the move

    Returns:
        Weighted cross-entropy loss
    """
    B, C, H, W = logits.shape
    assert C == 16, "Expected 16 classes for 2048 tile exponents"

    # Compute per-cell cross-entropy without reduction
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    target_flat = target.reshape(-1)  # [B*H*W]
    loss_per_cell = F.cross_entropy(logits_flat, target_flat, reduction="none")  # [B*H*W]
    loss_per_cell = loss_per_cell.view(B, H, W)  # [B, H, W]

    # Create weight mask
    weights = torch.ones_like(loss_per_cell)

    # Downweight empty cells
    empty_mask = (target == 0)
    weights = torch.where(empty_mask, torch.tensor(empty_weight, device=logits.device), weights)

    # Upweight changed cells
    changed_mask = (cur_grid != target)
    weights = torch.where(changed_mask, weights * changed_alpha, weights)

    # Apply weights and compute mean
    weighted_loss = (loss_per_cell * weights).sum() / weights.sum()

    return weighted_loss


def policy_loss_soft(logits: torch.Tensor, target_probs: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    # logits: [B,A], target_probs: [B,A], legal_mask: [B,A]
    logp = masked_log_softmax(logits, legal_mask, dim=-1)
    return -(target_probs * logp).sum(dim=-1).mean()


def chance_mask_2048(cur_grid: torch.Tensor, after_grid: torch.Tensor, chance_size: int = 33) -> torch.Tensor:
    # cur_grid, after_grid: [B,4,4] int64 exponent classes
    B = cur_grid.shape[0]
    device = cur_grid.device
    mask = torch.zeros((B, chance_size), dtype=torch.bool, device=device)
    changed = torch.any(cur_grid != after_grid, dim=(1, 2))
    empties = (after_grid == 0)

    for b in range(B):
        if (not bool(changed[b].item())) or (not bool(empties[b].any().item())):
            mask[b, 0] = True
        else:
            for pos in range(16):
                r = pos // 4
                c = pos % 4
                if bool(empties[b, r, c].item()):
                    mask[b, 1 + 2*pos + 0] = True
                    mask[b, 1 + 2*pos + 1] = True
    return mask


def compute_2048_metrics(pred_grid: np.ndarray, target_grid: np.ndarray, cur_grid: np.ndarray) -> Dict[str, float]:
    """
    Compute diagnostic metrics for 2048 afterstate prediction.

    Args:
        pred_grid: predicted afterstate grid (4, 4) int64
        target_grid: ground truth afterstate grid (4, 4) int64
        cur_grid: current state grid before action (4, 4) int64

    Returns:
        Dict with keys: after_cell_acc, after_nonempty_acc, after_changed_acc
    """
    # Per-cell accuracy (mean over 16 cells)
    correct_cells = (pred_grid == target_grid)
    after_cell_acc = float(np.mean(correct_cells))

    # Accuracy on non-empty target cells
    nonempty_mask = (target_grid != 0)
    if np.any(nonempty_mask):
        after_nonempty_acc = float(np.mean(correct_cells[nonempty_mask]))
    else:
        after_nonempty_acc = 1.0  # all empty, trivially correct if all predicted empty

    # Accuracy on cells that changed due to the move
    changed_mask = (cur_grid != target_grid)
    if np.any(changed_mask):
        after_changed_acc = float(np.mean(correct_cells[changed_mask]))
    else:
        after_changed_acc = 1.0  # no cells changed (illegal move?)

    return {
        "after_cell_acc": after_cell_acc,
        "after_nonempty_acc": after_nonempty_acc,
        "after_changed_acc": after_changed_acc,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", type=str, choices=["othello", "2048"], default="othello")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", type=str, default="outputs_stoch_muzero_harness")

    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--num_styles", type=int, default=16)

    p.add_argument("--collect_episodes", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=200)

    p.add_argument("--train_steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--unroll", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)

    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=5000)

    # loss weights
    p.add_argument("--w_policy", type=float, default=1.0)
    p.add_argument("--w_value", type=float, default=0.25)
    p.add_argument("--w_reward", type=float, default=0.25)
    p.add_argument("--w_chance", type=float, default=1.0)
    p.add_argument("--w_aux", type=float, default=1.0)
    p.add_argument("--w_after_aux", type=float, default=1.0)
    p.add_argument("--w_style", type=float, default=0.2)
    p.add_argument("--w_inv", type=float, default=1.0)

    # 2048-specific loss reweighting
    p.add_argument("--empty_weight", type=float, default=0.2, help="Weight for empty cells in afterstate loss (2048)")
    p.add_argument("--changed_alpha", type=float, default=3.0, help="Multiplicative boost for changed cells in afterstate loss (2048)")

    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    outdir = Path(args.outdir) / args.game
    outdir.mkdir(parents=True, exist_ok=True)

    # game
    if args.game == "othello":
        game = Othello(img_size=args.img_size, num_styles=args.num_styles)
        aux_specs = {
            "board": AuxSpec(key="board", num_classes=3, shape=(8, 8)),
            "player": AuxSpec(key="player", num_classes=2, shape=()),
        }
        state_from_aux = _othello_state_from_aux
    else:
        game = Game2048(img_size=args.img_size, num_styles=args.num_styles)
        aux_specs = {
            "grid": AuxSpec(key="grid", num_classes=16, shape=(4, 4)),
        }
        state_from_aux = _game2048_state_from_aux

    print(f"Game={game.name} | action_size={game.action_size} | chance_size={game.chance_size} | obs={game.obs_shape} | styles={game.num_styles}")

    # collect data
    buf = collect_random_episodes(game, num_episodes=args.collect_episodes, max_steps=args.max_steps, rng=rng)
    print(f"Collected episodes: {len(buf)}")

    # model
    model = StochMuZeroNet(
        obs_shape=game.obs_shape,
        action_size=game.action_size,
        chance_size=game.chance_size,
        num_styles=game.num_styles,
        s_dim=256,
        u_dim=32,
        aux_specs=aux_specs,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # small eval episode
    eval_ep = buf.episodes[len(buf.episodes)//2]

    # training loop
    model.train()
    for step in range(1, args.train_steps + 1):
        batch = buf.sample_batch(args.batch, args.unroll, rng)

        obs0_u8 = batch["obs0"]
        style0 = batch["style0"]
        actions = batch["actions"]
        chances = batch["chances"]
        rewards = batch["rewards"]
        policy_t = batch["policy"]
        value_t = batch["value"]
        legal_t = batch["legal"]
        aux_t = batch["aux"]
        after_aux_t = batch["after_aux"]

        # obs tensors
        obs0 = torch.from_numpy(obs0_u8).to(device).float() / 255.0
        obs0 = obs0.unsqueeze(1)  # [B,1,H,W]

        # augmented obs for invariance
        obs0_aug_u8 = np.stack([augment_obs_u8(x, rng) for x in obs0_u8], axis=0)
        obs0_aug = torch.from_numpy(obs0_aug_u8).to(device).float() / 255.0
        obs0_aug = obs0_aug.unsqueeze(1)

        style0_t = torch.from_numpy(style0).to(device).long()

        actions_t = torch.from_numpy(actions).to(device).long()
        chances_t = torch.from_numpy(chances).to(device).long()
        rewards_t = torch.from_numpy(rewards).to(device).float()

        policy_t_t = torch.from_numpy(policy_t).to(device).float()
        value_t_t = torch.from_numpy(value_t).to(device).float()
        legal_t_t = torch.from_numpy(legal_t).to(device).bool()

        # aux targets
        aux_targets = {k: torch.from_numpy(v).to(device).long() for k, v in aux_t.items()}
        after_aux_targets = {k: torch.from_numpy(v).to(device).long() for k, v in after_aux_t.items()}

        # encode
        s, u = model.encode(obs0)
        s_aug, _u2 = model.encode(obs0_aug)
        inv_loss = F.mse_loss(s, s_aug)

        # root predictions
        pol_logits0, val0 = model.predict_policy_value(s)
        pol_loss = policy_loss_soft(pol_logits0, policy_t_t[:, 0], legal_t_t[:, 0])
        val_loss = F.mse_loss(val0, value_t_t[:, 0])

        aux_logits0 = model.decode_aux(s)
        aux_loss0 = 0.0
        for k, logits in aux_logits0.items():
            if game.name == "2048" and k == "grid":
                # Use weighted loss for perception too!
                # We use changed_alpha=1.0 because 'changing' doesn't apply to a static image
                aux_loss0 = aux_loss0 + after_aux_loss_2048_weighted(
                    logits, aux_targets[k][:, 0], aux_targets["grid"][:, 0], 
                    empty_weight=args.empty_weight, 
                    changed_alpha=1.0
                )
            else:
                aux_loss0 = aux_loss0 + aux_loss_from_logits(logits, aux_targets[k][:, 0])
        
        style_loss = torch.tensor(0.0, device=device)
        style_logits = model.predict_style_logits(u)
        if style_logits is not None:
            style_loss = F.cross_entropy(style_logits, style0_t, reduction="mean")

        # unroll
        rew_loss = torch.tensor(0.0, device=device)
        chance_loss = torch.tensor(0.0, device=device)
        after_aux_loss = torch.tensor(0.0, device=device)

        s_k = s
        for k in range(args.unroll):
            a_k = actions_t[:, k]
            c_k = chances_t[:, k]
            r_k = rewards_t[:, k]

            a_s = model.afterstate(s_k, a_k)

            # after-aux (rule core) supervision
            after_logits = model.decode_after_aux(a_s)
            for kk, logits in after_logits.items():
                if game.name == "2048" and kk == "grid":
                    # Use weighted loss for 2048 grid prediction
                    cur_grid_k = aux_targets["grid"][:, k]
                    after_aux_loss = after_aux_loss + after_aux_loss_2048_weighted(
                        logits, after_aux_targets[kk][:, k], cur_grid_k,
                        empty_weight=args.empty_weight,
                        changed_alpha=args.changed_alpha,
                    )
                else:
                    after_aux_loss = after_aux_loss + aux_loss_from_logits(logits, after_aux_targets[kk][:, k])

            # chance prediction
            chance_logits = model.predict_chance_logits(a_s)  # [B,C]
            if game.chance_size <= 1:
                # deterministic
                chance_loss = chance_loss + torch.tensor(0.0, device=device)
            else:
                # mask invalid chance outcomes using *true* (cur, after) aux targets
                if game.name == "2048":
                    cur_grid = aux_targets["grid"][:, k]
                    aft_grid = after_aux_targets["grid"][:, k]
                    cmask = chance_mask_2048(cur_grid, aft_grid, chance_size=game.chance_size)
                else:
                    cmask = torch.ones_like(chance_logits, dtype=torch.bool)

                logp = masked_log_softmax(chance_logits, cmask, dim=-1)
                chance_loss = chance_loss + (-logp[torch.arange(logp.shape[0]), c_k]).mean()

            # reward prediction
            r_pred = model.predict_reward(a_s, c_k)
            rew_loss = rew_loss + F.mse_loss(r_pred, r_k)

            # transition (teacher-forced chance)
            s_k = model.next_state(a_s, c_k)

            # policy/value/aux at next state
            pol_logits, val = model.predict_policy_value(s_k)
            pol_loss = pol_loss + policy_loss_soft(pol_logits, policy_t_t[:, k + 1], legal_t_t[:, k + 1])
            val_loss = val_loss + F.mse_loss(val, value_t_t[:, k + 1])

            aux_logits = model.decode_aux(s_k)
            for kk, logits in aux_logits.items():
                aux_loss0 = aux_loss0 + aux_loss_from_logits(logits, aux_targets[kk][:, k + 1])

        # normalize unroll additions
        pol_loss = pol_loss / (args.unroll + 1)
        val_loss = val_loss / (args.unroll + 1)
        aux_loss0 = aux_loss0 / (args.unroll + 1)
        after_aux_loss = after_aux_loss / max(1, args.unroll)
        chance_loss = chance_loss / max(1, args.unroll)
        rew_loss = rew_loss / max(1, args.unroll)

        loss = (
            args.w_policy * pol_loss +
            args.w_value * val_loss +
            args.w_aux * aux_loss0 +
            args.w_after_aux * after_aux_loss +
            args.w_chance * chance_loss +
            args.w_reward * rew_loss +
            args.w_style * style_loss +
            args.w_inv * inv_loss
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()

        if step % 200 == 0:
            print(
                f"step {step:06d} | loss {loss.item():.4f} "
                f"| pol {pol_loss.item():.4f} val {val_loss.item():.4f} "
                f"aux {aux_loss0.item():.4f} after_aux {after_aux_loss.item():.4f} "
                f"chance {chance_loss.item():.4f} rew {rew_loss.item():.4f} "
                f"style {style_loss.item():.4f} inv {inv_loss.item():.4f}"
            )

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # pick a rollout segment
                T = eval_ep.actions.shape[0]
                K = min(16, T)
                t0 = int(rng.randint(0, max(1, T - K)))
                obs0_u8 = eval_ep.obs_u8[t0]
                obs0 = torch.from_numpy(obs0_u8).to(device).float() / 255.0
                obs0 = obs0.unsqueeze(0).unsqueeze(0)

                s, u = model.encode(obs0)

                # record gt/pred states as images
                gt_imgs = []
                pr_imgs = []

                # gt at t0
                gt_aux0 = {k: eval_ep.aux[k][t0] for k in eval_ep.aux.keys()}
                gt_state0 = state_from_aux(gt_aux0)
                gt_imgs.append(game.render(gt_state0, style_id=3))

                # pred at t0
                pr_aux_logits0 = model.decode_aux(s)
                pr_aux0 = {}
                for k, logits in pr_aux_logits0.items():
                    if logits.dim() == 2:
                        pr_aux0[k] = torch.argmax(logits, dim=1)[0].cpu().numpy()
                    else:
                        pr_aux0[k] = torch.argmax(logits, dim=1)[0].cpu().numpy()
                pr_state0 = state_from_aux(pr_aux0)
                pr_imgs.append(game.render(pr_state0, style_id=3))

                exact_next = 0
                exact_after = 0
                chance_acc = 0

                # New diagnostic metrics for 2048
                after_cell_acc_sum = 0.0
                after_nonempty_acc_sum = 0.0
                after_changed_acc_sum = 0.0
                chance_nll_teacher_mask_sum = 0.0

                for k in range(K):
                    a = int(eval_ep.actions[t0 + k])
                    c = int(eval_ep.chances[t0 + k])

                    a_t = torch.tensor([a], device=device)
                    c_t = torch.tensor([c], device=device)

                    a_s = model.afterstate(s, a_t)

                    # afterstate aux accuracy
                    after_logits = model.decode_after_aux(a_s)
                    gt_after_aux = {kk: eval_ep.after_aux[kk][t0 + k] for kk in eval_ep.after_aux.keys()}
                    pr_after_aux = {}
                    ok_after = True
                    for kk, logits in after_logits.items():
                        pr = torch.argmax(logits, dim=1)[0].cpu().numpy()
                        pr_after_aux[kk] = pr
                        ok_after = ok_after and np.array_equal(pr, gt_after_aux[kk])
                    exact_after += int(ok_after)

                    # Compute per-cell metrics for 2048
                    if game.name == "2048":
                        pred_grid = pr_after_aux["grid"]
                        target_grid = gt_after_aux["grid"]
                        cur_grid = eval_ep.aux["grid"][t0 + k]
                        metrics_dict = compute_2048_metrics(pred_grid, target_grid, cur_grid)
                        after_cell_acc_sum += metrics_dict["after_cell_acc"]
                        after_nonempty_acc_sum += metrics_dict["after_nonempty_acc"]
                        after_changed_acc_sum += metrics_dict["after_changed_acc"]

                    # chance accuracy (argmax under true mask)
                    if game.chance_size <= 1:
                        chance_acc += 1
                    else:
                        ch_logits = model.predict_chance_logits(a_s)[0]  # [C]
                        if game.name == "2048":
                            cur_grid = torch.from_numpy(eval_ep.aux["grid"][t0 + k]).to(device).long().unsqueeze(0)
                            aft_grid = torch.from_numpy(eval_ep.after_aux["grid"][t0 + k]).to(device).long().unsqueeze(0)
                            cmask = chance_mask_2048(cur_grid, aft_grid, chance_size=game.chance_size)[0]
                        else:
                            cmask = torch.ones((game.chance_size,), dtype=torch.bool, device=device)
                        logp = masked_log_softmax(ch_logits.unsqueeze(0), cmask.unsqueeze(0), dim=-1)[0]
                        c_hat = int(torch.argmax(logp).item())
                        chance_acc += int(c_hat == c)

                        # Compute chance NLL using teacher mask
                        if game.name == "2048":
                            true_chance_logp = logp[c].item()
                            chance_nll_teacher_mask_sum += -true_chance_logp

                    # transition teacher-forced
                    s = model.next_state(a_s, c_t)

                    # gt next
                    gt_aux = {kk: eval_ep.aux[kk][t0 + k + 1] for kk in eval_ep.aux.keys()}
                    gt_state = state_from_aux(gt_aux)
                    gt_imgs.append(game.render(gt_state, style_id=3))

                    # pred next
                    pr_logits = model.decode_aux(s)
                    pr_aux = {}
                    ok = True
                    for kk, logits in pr_logits.items():
                        pr = torch.argmax(logits, dim=1)[0].cpu().numpy()
                        pr_aux[kk] = pr
                        ok = ok and np.array_equal(pr, gt_aux[kk])
                    exact_next += int(ok)
                    pr_state = state_from_aux(pr_aux)
                    pr_imgs.append(game.render(pr_state, style_id=3))

                gt_imgs = np.stack(gt_imgs, axis=0)
                pr_imgs = np.stack(pr_imgs, axis=0)

                _save_compare_strip_png(outdir / f"rollout_gt_vs_pred_step{step}.png", gt_imgs, pr_imgs)

                eval_msg = (
                    f"[EVAL step {step}] exact_after={exact_after}/{K} | "
                    f"chance_acc={chance_acc}/{K} | exact_next={exact_next}/{K}"
                )

                # Add 2048-specific metrics
                if game.name == "2048":
                    eval_msg += (
                        f" | after_cell_acc={after_cell_acc_sum/K:.3f}"
                        f" | after_nonempty_acc={after_nonempty_acc_sum/K:.3f}"
                        f" | after_changed_acc={after_changed_acc_sum/K:.3f}"
                        f" | chance_nll_teacher_mask={chance_nll_teacher_mask_sum/K:.3f}"
                    )

                eval_msg += " | saved rollout png"
                print(eval_msg)

            model.train()

        if step % args.save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, outdir / f"ckpt_step{step}.pt")
            print(f"Saved checkpoint -> {outdir / f'ckpt_step{step}.pt'}")

    # final save
    torch.save({"model": model.state_dict(), "args": vars(args)}, outdir / "ckpt_final.pt")
    print("Done. Outputs in:", outdir)


if __name__ == "__main__":
    main()
