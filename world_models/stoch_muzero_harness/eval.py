from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch

from .games import Othello, Game2048
from .model import StochMuZeroNet, AuxSpec, masked_log_softmax
from .mcts import StochasticMCTS, MCTSConfig


def load_model(game, ckpt_path: str, device: torch.device) -> StochMuZeroNet:
    if game.name == "othello":
        aux_specs = {
            "board": AuxSpec(key="board", num_classes=3, shape=(8, 8)),
            "player": AuxSpec(key="player", num_classes=2, shape=()),
        }
    else:
        aux_specs = {"grid": AuxSpec(key="grid", num_classes=16, shape=(4, 4))}

    model = StochMuZeroNet(
        obs_shape=game.obs_shape,
        action_size=game.action_size,
        chance_size=game.chance_size,
        num_styles=game.num_styles,
        s_dim=256,
        u_dim=32,
        aux_specs=aux_specs,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def chance_mask_2048(cur_grid: torch.Tensor, after_grid: torch.Tensor, chance_size: int = 33) -> torch.Tensor:
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


def eval_prediction(game, model: StochMuZeroNet, episodes: int, max_steps: int, seed: int, device: torch.device):
    rng = np.random.RandomState(seed)

    exact_after = 0
    exact_next = 0
    chance_acc = 0
    total = 0

    # New diagnostic metrics for 2048
    after_cell_acc_sum = 0.0
    after_nonempty_acc_sum = 0.0
    after_changed_acc_sum = 0.0
    chance_nll_teacher_mask_sum = 0.0

    for _ in range(episodes):
        state = game.reset(rng)
        style = int(rng.randint(0, game.num_styles))
        obs0_u8 = game.render(state, style)
        aux0 = game.encode_aux(state)

        # for comparison
        for _t in range(max_steps):
            legal = game.legal_actions(state)
            legal_ids = np.where(legal)[0]
            if len(legal_ids) == 0:
                break
            a = int(rng.choice(legal_ids))

            after, r, valid, info = game.apply_action(state, a)
            after_aux = game.encode_aux(after)
            c = game.sample_chance(after, info, rng)
            nxt = game.apply_chance(after, c, info)
            nxt_aux = game.encode_aux(nxt)

            # model rollout one step (teacher forced chance)
            obs0 = torch.from_numpy(obs0_u8).to(device).float() / 255.0
            obs0 = obs0.unsqueeze(0).unsqueeze(0)
            s, _u = model.encode(obs0)

            a_t = torch.tensor([a], device=device)
            c_t = torch.tensor([c], device=device)

            a_s = model.afterstate(s, a_t)
            after_logits = model.decode_after_aux(a_s)
            ok_after = True
            for k, logits in after_logits.items():
                pr = torch.argmax(logits, dim=1)[0].cpu().numpy()
                ok_after = ok_after and np.array_equal(pr, after_aux[k])
            exact_after += int(ok_after)

            # Compute per-cell metrics for 2048
            if game.name == "2048":
                pred_grid = torch.argmax(after_logits["grid"], dim=1)[0].cpu().numpy()
                target_grid = after_aux["grid"]
                cur_grid = aux0["grid"]
                metrics_dict = compute_2048_metrics(pred_grid, target_grid, cur_grid)
                after_cell_acc_sum += metrics_dict["after_cell_acc"]
                after_nonempty_acc_sum += metrics_dict["after_nonempty_acc"]
                after_changed_acc_sum += metrics_dict["after_changed_acc"]

            if game.chance_size <= 1:
                chance_acc += 1
            else:
                ch_logits = model.predict_chance_logits(a_s)
                # mask by true (cur,after)
                cur_grid = torch.from_numpy(aux0["grid"]).to(device).long().unsqueeze(0)
                aft_grid = torch.from_numpy(after_aux["grid"]).to(device).long().unsqueeze(0)
                cmask = chance_mask_2048(cur_grid, aft_grid, chance_size=game.chance_size)
                logp = masked_log_softmax(ch_logits, cmask, dim=-1)
                c_hat = int(torch.argmax(logp[0]).item())
                chance_acc += int(c_hat == c)

                # Compute chance NLL using teacher mask
                if game.name == "2048":
                    # NLL of true chance outcome under teacher mask
                    true_chance_logp = logp[0, c].item()
                    chance_nll_teacher_mask_sum += -true_chance_logp

            s1 = model.next_state(a_s, c_t)
            nxt_logits = model.decode_aux(s1)
            ok_next = True
            for k, logits in nxt_logits.items():
                pr = torch.argmax(logits, dim=1)[0].cpu().numpy()
                ok_next = ok_next and np.array_equal(pr, nxt_aux[k])
            exact_next += int(ok_next)

            total += 1

            # advance
            state = nxt
            aux0 = nxt_aux
            style = int(rng.randint(0, game.num_styles))
            obs0_u8 = game.render(state, style)

            if game.is_terminal(state):
                break

    if total == 0:
        print("No transitions evaluated.")
        return

    result = (
        f"Prediction eval over {total} transitions: "
        f"exact_after={exact_after/total:.3f} | "
        f"chance_acc={chance_acc/total:.3f} | "
        f"exact_next={exact_next/total:.3f}"
    )

    # Add 2048-specific metrics if applicable
    if game.name == "2048" and total > 0:
        result += (
            f" | after_cell_acc={after_cell_acc_sum/total:.3f}"
            f" | after_nonempty_acc={after_nonempty_acc_sum/total:.3f}"
            f" | after_changed_acc={after_changed_acc_sum/total:.3f}"
            f" | chance_nll_teacher_mask={chance_nll_teacher_mask_sum/total:.3f}"
        )

    print(result)


def eval_mcts_2048(game, model: StochMuZeroNet, episodes: int, max_steps: int, seed: int, device: torch.device,
                  sims: int, entropy_thr: float):
    rng = np.random.RandomState(seed)

    cfg = MCTSConfig(num_simulations=sims, chance_entropy_threshold=entropy_thr)
    scores = []
    max_tiles = []

    for ep in range(episodes):
        state = game.reset(rng)
        total_reward = 0.0

        for t in range(max_steps):
            legal = game.legal_actions(state)
            if not bool(legal.any()):
                break

            style = int(rng.randint(0, game.num_styles))
            obs_u8 = game.render(state, style)
            obs01 = torch.from_numpy(obs_u8).to(device).float() / 255.0
            obs01 = obs01.unsqueeze(0).unsqueeze(0)

            mcts = StochasticMCTS(model, cfg, action_mask=legal)
            a, _probs = mcts.run(obs01, rng)

            after, r, valid, info = game.apply_action(state, a)
            c = game.sample_chance(after, info, rng)
            state = game.apply_chance(after, c, info)
            total_reward += float(r)

            if game.is_terminal(state):
                break

        scores.append(total_reward)
        max_tiles.append(int(np.max(state.grid)))

    print(
        f"MCTS(2048) episodes={episodes} sims={sims} thr={entropy_thr} | "
        f"avg_score={np.mean(scores):.1f} ± {np.std(scores):.1f} | "
        f"avg_max_tile={np.mean(max_tiles):.1f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", type=str, choices=["othello", "2048"], default="2048")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--num_styles", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=200)

    p.add_argument("--mcts_sims", type=int, default=0)
    p.add_argument("--entropy_thr", type=float, default=0.5)

    args = p.parse_args()

    device = torch.device(args.device)

    if args.game == "othello":
        game = Othello(img_size=args.img_size, num_styles=args.num_styles)
    else:
        game = Game2048(img_size=args.img_size, num_styles=args.num_styles)

    model = load_model(game, args.ckpt, device)

    eval_prediction(game, model, episodes=args.episodes, max_steps=args.max_steps, seed=args.seed, device=device)

    if args.game == "2048" and args.mcts_sims > 0:
        eval_mcts_2048(game, model, episodes=max(5, args.episodes // 5), max_steps=args.max_steps, seed=args.seed + 1,
                      device=device, sims=args.mcts_sims, entropy_thr=args.entropy_thr)


if __name__ == "__main__":
    main()
