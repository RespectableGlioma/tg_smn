from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(game: str, args):
    cmd = [
        sys.executable, "-m", "world_models.stoch_muzero_harness.train",
        "--game", game,
        "--seed", str(args.seed),
        "--device", args.device,
        "--outdir", args.outdir,
        "--img_size", str(args.img_size),
        "--num_styles", str(args.num_styles),
        "--collect_episodes", str(args.collect_episodes),
        "--max_steps", str(args.max_steps),
        "--train_steps", str(args.train_steps),
        "--batch", str(args.batch),
        "--unroll", str(args.unroll),
        "--lr", str(args.lr),
        "--eval_every", str(args.eval_every),
        "--save_every", str(args.save_every),
    ]
    print("\nRUN:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
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
    args = p.parse_args()

    run("othello", args)
    run("2048", args)


if __name__ == "__main__":
    main()
