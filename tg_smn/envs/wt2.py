from __future__ import annotations

import random
from dataclasses import asdict
from typing import List

import numpy as np
import torch
from datasets import load_dataset

from ..config import WT2EnvCfg
from ..tokenization import build_vocab, encode_docs, split_into_docs
from .base import EnvData


def _make_vocab_perm(vocab_size: int, fixed_ids: List[int], seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    perm = torch.arange(vocab_size)
    fixed = set(fixed_ids)
    movable = [i for i in range(vocab_size) if i not in fixed]
    shuffled = torch.tensor(movable)[torch.randperm(len(movable), generator=g)]
    perm[movable] = shuffled
    return perm


def _apply_perm(stream: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return perm[stream]


def _drift_perms(vocab_size: int, fixed_ids: List[int], n_tasks: int, base_seed: int, swaps_per_task: int) -> List[torch.Tensor]:
    # Start from a random perm, then apply small random swaps each task.
    base = _make_vocab_perm(vocab_size, fixed_ids, seed=base_seed)
    perms = [base.clone()]
    rng = random.Random(base_seed + 999)
    movable = [i for i in range(vocab_size) if i not in set(fixed_ids)]

    for t in range(1, n_tasks):
        p = perms[-1].clone()
        for _ in range(swaps_per_task):
            a = rng.choice(movable)
            b = rng.choice(movable)
            # swap images of a and b
            va = int(p[a].item())
            vb = int(p[b].item())
            p[a] = vb
            p[b] = va
        perms.append(p)
    return perms


def build_wt2_env(cfg: WT2EnvCfg, *, min_freq: int = 2) -> EnvData:
    """Build a continual LM environment from WikiText-2.

    If cfg.permuted_vocab=True, each task applies a different permutation to token IDs (excluding specials).
    This creates an adversarial nonstationary environment similar to PermutedMNIST.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_docs = split_into_docs(ds["train"]["text"])

    if cfg.max_docs_total is not None:
        train_docs = train_docs[: cfg.max_docs_total]

    # Shuffle docs deterministically
    rng = random.Random(0)
    idx = list(range(len(train_docs)))
    rng.shuffle(idx)
    train_docs = [train_docs[i] for i in idx]

    tasks = np.array_split(train_docs, cfg.n_tasks)

    # Build vocab from all train docs (unpermuted)
    vocab = build_vocab(train_docs, min_freq=min_freq)
    vocab_size = len(vocab.itos)

    fixed_ids = [vocab.pad_id, vocab.unk_id, vocab.eos_id]

    # Build perms
    perms: List[torch.Tensor] = []
    if cfg.permuted_vocab:
        if cfg.perm_mode == "unique":
            perms = [_make_vocab_perm(vocab_size, fixed_ids, seed=1234 + t) for t in range(cfg.n_tasks)]
        elif cfg.perm_mode == "repeat":
            base_perms = [_make_vocab_perm(vocab_size, fixed_ids, seed=1234 + t) for t in range(cfg.repeat_k)]
            perms = [base_perms[t % cfg.repeat_k] for t in range(cfg.n_tasks)]
        elif cfg.perm_mode == "drift":
            perms = _drift_perms(vocab_size, fixed_ids, cfg.n_tasks, base_seed=1234, swaps_per_task=cfg.drift_swaps)
        else:
            raise ValueError(f"Unknown perm_mode: {cfg.perm_mode}")
    else:
        perms = [torch.arange(vocab_size) for _ in range(cfg.n_tasks)]

    task_train_streams: List[torch.Tensor] = []
    task_val_streams: List[torch.Tensor] = []
    task_test_streams: List[torch.Tensor] = []

    # Use WT2 validation/test docs as a source for per-task held-out docs.
    val_docs_all = split_into_docs(ds["validation"]["text"])
    test_docs_all = split_into_docs(ds["test"]["text"])
    val_splits = np.array_split(val_docs_all, cfg.n_tasks)
    test_splits = np.array_split(test_docs_all, cfg.n_tasks)

    for t in range(cfg.n_tasks):
        docs_list = list(tasks[t])
        n_val = max(1, int(len(docs_list) * cfg.val_frac_per_task))
        val_part = docs_list[:n_val]
        trn_part = docs_list[n_val:]

        trn = encode_docs(trn_part, vocab)
        va = encode_docs(val_part, vocab)

        # Per-task test/val from global splits (still useful even for permuted vocab)
        te = encode_docs(list(test_splits[t]), vocab)

        perm = perms[t]
        if cfg.permuted_vocab:
            trn = _apply_perm(trn, perm)
            va = _apply_perm(va, perm)
            te = _apply_perm(te, perm)

        task_train_streams.append(trn)
        task_val_streams.append(va)
        task_test_streams.append(te)

    meta = {
        "cfg": asdict(cfg),
        "permuted_vocab": cfg.permuted_vocab,
        "perm_mode": cfg.perm_mode,
    }

    return EnvData(
        name=cfg.name,
        vocab_size=vocab_size,
        pad_id=vocab.pad_id,
        unk_id=vocab.unk_id,
        eos_id=vocab.eos_id,
        task_train_streams=task_train_streams,
        task_val_streams=task_val_streams,
        task_test_streams=task_test_streams,
        meta=meta,
    )
