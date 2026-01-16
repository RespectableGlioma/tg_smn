from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class LMSeqDataset(Dataset):
    """Turns a token stream into fixed-length next-token prediction sequences."""

    def __init__(self, token_stream: torch.Tensor, seq_len: int):
        self.tokens = token_stream
        self.seq_len = seq_len
        self.n = max(0, (len(self.tokens) - (seq_len + 1)) // seq_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.seq_len
        chunk = self.tokens[i : i + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


@dataclass
class EnvData:
    name: str
    vocab_size: int
    pad_id: int
    unk_id: int
    eos_id: int

    task_train_streams: List[torch.Tensor]
    task_val_streams: List[torch.Tensor]
    task_test_streams: List[torch.Tensor]

    # Optional metadata (domain schedule, permutations, etc.)
    meta: Dict[str, Any]


def make_task_loaders(
    env: EnvData,
    seq_len: int,
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
    train_loaders, val_loaders, test_loaders = [], [], []

    for s in env.task_train_streams:
        ds = LMSeqDataset(s, seq_len)
        train_loaders.append(
            DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        )

    for s in env.task_val_streams:
        ds = LMSeqDataset(s, seq_len)
        val_loaders.append(
            DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        )

    for s in env.task_test_streams:
        ds = LMSeqDataset(s, seq_len)
        test_loaders.append(
            DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        )

    return train_loaders, val_loaders, test_loaders
