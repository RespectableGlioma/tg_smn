from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch


class ReplayBufferLM:
    def __init__(self, max_seqs: int = 20000):
        self.max = int(max_seqs)
        self.x: List[torch.Tensor] = []
        self.y: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.x)

    def add_batch(self, xb: torch.Tensor, yb: torch.Tensor) -> None:
        xb = xb.detach().cpu()
        yb = yb.detach().cpu()
        for i in range(xb.size(0)):
            if len(self.x) < self.max:
                self.x.append(xb[i])
                self.y.append(yb[i])
            else:
                j = random.randrange(0, self.max)
                self.x[j] = xb[i]
                self.y[j] = yb[i]

    def sample(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = np.random.randint(0, len(self.x), size=batch_size)
        xb = torch.stack([self.x[i] for i in idx]).to(device)
        yb = torch.stack([self.y[i] for i in idx]).to(device)
        return xb, yb
