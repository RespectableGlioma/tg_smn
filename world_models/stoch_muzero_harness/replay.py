from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Episode:
    obs_u8: np.ndarray            # [T+1,H,W] uint8
    style: np.ndarray             # [T+1] int64
    aux: Dict[str, np.ndarray]    # key -> [T+1,...] int64
    actions: np.ndarray           # [T] int64
    chances: np.ndarray           # [T] int64
    rewards: np.ndarray           # [T] float32
    done: np.ndarray              # [T] bool
    policy: np.ndarray            # [T+1,A] float32 targets
    value: np.ndarray             # [T+1] float32 targets
    legal: np.ndarray             # [T+1,A] bool
    after_aux: Dict[str, np.ndarray]  # key -> [T,...] int64 targets for afterstate


class ReplayBuffer:
    def __init__(self, capacity_episodes: int = 2000):
        self.capacity = capacity_episodes
        self.episodes: List[Episode] = []
        self._lengths: List[int] = []

    def add(self, ep: Episode):
        self.episodes.append(ep)
        self._lengths.append(int(ep.actions.shape[0]))
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
            self._lengths.pop(0)

    def __len__(self) -> int:
        return len(self.episodes)

    def sample_batch(self, batch_size: int, unroll: int, rng: np.random.RandomState):
        assert len(self.episodes) > 0, "replay buffer empty"
        # choose episodes with enough length
        idxs = [i for i, T in enumerate(self._lengths) if T >= unroll]
        if len(idxs) == 0:
            raise RuntimeError(f"No episodes with length >= unroll({unroll}). Collected too-short episodes.")
        chosen = rng.choice(idxs, size=batch_size, replace=True)

        # infer shapes from first ep
        ep0 = self.episodes[int(chosen[0])]
        H, W = ep0.obs_u8.shape[1:]
        A = ep0.policy.shape[1]

        obs0 = np.zeros((batch_size, H, W), dtype=np.uint8)
        style0 = np.zeros((batch_size,), dtype=np.int64)

        actions = np.zeros((batch_size, unroll), dtype=np.int64)
        chances = np.zeros((batch_size, unroll), dtype=np.int64)
        rewards = np.zeros((batch_size, unroll), dtype=np.float32)

        policy = np.zeros((batch_size, unroll + 1, A), dtype=np.float32)
        value = np.zeros((batch_size, unroll + 1), dtype=np.float32)
        legal = np.zeros((batch_size, unroll + 1, A), dtype=np.bool_)

        aux: Dict[str, np.ndarray] = {}
        after_aux: Dict[str, np.ndarray] = {}

        for k, v in ep0.aux.items():
            aux[k] = np.zeros((batch_size, unroll + 1, *v.shape[1:]), dtype=v.dtype)
        for k, v in ep0.after_aux.items():
            after_aux[k] = np.zeros((batch_size, unroll, *v.shape[1:]), dtype=v.dtype)

        for b, ei in enumerate(chosen):
            ep = self.episodes[int(ei)]
            T = ep.actions.shape[0]
            t0 = int(rng.randint(0, T - unroll + 1))  # inclusive
            obs0[b] = ep.obs_u8[t0]
            style0[b] = ep.style[t0]

            actions[b] = ep.actions[t0:t0+unroll]
            chances[b] = ep.chances[t0:t0+unroll]
            rewards[b] = ep.rewards[t0:t0+unroll]

            policy[b] = ep.policy[t0:t0+unroll+1]
            value[b] = ep.value[t0:t0+unroll+1]
            legal[b] = ep.legal[t0:t0+unroll+1]

            for k in aux.keys():
                aux[k][b] = ep.aux[k][t0:t0+unroll+1]
            for k in after_aux.keys():
                after_aux[k][b] = ep.after_aux[k][t0:t0+unroll]

        return {
            "obs0": obs0,
            "style0": style0,
            "actions": actions,
            "chances": chances,
            "rewards": rewards,
            "policy": policy,
            "value": value,
            "legal": legal,
            "aux": aux,
            "after_aux": after_aux,
        }
