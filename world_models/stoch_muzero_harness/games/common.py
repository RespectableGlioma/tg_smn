from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Protocol, Optional
import numpy as np


@dataclass
class StepOutput:
    next_state: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class Game(Protocol):
    """Game interface required by the harness.

    Notes:
      - Observations are grayscale uint8 images (H,W).
      - State is an internal symbolic representation (e.g., board arrays).
      - The decomposition follows Stochastic MuZero:
          afterstate = apply_action(state, action)           # deterministic rule core
          chance     = sample_chance(afterstate)             # stochastic event
          next_state = apply_chance(afterstate, chance)      # deterministic given chance
    """

    name: str
    obs_shape: Tuple[int, int]
    action_size: int
    chance_size: int
    num_styles: int

    def reset(self, rng: np.random.RandomState) -> Any: ...
    def legal_actions(self, state: Any) -> np.ndarray: ...
    def apply_action(self, state: Any, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]: ...
    def chance_mask(self, afterstate: Any, action_info: Dict[str, Any]) -> np.ndarray: ...
    def chance_probs(self, afterstate: Any, action_info: Dict[str, Any]) -> np.ndarray: ...
    def sample_chance(self, afterstate: Any, action_info: Dict[str, Any], rng: np.random.RandomState) -> int: ...
    def apply_chance(self, afterstate: Any, chance: int, action_info: Dict[str, Any]) -> Any: ...
    def is_terminal(self, state: Any) -> bool: ...
    def encode_aux(self, state: Any) -> Dict[str, np.ndarray]: ...
    def render(self, state: Any, style_id: int) -> np.ndarray: ...


def augment_obs_u8(obs_u8: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Simple augmentation for invariance: brightness + gaussian noise.

    Args:
      obs_u8: uint8 image (H,W)
      rng: RNG

    Returns:
      uint8 image (H,W)
    """
    x = obs_u8.astype(np.float32) / 255.0
    b = rng.uniform(0.6, 1.4)
    x = x * b
    x = x + rng.randn(*x.shape).astype(np.float32) * 0.05
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)
