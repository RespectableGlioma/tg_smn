"""Atari games via Gymnasium/ALE for macro discovery.

Self-contained version that doesn't require base.py to be present.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
except ImportError:
    raise ImportError(
        "gymnasium and ale-py required: pip install gymnasium[atari] ale-py"
    )

# Inline base class (avoids import issues in Colab)
ChanceOutcome = int

class Game(ABC):
    """Abstract base for games with afterstate separation."""

    @property
    @abstractmethod
    def action_space_size(self) -> int: pass

    @property
    @abstractmethod
    def chance_space_size(self) -> int: pass

    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]: pass

    @property
    def is_two_player(self) -> bool: return False

    def current_player(self, state) -> int: return 0

    @abstractmethod
    def reset(self): pass

    @abstractmethod
    def clone_state(self, state): pass

    @abstractmethod
    def legal_actions(self, state) -> List[int]: pass

    @abstractmethod
    def apply_action(self, state, action: int) -> Tuple[Any, float, Dict]: pass

    @abstractmethod
    def sample_chance(self, afterstate, info: Dict) -> int: pass

    @abstractmethod
    def get_chance_distribution(self, afterstate, info: Dict) -> np.ndarray: pass

    @abstractmethod
    def apply_chance(self, afterstate, chance: int): pass

    @abstractmethod
    def is_terminal(self, state) -> bool: pass

    @abstractmethod
    def encode_state(self, state) -> torch.Tensor: pass

    @abstractmethod
    def encode_afterstate(self, afterstate) -> torch.Tensor: pass


@dataclass
class AtariState:
    """Wrapper for Atari game state."""
    env: gym.Env
    observation: np.ndarray
    ale_state: Optional[bytes] = None
    done: bool = False
    lives: int = 0
    score: int = 0


class GameAtari(Game):
    """Atari game wrapper with deterministic mode for macro discovery."""

    ACTIONS = {
        0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
        6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT",
        10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE",
    }

    def __init__(
        self,
        game_name: str = "Breakout",
        frame_stack: int = 4,
        frame_skip: int = 4,
        grayscale: bool = True,
        flatten: bool = True,
        deterministic: bool = True,
        terminal_on_life_loss: bool = False,
    ):
        self.game_name = game_name
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.grayscale = grayscale
        self.flatten = flatten
        self.deterministic = deterministic
        self.terminal_on_life_loss = terminal_on_life_loss

        self._env = self._make_env()
        self._action_space_size = self._env.action_space.n
        self._action_meanings = self._env.unwrapped.get_action_meanings()

    def _make_env(self) -> gym.Env:
        env_name = f"ALE/{self.game_name}-v5"
        env = gym.make(
            env_name,
            repeat_action_probability=0.0 if self.deterministic else 0.25,
            frameskip=1,
            render_mode=None,
        )
        env = AtariPreprocessing(
            env,
            noop_max=0 if self.deterministic else 30,
            frame_skip=self.frame_skip,
            screen_size=84,
            terminal_on_life_loss=self.terminal_on_life_loss,
            grayscale_obs=self.grayscale,
            grayscale_newaxis=True,
            scale_obs=True,
        )
        env = FrameStackObservation(env, self.frame_stack)
        return env

    @property
    def action_space_size(self) -> int:
        return self._action_space_size

    @property
    def chance_space_size(self) -> int:
        return 1  # Deterministic

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        if self.flatten:
            return (84 * 84 * self.frame_stack,)
        return (self.frame_stack, 84, 84)

    @property
    def is_two_player(self) -> bool:
        return False

    def current_player(self, state: AtariState) -> int:
        return 0

    def reset(self) -> AtariState:
        env = self._make_env()
        obs, info = env.reset()
        ale_state = None
        if hasattr(env.unwrapped, 'ale'):
            ale_state = env.unwrapped.ale.cloneState()
        return AtariState(env=env, observation=obs, ale_state=ale_state,
                         done=False, lives=info.get('lives', 0), score=0)

    def clone_state(self, state: AtariState) -> AtariState:
        new_env = self._make_env()
        new_env.reset()
        if state.ale_state and hasattr(new_env.unwrapped, 'ale'):
            new_env.unwrapped.ale.restoreState(state.ale_state)
        return AtariState(env=new_env, observation=state.observation.copy(),
                         ale_state=state.ale_state, done=state.done,
                         lives=state.lives, score=state.score)

    def legal_actions(self, state: AtariState) -> List[int]:
        return [] if state.done else list(range(self._action_space_size))

    def apply_action(self, state: AtariState, action: int) -> Tuple[AtariState, float, Dict]:
        if state.done:
            return state, 0.0, {"done": True}

        obs, reward, terminated, truncated, info = state.env.step(action)
        done = terminated or truncated

        ale_state = None
        if hasattr(state.env.unwrapped, 'ale'):
            ale_state = state.env.unwrapped.ale.cloneState()

        new_state = AtariState(
            env=state.env, observation=obs, ale_state=ale_state,
            done=done, lives=info.get('lives', state.lives),
            score=state.score + int(reward),
        )
        return new_state, float(reward), info

    def sample_chance(self, afterstate: AtariState, info: Dict) -> int:
        return 0

    def get_chance_distribution(self, afterstate: AtariState, info: Dict) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)

    def apply_chance(self, afterstate: AtariState, chance: int) -> AtariState:
        return afterstate

    def is_terminal(self, state: AtariState) -> bool:
        return state.done

    def encode_state(self, state: AtariState) -> torch.Tensor:
        obs = state.observation
        if isinstance(obs, np.ndarray):
            if obs.ndim == 4 and obs.shape[-1] == 1:
                obs = obs.squeeze(-1)
            elif obs.ndim == 3 and obs.shape[0] != self.frame_stack:
                obs = obs.transpose(2, 0, 1)
        obs = np.asarray(obs, dtype=np.float32)
        if self.flatten:
            obs = obs.reshape(-1)
        return torch.tensor(obs, dtype=torch.float32)

    def encode_afterstate(self, afterstate: AtariState) -> torch.Tensor:
        return self.encode_state(afterstate)

    def render(self, state: AtariState) -> str:
        return f"{self.game_name} | Score: {state.score} | Lives: {state.lives}"

    def action_to_string(self, action: int) -> str:
        if action < len(self._action_meanings):
            return self._action_meanings[action]
        return f"Action_{action}"


def make_breakout(deterministic=True, flatten=True):
    return GameAtari("Breakout", deterministic=deterministic, flatten=flatten)

def make_pong(deterministic=True, flatten=True):
    return GameAtari("Pong", deterministic=deterministic, flatten=flatten)

def atari_action_decoder(action: int, game=None) -> str:
    if game:
        return game.action_to_string(action)
    return GameAtari.ACTIONS.get(action, f"Action_{action}")
