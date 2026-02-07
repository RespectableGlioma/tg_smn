"""Atari games via Gymnasium/ALE for macro discovery.

Atari games have deterministic physics (ball bounces, gravity, collisions)
making them ideal for discovering temporal abstractions. The stochasticity
in standard Atari benchmarks comes from:
- Sticky actions (disabled here for macro discovery)
- Random initial states (optional)

For macro discovery, we use DETERMINISTIC mode so that:
- All transitions have entropy ≈ 0
- The model can learn the game physics as compressible causal structure
- Macros represent reusable action sequences (positioning, attack patterns, etc.)

Observation encoding:
- 84x84 grayscale pixels (standard for DQN/MuZero)
- 4 stacked frames for temporal context
- Total: 84 × 84 × 4 = 28,224 features (flattened for MLP)

For CNN-based models, use observation_shape = (4, 84, 84) instead.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
except ImportError:
    raise ImportError(
        "gymnasium and ale-py are required: pip install gymnasium[atari] ale-py"
    )

from .base import Game, ChanceOutcome


@dataclass
class AtariState:
    """Wrapper for Atari game state."""

    env: gym.Env  # Reference to environment (for cloning)
    observation: np.ndarray  # Current stacked frames (4, 84, 84)
    ale_state: Optional[bytes] = None  # ALE internal state for cloning
    done: bool = False
    lives: int = 0
    score: int = 0


class GameAtari(Game):
    """
    Atari game wrapper for Stochastic MuZero.

    Uses deterministic mode (no sticky actions) so all transitions
    have low entropy, enabling macro discovery.

    Supported games:
    - Breakout: Ball physics, paddle positioning
    - Pong: Ball trajectory prediction
    - SpaceInvaders: Enemy patterns, shooting sequences
    - MsPacman: Navigation patterns (note: ghost AI adds stochasticity)
    - Qbert: Platform navigation
    - Seaquest: Diving/surfacing patterns

    Args:
        game_name: Atari game name (e.g., "Breakout", "Pong")
        frame_stack: Number of frames to stack (default 4)
        frame_skip: Action repeat / frame skip (default 4)
        grayscale: Use grayscale observations (default True)
        flatten: Flatten observation for MLP (default True)
        deterministic: Use deterministic mode - no sticky actions (default True)
        terminal_on_life_loss: End episode on life loss (default False)
        max_episode_steps: Maximum steps per episode (default 27000 = 30 min at 15 fps)
    """

    # Standard Atari action meanings (minimal action set varies by game)
    ACTIONS = {
        0: "NOOP",
        1: "FIRE",
        2: "UP",
        3: "RIGHT",
        4: "LEFT",
        5: "DOWN",
        6: "UPRIGHT",
        7: "UPLEFT",
        8: "DOWNRIGHT",
        9: "DOWNLEFT",
        10: "UPFIRE",
        11: "RIGHTFIRE",
        12: "LEFTFIRE",
        13: "DOWNFIRE",
        14: "UPRIGHTFIRE",
        15: "UPLEFTFIRE",
        16: "DOWNRIGHTFIRE",
        17: "DOWNLEFTFIRE",
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
        max_episode_steps: int = 27000,
    ):
        self.game_name = game_name
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.grayscale = grayscale
        self.flatten = flatten
        self.deterministic = deterministic
        self.terminal_on_life_loss = terminal_on_life_loss
        self.max_episode_steps = max_episode_steps

        # Create environment to get action space
        self._env = self._make_env()
        self._action_space_size = self._env.action_space.n
        self._action_meanings = self._env.unwrapped.get_action_meanings()

    def _make_env(self) -> gym.Env:
        """Create the Atari environment with preprocessing."""
        # Use v5 for gymnasium compatibility
        env_name = f"ALE/{self.game_name}-v5"

        # Create base environment
        # repeat_action_probability=0 for deterministic mode
        env = gym.make(
            env_name,
            repeat_action_probability=0.0 if self.deterministic else 0.25,
            frameskip=1,  # We handle frame skip in AtariPreprocessing
            render_mode=None,
        )

        # Apply standard Atari preprocessing
        env = AtariPreprocessing(
            env,
            noop_max=0 if self.deterministic else 30,  # No random noops for determinism
            frame_skip=self.frame_skip,
            screen_size=84,
            terminal_on_life_loss=self.terminal_on_life_loss,
            grayscale_obs=self.grayscale,
            grayscale_newaxis=True,  # Add channel dim: (84, 84, 1)
            scale_obs=True,  # Scale to [0, 1]
        )

        # Stack frames for temporal context
        env = FrameStackObservation(env, self.frame_stack)

        return env

    @property
    def action_space_size(self) -> int:
        return self._action_space_size

    @property
    def chance_space_size(self) -> int:
        # Deterministic Atari has no chance events
        # (sticky actions disabled, all physics deterministic)
        return 1

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        if self.flatten:
            # 84 * 84 * 4 = 28224 for MLP
            return (84 * 84 * self.frame_stack,)
        else:
            # (4, 84, 84) for CNN
            return (self.frame_stack, 84, 84)

    @property
    def is_two_player(self) -> bool:
        return False

    def current_player(self, state: AtariState) -> int:
        return 0  # Single player

    def reset(self) -> AtariState:
        """Reset environment and return initial state."""
        # Create fresh environment
        env = self._make_env()
        obs, info = env.reset()

        # Get ALE state for cloning
        ale_state = None
        if hasattr(env.unwrapped, 'ale'):
            ale_state = env.unwrapped.ale.cloneState()

        return AtariState(
            env=env,
            observation=obs,
            ale_state=ale_state,
            done=False,
            lives=info.get('lives', 0),
            score=0,
        )

    def clone_state(self, state: AtariState) -> AtariState:
        """Clone the game state.

        Note: Full state cloning in Atari requires ALE state restoration,
        which can be expensive. For MCTS, we typically use the learned
        dynamics model instead of true state cloning.
        """
        # Create new environment
        new_env = self._make_env()
        new_env.reset()

        # Restore ALE state if available
        if state.ale_state is not None and hasattr(new_env.unwrapped, 'ale'):
            new_env.unwrapped.ale.restoreState(state.ale_state)
            # Get observation from restored state
            obs = new_env.unwrapped.ale.getScreenGrayscale() if self.grayscale else new_env.unwrapped.ale.getScreenRGB()
        else:
            obs = state.observation.copy()

        return AtariState(
            env=new_env,
            observation=obs,
            ale_state=state.ale_state,
            done=state.done,
            lives=state.lives,
            score=state.score,
        )

    def legal_actions(self, state: AtariState) -> List[int]:
        """All actions are legal in Atari (though some may be no-ops)."""
        if state.done:
            return []
        return list(range(self._action_space_size))

    def apply_action(
        self, state: AtariState, action: int
    ) -> Tuple[AtariState, float, Dict[str, Any]]:
        """Apply action and return (afterstate, reward, info).

        In deterministic Atari, the afterstate IS the next state
        since there are no chance events.
        """
        if state.done:
            return state, 0.0, {"done": True}

        # Step environment
        obs, reward, terminated, truncated, info = state.env.step(action)
        done = terminated or truncated

        # Get ALE state for potential cloning
        ale_state = None
        if hasattr(state.env.unwrapped, 'ale'):
            ale_state = state.env.unwrapped.ale.cloneState()

        new_state = AtariState(
            env=state.env,
            observation=obs,
            ale_state=ale_state,
            done=done,
            lives=info.get('lives', state.lives),
            score=state.score + int(reward),
        )

        return new_state, float(reward), info

    def sample_chance(
        self, afterstate: AtariState, info: Dict[str, Any]
    ) -> ChanceOutcome:
        """No chance events in deterministic Atari."""
        return 0

    def get_chance_distribution(
        self, afterstate: AtariState, info: Dict[str, Any]
    ) -> np.ndarray:
        """Deterministic: probability 1 for outcome 0."""
        return np.array([1.0], dtype=np.float32)

    def apply_chance(
        self, afterstate: AtariState, chance: ChanceOutcome
    ) -> AtariState:
        """Identity for deterministic games."""
        return afterstate

    def is_terminal(self, state: AtariState) -> bool:
        return state.done

    def encode_state(self, state: AtariState) -> torch.Tensor:
        """Encode observation as tensor.

        Input observation shape: (4, 84, 84, 1) from FrameStack
        Output: (28224,) if flatten else (4, 84, 84)
        """
        obs = state.observation

        # Handle different observation formats
        if isinstance(obs, np.ndarray):
            # FrameStack returns (4, 84, 84, 1) - remove last dim
            if obs.ndim == 4 and obs.shape[-1] == 1:
                obs = obs.squeeze(-1)  # (4, 84, 84)
            elif obs.ndim == 3 and obs.shape[0] != self.frame_stack:
                # Might be (84, 84, 4) - transpose
                obs = obs.transpose(2, 0, 1)

        obs = np.asarray(obs, dtype=np.float32)

        if self.flatten:
            obs = obs.reshape(-1)

        return torch.tensor(obs, dtype=torch.float32)

    def encode_afterstate(self, afterstate: AtariState) -> torch.Tensor:
        """Same as encode_state for deterministic games."""
        return self.encode_state(afterstate)

    def render(self, state: AtariState) -> str:
        """Text representation of game state."""
        return (
            f"{self.game_name} | "
            f"Score: {state.score} | "
            f"Lives: {state.lives} | "
            f"Done: {state.done}"
        )

    def action_to_string(self, action: int) -> str:
        """Convert action index to readable string."""
        if action < len(self._action_meanings):
            return self._action_meanings[action]
        return f"Action_{action}"

    def get_score(self, state: AtariState) -> int:
        """Get current game score."""
        return state.score


# Convenience constructors for common games

def make_breakout(deterministic: bool = True, flatten: bool = True) -> GameAtari:
    """Create Breakout environment.

    Good for macro discovery:
    - Ball trajectory prediction
    - Paddle positioning sequences
    - Brick-breaking patterns
    """
    return GameAtari(
        game_name="Breakout",
        deterministic=deterministic,
        flatten=flatten,
    )


def make_pong(deterministic: bool = True, flatten: bool = True) -> GameAtari:
    """Create Pong environment.

    Good for macro discovery:
    - Ball interception patterns
    - Serve sequences
    - Defensive positioning
    """
    return GameAtari(
        game_name="Pong",
        deterministic=deterministic,
        flatten=flatten,
    )


def make_space_invaders(deterministic: bool = True, flatten: bool = True) -> GameAtari:
    """Create Space Invaders environment.

    Good for macro discovery:
    - Shooting patterns
    - Dodging sequences
    - Enemy wave timing
    """
    return GameAtari(
        game_name="SpaceInvaders",
        deterministic=deterministic,
        flatten=flatten,
    )


def make_qbert(deterministic: bool = True, flatten: bool = True) -> GameAtari:
    """Create Q*bert environment.

    Good for macro discovery:
    - Platform navigation patterns
    - Color completion sequences
    - Enemy avoidance
    """
    return GameAtari(
        game_name="Qbert",
        deterministic=deterministic,
        flatten=flatten,
    )


def make_seaquest(deterministic: bool = True, flatten: bool = True) -> GameAtari:
    """Create Seaquest environment.

    Good for macro discovery:
    - Diving/surfacing patterns
    - Rescue sequences
    - Enemy engagement
    """
    return GameAtari(
        game_name="Seaquest",
        deterministic=deterministic,
        flatten=flatten,
    )


# Action decoder for macro display
def atari_action_decoder(action: int, game: Optional[GameAtari] = None) -> str:
    """Decode action index to readable string."""
    if game is not None:
        return game.action_to_string(action)
    return GameAtari.ACTIONS.get(action, f"Action_{action}")
