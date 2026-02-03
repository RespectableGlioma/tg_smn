"""Abstract base class for game environments.

Games in Stochastic MuZero have a clear separation between:
1. Deterministic action effects (afterstate)
2. Stochastic chance outcomes (environment randomness)
3. Deterministic chance application (final state)

This allows the model to learn:
- φ(s, a) → afterstate (the "rule" - deterministic)
- σ(c | afterstate) → chance distribution (environment stochasticity)
- g(afterstate, c) → next state (deterministic given chance)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar
import numpy as np
import torch

State = TypeVar("State")
Afterstate = TypeVar("Afterstate")
ChanceOutcome = int  # Chance outcomes are discrete integers


@dataclass
class StepResult:
    """Result of applying an action and chance outcome."""

    afterstate: Any  # State after deterministic action, before chance
    next_state: Any  # State after chance outcome
    reward: float
    done: bool
    chance_outcome: int  # The sampled chance outcome
    info: Dict[str, Any]  # Additional information (e.g., valid moves changed)


class Game(ABC):
    """
    Abstract base class for game environments with afterstate separation.

    The transition model is:
        s_t --[action a]--> afterstate --[chance c]--> s_{t+1}

    Where:
        - s_t → afterstate is DETERMINISTIC (the "rule")
        - afterstate → s_{t+1} depends on stochastic chance c
        - For fully deterministic games, chance_space_size = 1
    """

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Number of possible actions."""
        pass

    @property
    @abstractmethod
    def chance_space_size(self) -> int:
        """Number of possible chance outcomes (1 for deterministic games)."""
        pass

    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """Shape of the observation/state encoding."""
        pass

    @abstractmethod
    def reset(self) -> State:
        """
        Reset the game to initial state.

        Returns:
            Initial state of the game.
        """
        pass

    @abstractmethod
    def clone_state(self, state: State) -> State:
        """Create a deep copy of the state."""
        pass

    @abstractmethod
    def legal_actions(self, state: State) -> List[int]:
        """
        Get list of legal actions in the given state.

        Args:
            state: Current game state.

        Returns:
            List of legal action indices.
        """
        pass

    @abstractmethod
    def apply_action(self, state: State, action: int) -> Tuple[Afterstate, float, Dict[str, Any]]:
        """
        Apply action to get afterstate (DETERMINISTIC).

        This is the "rule core" - the deterministic part of the transition.
        No randomness should occur here.

        Args:
            state: Current game state.
            action: Action to apply.

        Returns:
            afterstate: State after deterministic action application.
            reward: Immediate reward from the action.
            info: Additional information needed for chance sampling.
        """
        pass

    @abstractmethod
    def sample_chance(self, afterstate: Afterstate, info: Dict[str, Any]) -> ChanceOutcome:
        """
        Sample a chance outcome given the afterstate.

        For deterministic games, this should always return 0.

        Args:
            afterstate: State after action, before chance.
            info: Information from apply_action (e.g., available spawn positions).

        Returns:
            Sampled chance outcome index.
        """
        pass

    @abstractmethod
    def get_chance_distribution(
        self, afterstate: Afterstate, info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Get the full probability distribution over chance outcomes.

        For deterministic games, this should return [1.0].

        Args:
            afterstate: State after action, before chance.
            info: Information from apply_action.

        Returns:
            Probability distribution over chance outcomes of shape (chance_space_size,).
        """
        pass

    @abstractmethod
    def apply_chance(self, afterstate: Afterstate, chance: ChanceOutcome) -> State:
        """
        Apply chance outcome to get next state (DETERMINISTIC given chance).

        Args:
            afterstate: State after action, before chance.
            chance: The chance outcome to apply.

        Returns:
            Next game state.
        """
        pass

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Check if the state is terminal (game over)."""
        pass

    @abstractmethod
    def encode_state(self, state: State) -> torch.Tensor:
        """
        Encode state as a tensor for neural network input.

        Args:
            state: Game state to encode.

        Returns:
            Tensor of shape observation_shape.
        """
        pass

    @abstractmethod
    def encode_afterstate(self, afterstate: Afterstate) -> torch.Tensor:
        """
        Encode afterstate as a tensor.

        Args:
            afterstate: Afterstate to encode.

        Returns:
            Tensor of shape observation_shape.
        """
        pass

    def step(self, state: State, action: int) -> StepResult:
        """
        Full step: apply action, sample chance, apply chance.

        Convenience method that combines the three-phase transition.

        Args:
            state: Current game state.
            action: Action to take.

        Returns:
            StepResult with afterstate, next_state, reward, done, chance_outcome, info.
        """
        afterstate, reward, info = self.apply_action(state, action)
        chance = self.sample_chance(afterstate, info)
        next_state = self.apply_chance(afterstate, chance)
        done = self.is_terminal(next_state)

        return StepResult(
            afterstate=afterstate,
            next_state=next_state,
            reward=reward,
            done=done,
            chance_outcome=chance,
            info=info,
        )

    @property
    def is_two_player(self) -> bool:
        """Whether this is a two-player alternating game (requires value negation)."""
        return False

    def current_player(self, state: State) -> int:
        """Return current player (0 or 1). Override for two-player games."""
        return 0

    def get_canonical_state(self, state: State) -> State:
        """
        Get canonical form of state (for symmetry handling).

        Override this for games with symmetries (e.g., rotations in Go).
        Default implementation returns the state unchanged.
        """
        return state
