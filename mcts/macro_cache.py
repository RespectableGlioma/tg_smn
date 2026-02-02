"""Macro-operator cache for learned temporal abstractions.

This is the core innovation: discovering and caching compressible
causal structure as reusable macro-operators.

A macro-operator is a sequence of actions that:
1. Has low transition entropy (deterministic)
2. Can be predicted directly (composition error is low)
3. Generalizes across states (context invariant)

When these criteria are met, we can skip planning through
intermediate states and jump directly to the outcome.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict


@dataclass
class MacroOperator:
    """
    A learned temporal abstraction (macro-operator).

    Represents a compressible action sequence that can be
    treated as a single atomic transition for planning.
    """

    id: int
    action_sequence: Tuple[int, ...]  # Actions comprising this macro
    length: int  # Number of steps (k)

    # State signature for applicability checking
    precondition_features: Optional[torch.Tensor] = None

    # Statistics
    confidence: float = 1.0  # How reliable is this macro
    usage_count: int = 0  # Times used in planning
    success_count: int = 0  # Times prediction was accurate
    creation_step: int = 0  # Training step when discovered

    # Entropy tracking
    entropy_history: List[float] = field(default_factory=list)
    max_entropy_seen: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of successful predictions."""
        if self.usage_count == 0:
            return 1.0
        return self.success_count / self.usage_count

    def record_usage(self, success: bool, entropy: float) -> None:
        """Record usage outcome."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.entropy_history.append(entropy)
        self.max_entropy_seen = max(self.max_entropy_seen, entropy)


class SpatialHash:
    """
    Simple spatial hash for fast nearest-neighbor lookup.

    Used to find macros that might apply to a given state
    based on state similarity.
    """

    def __init__(self, num_buckets: int = 1024, feature_dim: int = 256):
        self.num_buckets = num_buckets
        self.feature_dim = feature_dim
        self.buckets: Dict[int, List[int]] = defaultdict(list)

        # Random projection for hashing
        self.projection = torch.randn(feature_dim, num_buckets)

    def _hash(self, features: torch.Tensor) -> int:
        """Compute hash bucket for features."""
        with torch.no_grad():
            proj = features @ self.projection
            bucket = int(proj.argmax().item())
        return bucket

    def insert(self, features: torch.Tensor, macro_id: int) -> None:
        """Insert macro into hash table."""
        bucket = self._hash(features)
        if macro_id not in self.buckets[bucket]:
            self.buckets[bucket].append(macro_id)

    def query(self, features: torch.Tensor) -> List[int]:
        """Find candidate macros for given features."""
        bucket = self._hash(features)
        return self.buckets[bucket].copy()


class MacroCache:
    """
    Cache for discovered macro-operators.

    Handles:
    - Macro discovery from trajectories
    - Macro lookup for planning
    - Macro statistics and confidence tracking
    - Pruning unreliable macros
    """

    def __init__(
        self,
        state_dim: int = 256,
        entropy_threshold: float = 0.1,
        composition_threshold: float = 0.01,
        min_macro_length: int = 2,
        max_macro_length: int = 8,
        max_macros: int = 1000,
        confidence_decay: float = 0.9,
        confidence_boost: float = 1.05,
        min_confidence: float = 0.5,
    ):
        self.state_dim = state_dim
        self.entropy_threshold = entropy_threshold
        self.composition_threshold = composition_threshold
        self.min_macro_length = min_macro_length
        self.max_macro_length = max_macro_length
        self.max_macros = max_macros
        self.confidence_decay = confidence_decay
        self.confidence_boost = confidence_boost
        self.min_confidence = min_confidence

        # Storage
        self.macros: Dict[int, MacroOperator] = {}
        self.action_index: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        self.spatial_index = SpatialHash(feature_dim=state_dim)

        # Statistics
        self.total_discoveries = 0
        self.total_uses = 0
        self.total_successes = 0

        # Next macro ID
        self._next_id = 0

    def discover_macro(
        self,
        states: List[torch.Tensor],
        actions: List[int],
        entropies: List[float],
        training_step: int = 0,
        model: Optional[torch.nn.Module] = None,
    ) -> Optional[MacroOperator]:
        """
        Try to discover a macro from a trajectory segment.

        Criteria for macro discovery:
        1. All transitions have low entropy (< threshold)
        2. Composition is accurate (optional, if model provided)
        3. Sequence is within length bounds

        Args:
            states: List of k+1 states [s_0, s_1, ..., s_k]
            actions: List of k actions [a_0, a_1, ..., a_{k-1}]
            entropies: List of k entropies [H_0, H_1, ..., H_{k-1}]
            training_step: Current training step
            model: Optional model for composition check

        Returns:
            MacroOperator if discovered, None otherwise
        """
        k = len(actions)

        # Check length bounds
        if k < self.min_macro_length or k > self.max_macro_length:
            return None

        # Check entropy criterion (all transitions must be low-entropy)
        max_entropy = max(entropies)
        if max_entropy > self.entropy_threshold:
            return None

        # Check if this action sequence already exists
        action_tuple = tuple(actions)
        if action_tuple in self.action_index:
            # Update existing macros with new evidence
            for macro_id in self.action_index[action_tuple]:
                macro = self.macros[macro_id]
                macro.entropy_history.append(max_entropy)
            return None

        # Optional: composition check
        if model is not None:
            composition_error = self._check_composition(
                model, states[0], actions, states[-1]
            )
            if composition_error > self.composition_threshold:
                return None

        # Create new macro
        macro = MacroOperator(
            id=self._next_id,
            action_sequence=action_tuple,
            length=k,
            precondition_features=states[0].detach().clone(),
            confidence=1.0,
            creation_step=training_step,
            entropy_history=[max_entropy],
            max_entropy_seen=max_entropy,
        )

        self._next_id += 1
        self.total_discoveries += 1

        # Store macro
        self.macros[macro.id] = macro
        self.action_index[action_tuple].append(macro.id)
        self.spatial_index.insert(states[0], macro.id)

        # Prune if over capacity
        if len(self.macros) > self.max_macros:
            self._prune_worst_macro()

        return macro

    def _check_composition(
        self,
        model: torch.nn.Module,
        initial_state: torch.Tensor,
        actions: List[int],
        final_state: torch.Tensor,
    ) -> float:
        """
        Check if direct prediction matches step-by-step rollout.

        Returns MSE between predicted and actual final state.
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if initial_state.dim() == 1:
                initial_state = initial_state.unsqueeze(0)
                final_state = final_state.unsqueeze(0)

            # Step-by-step rollout
            current_state = initial_state
            for action in actions:
                action_tensor = torch.tensor([action], device=current_state.device)
                # Sample chance from model
                afterstate = model.afterstate_dynamics(current_state, action_tensor)
                chance_logits, _ = model.chance_encoder(afterstate)
                chance = torch.argmax(chance_logits, dim=-1)  # Take most likely
                current_state, _ = model.dynamics(afterstate, chance)

            # Compute error
            error = F.mse_loss(current_state, final_state).item()

        return error

    def get_applicable_macros(
        self,
        state: torch.Tensor,
        legal_actions: List[int],
    ) -> List[MacroOperator]:
        """
        Find macros that might apply to this state.

        Args:
            state: Current state
            legal_actions: Currently legal actions

        Returns:
            List of potentially applicable macros
        """
        # Get candidates from spatial hash
        candidate_ids = self.spatial_index.query(state)

        applicable = []
        for macro_id in candidate_ids:
            if macro_id not in self.macros:
                continue

            macro = self.macros[macro_id]

            # Check if first action is legal
            if macro.action_sequence[0] not in legal_actions:
                continue

            # Check confidence threshold
            if macro.confidence < self.min_confidence:
                continue

            applicable.append(macro)

        # Sort by confidence (highest first)
        applicable.sort(key=lambda m: m.confidence, reverse=True)

        return applicable

    def update_macro(
        self,
        macro_id: int,
        success: bool,
        entropy: float,
    ) -> None:
        """
        Update macro statistics after usage.

        Args:
            macro_id: ID of the used macro
            success: Whether prediction was accurate
            entropy: Observed entropy during usage
        """
        if macro_id not in self.macros:
            return

        macro = self.macros[macro_id]
        macro.record_usage(success, entropy)
        self.total_uses += 1

        # Update confidence
        if entropy > self.entropy_threshold:
            # Transition became stochastic - decay confidence
            macro.confidence *= self.confidence_decay
        elif success:
            # Successful prediction - boost confidence
            macro.confidence = min(1.0, macro.confidence * self.confidence_boost)
            self.total_successes += 1
        else:
            # Failed prediction - decay confidence
            macro.confidence *= self.confidence_decay

        # Remove if confidence too low
        if macro.confidence < self.min_confidence / 2:
            self._remove_macro(macro_id)

    def _remove_macro(self, macro_id: int) -> None:
        """Remove a macro from all indices."""
        if macro_id not in self.macros:
            return

        macro = self.macros[macro_id]

        # Remove from action index
        action_tuple = macro.action_sequence
        if action_tuple in self.action_index:
            self.action_index[action_tuple] = [
                mid for mid in self.action_index[action_tuple] if mid != macro_id
            ]
            if not self.action_index[action_tuple]:
                del self.action_index[action_tuple]

        # Remove from main storage
        del self.macros[macro_id]

    def _prune_worst_macro(self) -> None:
        """Remove the lowest-confidence macro."""
        if not self.macros:
            return

        worst_id = min(self.macros.keys(), key=lambda m: self.macros[m].confidence)
        self._remove_macro(worst_id)

    def get_statistics(self) -> Dict[str, float]:
        """Get cache statistics."""
        return {
            "num_macros": len(self.macros),
            "total_discoveries": self.total_discoveries,
            "total_uses": self.total_uses,
            "total_successes": self.total_successes,
            "success_rate": (
                self.total_successes / self.total_uses if self.total_uses > 0 else 0.0
            ),
            "avg_confidence": (
                np.mean([m.confidence for m in self.macros.values()])
                if self.macros
                else 0.0
            ),
            "avg_length": (
                np.mean([m.length for m in self.macros.values()])
                if self.macros
                else 0.0
            ),
        }


def discover_macros_from_trajectory(
    trajectory: List[Dict],
    macro_cache: MacroCache,
    min_length: int = 2,
    max_length: int = 8,
    training_step: int = 0,
    model: Optional[torch.nn.Module] = None,
) -> List[MacroOperator]:
    """
    Discover all potential macros from a trajectory.

    Scans the trajectory for low-entropy segments that could
    be compressed into macro-operators.

    Args:
        trajectory: List of transition dicts with keys:
            - 'state': torch.Tensor
            - 'action': int
            - 'entropy': float
        macro_cache: MacroCache to add discoveries to
        min_length: Minimum macro length
        max_length: Maximum macro length
        training_step: Current training step
        model: Optional model for composition checking

    Returns:
        List of newly discovered macros
    """
    discoveries = []
    n = len(trajectory)

    if n < min_length:
        return discoveries

    # Scan for low-entropy segments of various lengths
    for length in range(min_length, min(max_length + 1, n + 1)):
        for start in range(n - length + 1):
            segment = trajectory[start : start + length]

            # Extract states, actions, entropies
            states = [t["state"] for t in segment]
            # Add final state (next_state of last transition)
            if "next_state" in segment[-1]:
                states.append(segment[-1]["next_state"])
            else:
                continue

            actions = [t["action"] for t in segment]
            entropies = [t["entropy"] for t in segment]

            # Try to create macro
            macro = macro_cache.discover_macro(
                states=states,
                actions=actions,
                entropies=entropies,
                training_step=training_step,
                model=model,
            )

            if macro is not None:
                discoveries.append(macro)

    return discoveries
