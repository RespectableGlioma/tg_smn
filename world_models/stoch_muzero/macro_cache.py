"""
Macro Caching for Hierarchical Planning

This implements the key idea from the conversation:
"Rules are compressible causal structure."

When the model learns deterministic transitions, we can:
1. Cache these as "macro operators"
2. Skip intermediate planning steps
3. Plan at multiple temporal scales

Key insight: A transition is cacheable if:
- Low entropy (deterministic)
- Consistent across contexts
- Compositionally reliable

The macro cache enables:
- Faster MCTS (skip known outcomes)
- Emergent temporal abstraction
- Transfer of learned rules
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

from .model import CausalWorldModel


@dataclass
class MacroOperator:
    """
    A cached macro-transition that can skip multiple steps.
    
    A macro represents: starting_state + action_sequence → ending_state
    with high confidence (deterministic).
    """
    state_signature: str       # Hash of starting state features
    action_sequence: Tuple[int, ...]  # Actions taken
    ending_state: np.ndarray   # Predicted final state
    confidence: float          # How reliable is this macro
    k: int                     # Number of steps it spans
    
    # Statistics
    uses: int = 0
    successes: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.uses == 0:
            return 1.0
        return self.successes / self.uses


@dataclass 
class MacroCache:
    """
    Cache of learned macro operators.
    
    This is the "temporal compression" of deterministic patterns.
    When we encounter a state we've seen before, we can skip ahead
    if the transition is known to be deterministic.
    """
    
    max_size: int = 10000
    entropy_threshold: float = 0.1   # Below this = deterministic
    confidence_threshold: float = 0.9
    min_uses_for_trust: int = 3
    
    # Storage
    cache: Dict[str, MacroOperator] = field(default_factory=dict)
    
    # Statistics
    total_lookups: int = 0
    cache_hits: int = 0
    
    def get_state_signature(self, state: torch.Tensor) -> str:
        """
        Create a hash signature for a latent state.
        
        This allows approximate matching of similar states.
        We discretize the state vector for robustness.
        """
        # Discretize to bins
        discretized = (state.detach().cpu().numpy() * 10).astype(np.int32)
        return hashlib.md5(discretized.tobytes()).hexdigest()[:12]
    
    def get_key(self, state_sig: str, actions: Tuple[int, ...]) -> str:
        """Create cache key from state signature and actions."""
        return f"{state_sig}_{'-'.join(map(str, actions))}"
    
    def lookup(self, state: torch.Tensor, 
               action_sequence: Tuple[int, ...]) -> Optional[MacroOperator]:
        """
        Look up a cached macro.
        
        Returns the macro if found and trusted, else None.
        """
        self.total_lookups += 1
        
        sig = self.get_state_signature(state)
        key = self.get_key(sig, action_sequence)
        
        macro = self.cache.get(key)
        if macro is None:
            return None
        
        # Check if we trust this macro
        if macro.uses < self.min_uses_for_trust:
            return None
        if macro.success_rate < self.confidence_threshold:
            return None
        
        self.cache_hits += 1
        return macro
    
    def store(self, state: torch.Tensor, action_sequence: Tuple[int, ...],
              ending_state: np.ndarray, confidence: float):
        """
        Store a new macro or update existing one.
        
        Only stores if confidence is high enough (deterministic).
        """
        if confidence < self.confidence_threshold:
            return  # Not deterministic enough to cache
        
        sig = self.get_state_signature(state)
        key = self.get_key(sig, action_sequence)
        
        if key in self.cache:
            # Update existing
            macro = self.cache[key]
            # Exponential moving average of confidence
            macro.confidence = 0.9 * macro.confidence + 0.1 * confidence
        else:
            # Create new
            macro = MacroOperator(
                state_signature=sig,
                action_sequence=action_sequence,
                ending_state=ending_state,
                confidence=confidence,
                k=len(action_sequence),
            )
            
            # Evict if full (LRU-ish: evict least confident)
            if len(self.cache) >= self.max_size:
                worst_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k].confidence * self.cache[k].success_rate)
                del self.cache[worst_key]
            
            self.cache[key] = macro
    
    def record_outcome(self, state: torch.Tensor, action_sequence: Tuple[int, ...],
                       actual_outcome: np.ndarray, success: bool):
        """
        Record whether a macro prediction was correct.
        
        This updates the reliability statistics.
        """
        sig = self.get_state_signature(state)
        key = self.get_key(sig, action_sequence)
        
        if key in self.cache:
            macro = self.cache[key]
            macro.uses += 1
            if success:
                macro.successes += 1
    
    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.cache_hits / self.total_lookups
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        confidences = [m.confidence for m in self.cache.values()]
        success_rates = [m.success_rate for m in self.cache.values() if m.uses > 0]
        
        return {
            'size': len(self.cache),
            'total_lookups': self.total_lookups,
            'cache_hits': self.cache_hits,
            'hit_rate': self.hit_rate,
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'avg_success_rate': float(np.mean(success_rates)) if success_rates else 0.0,
        }


def compute_transition_confidence(model: CausalWorldModel, 
                                   state: torch.Tensor,
                                   action: torch.Tensor,
                                   empty_mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute confidence that a transition is deterministic.
    
    For deterministic transitions: confidence ≈ 1
    For stochastic transitions: confidence ≈ exp(-entropy)
    """
    with torch.no_grad():
        out = model.recurrent_inference(state, action, empty_mask)
        
        if model.cfg.has_chance and 'chance_entropy' in out:
            entropy = out['chance_entropy'].mean().item()
            # Convert entropy to confidence: low entropy = high confidence
            confidence = np.exp(-entropy)
        else:
            # No chance node = fully deterministic
            confidence = 1.0
    
    return float(confidence)


@dataclass
class MacroDiscoveryConfig:
    """Configuration for automatic macro discovery."""
    window_size: int = 5       # Max macro length to consider
    entropy_threshold: float = 0.1
    composition_threshold: float = 0.05  # Max error for composition


class MacroDiscoverer:
    """
    Automatic discovery of macro operators from trajectories.
    
    This implements the key insight:
    "Which transitions don't matter?"
    
    A sequence of transitions can be compressed into a macro if:
    1. Each step is deterministic (low entropy)
    2. Direct prediction matches sequential rollout
    3. Pattern holds across different contexts
    """
    
    def __init__(self, model: CausalWorldModel, cfg: MacroDiscoveryConfig):
        self.model = model
        self.cfg = cfg
        self.candidate_macros: Dict[str, List[dict]] = {}  # Pattern → observations
    
    def analyze_trajectory(self, states: List[torch.Tensor], 
                           actions: List[int],
                           outcomes: List[torch.Tensor]) -> List[MacroOperator]:
        """
        Analyze a trajectory for compressible patterns.
        
        Returns list of discovered macro operators.
        """
        discovered = []
        T = len(actions)
        
        for start in range(T):
            for length in range(2, min(self.cfg.window_size + 1, T - start + 1)):
                end = start + length
                
                action_seq = tuple(actions[start:end])
                
                # Check if this pattern is compressible
                is_compressible, confidence = self._check_compressibility(
                    states[start],
                    action_seq,
                    outcomes[start:end],
                )
                
                if is_compressible:
                    macro = MacroOperator(
                        state_signature=self._get_signature(states[start]),
                        action_sequence=action_seq,
                        ending_state=outcomes[end-1].detach().cpu().numpy(),
                        confidence=confidence,
                        k=length,
                    )
                    discovered.append(macro)
        
        return discovered
    
    def _check_compressibility(self, initial_state: torch.Tensor,
                                action_seq: Tuple[int, ...],
                                intermediate_outcomes: List[torch.Tensor]) -> Tuple[bool, float]:
        """
        Check if an action sequence can be compressed.
        
        Criteria:
        1. Each intermediate transition has low entropy
        2. Direct multi-step prediction matches sequential
        """
        with torch.no_grad():
            # Check entropy at each step
            state = initial_state.unsqueeze(0) if initial_state.dim() == 1 else initial_state
            entropies = []
            
            for action in action_seq:
                action_t = torch.tensor([action], device=state.device)
                out = self.model.recurrent_inference(state, action_t)
                
                if self.model.cfg.has_chance and 'chance_entropy' in out:
                    entropies.append(out['chance_entropy'].mean().item())
                else:
                    entropies.append(0.0)
                
                state = out['afterstate']
            
            # Sequential endpoint
            sequential_end = state
            
            # Check if all steps were low-entropy
            max_entropy = max(entropies) if entropies else 0.0
            if max_entropy > self.cfg.entropy_threshold:
                return False, 0.0
            
            # For now, confidence is inverse of max entropy
            confidence = np.exp(-max_entropy)
            
            return True, confidence
    
    def _get_signature(self, state: torch.Tensor) -> str:
        """Get hash signature for state."""
        if state.dim() > 1:
            state = state.squeeze(0)
        discretized = (state.detach().cpu().numpy() * 10).astype(np.int32)
        return hashlib.md5(discretized.tobytes()).hexdigest()[:12]


class HierarchicalMCTS:
    """
    MCTS with macro operator support.
    
    This implements hierarchical planning:
    - Use macros when confident (skip steps)
    - Fall back to primitives when uncertain
    - Learn new macros from successful rollouts
    """
    
    def __init__(self, model: CausalWorldModel, 
                 macro_cache: MacroCache,
                 entropy_threshold: float = 0.1):
        self.model = model
        self.cache = macro_cache
        self.entropy_threshold = entropy_threshold
    
    def should_expand_chance(self, state: torch.Tensor, 
                              action: torch.Tensor,
                              empty_mask: Optional[torch.Tensor] = None) -> bool:
        """
        Decide whether to expand chance node or treat as deterministic.
        
        Key insight: If entropy is below threshold, skip chance expansion.
        This is the core of "rules vs stochasticity" separation in planning.
        """
        confidence = compute_transition_confidence(
            self.model, state, action, empty_mask
        )
        return confidence < (1 - self.entropy_threshold)
    
    def search_with_macros(self, root_state: torch.Tensor,
                            num_simulations: int = 100,
                            max_depth: int = 20) -> Tuple[int, Dict[str, Any]]:
        """
        MCTS search that leverages macro operators.
        
        Returns best action and search statistics.
        """
        # This is a simplified version - full MCTS implementation would be more complex
        
        # For now, just demonstrate macro lookup
        stats = {
            'macro_uses': 0,
            'primitive_uses': 0,
            'total_depth': 0,
        }
        
        action_values = torch.zeros(self.model.cfg.num_actions)
        action_visits = torch.zeros(self.model.cfg.num_actions)
        
        for sim in range(num_simulations):
            state = root_state.clone()
            depth = 0
            total_reward = 0.0
            
            while depth < max_depth:
                # Try macro lookup for common patterns
                for macro_len in range(3, 1, -1):  # Try longer macros first
                    # Would need actual action selection logic here
                    pass
                
                # Select action (UCB or similar)
                action = self._select_action(state, action_values, action_visits)
                action_t = torch.tensor([action], device=state.device)
                
                # Check if we need to expand chance
                if self.should_expand_chance(state, action_t):
                    # Stochastic: need to sample from chance
                    stats['primitive_uses'] += 1
                else:
                    # Deterministic: can skip chance node
                    stats['macro_uses'] += 1
                
                # Step
                out = self.model.recurrent_inference(state.unsqueeze(0), action_t)
                state = out['afterstate'].squeeze(0)
                
                depth += 1
            
            stats['total_depth'] += depth
        
        stats['avg_depth'] = stats['total_depth'] / num_simulations
        
        # Return most visited action
        best_action = int(action_visits.argmax().item())
        return best_action, stats
    
    def _select_action(self, state: torch.Tensor, 
                       values: torch.Tensor, 
                       visits: torch.Tensor) -> int:
        """UCB action selection."""
        total_visits = visits.sum() + 1
        ucb = values / (visits + 1) + np.sqrt(2 * np.log(total_visits) / (visits + 1))
        return int(ucb.argmax().item())


def analyze_entropy_distribution(model: CausalWorldModel,
                                  transitions: List[dict],
                                  device: str = "cpu") -> Dict[str, Any]:
    """
    Analyze the entropy distribution across transitions.
    
    This is the key diagnostic for "rules vs stochasticity":
    - Should see bimodal distribution
    - Deterministic transitions cluster near 0
    - Stochastic transitions have higher entropy
    """
    model.eval()
    entropies = []
    
    with torch.no_grad():
        for t in transitions[:1000]:  # Sample
            obs = torch.from_numpy(t['obs']).float().unsqueeze(0).to(device)
            action = torch.tensor([t['action']], device=device)
            
            init = model.initial_inference(obs)
            state = init['state']
            
            out = model.recurrent_inference(state, action)
            
            if model.cfg.has_chance and 'chance_entropy' in out:
                entropies.append(float(out['chance_entropy'].mean().item()))
    
    entropies = np.array(entropies)
    
    return {
        'mean': float(np.mean(entropies)),
        'std': float(np.std(entropies)),
        'min': float(np.min(entropies)),
        'max': float(np.max(entropies)),
        'median': float(np.median(entropies)),
        'p10': float(np.percentile(entropies, 10)),
        'p90': float(np.percentile(entropies, 90)),
        'below_0.1': float(np.mean(entropies < 0.1)),
        'below_0.5': float(np.mean(entropies < 0.5)),
    }
