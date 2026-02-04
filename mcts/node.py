"""MCTS tree node for Stochastic MuZero.

The tree alternates between two node types:
1. Decision nodes: Agent chooses an action
2. Chance nodes: Environment samples a chance outcome

This separation is crucial for handling stochastic transitions
and enables proper backup of values through stochastic branches.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


@dataclass
class Node:
    """
    Tree node for Stochastic MCTS.

    Attributes:
        hidden_state: Latent representation (state or afterstate)
        is_chance_node: True if this is a chance node (env's turn)
        prior: Prior probability from policy network
        visit_count: Number of times this node was visited
        value_sum: Sum of backed-up values
        reward: Immediate reward leading to this node
        children: Dict mapping action/chance → child node
        parent: Parent node (None for root)
        action_from_parent: Action that led to this node

        # Macro-related fields
        macro_id: ID of macro that led here (if any)
        macro_confidence: Confidence of the macro path
    """

    hidden_state: Optional[torch.Tensor] = None
    is_chance_node: bool = False
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)
    parent: Optional["Node"] = None
    action_from_parent: Optional[int] = None

    # Macro-related
    macro_id: Optional[int] = None
    macro_confidence: float = 1.0

    # Chance-specific (for chance nodes)
    chance_entropy: float = 0.0

    # Two-player support (-1 = chance node / unknown, 0/1 = player)
    to_play: int = -1

    @property
    def expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0

    @property
    def value(self) -> float:
        """Get mean value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(
        self,
        hidden_state: torch.Tensor,
        policy_logits: torch.Tensor,
        legal_actions: List[int],
        is_chance_node: bool = False,
    ) -> None:
        """
        Expand this node with children for each action.

        Args:
            hidden_state: Latent state to store
            policy_logits: Policy logits over actions
            legal_actions: List of legal action indices
            is_chance_node: Whether children are chance nodes
        """
        self.hidden_state = hidden_state

        # Convert logits to probabilities for legal actions only
        policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()

        # Mask illegal actions and renormalize
        legal_mask = np.zeros(len(policy))
        legal_mask[legal_actions] = 1.0
        policy = policy * legal_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Uniform over legal actions if all masked
            policy[legal_actions] = 1.0 / len(legal_actions)

        # Create child nodes
        for action in legal_actions:
            child = Node(
                is_chance_node=is_chance_node,
                prior=float(policy[action]),
                parent=self,
                action_from_parent=action,
            )
            self.children[action] = child

    def expand_chance(
        self,
        hidden_state: torch.Tensor,
        chance_logits: torch.Tensor,
        top_k: int = 5,
        entropy_threshold: float = 0.5,
    ) -> Tuple[List[int], bool]:
        """
        Expand chance node with outcomes.

        For low-entropy nodes: enumerate top-k outcomes.
        For high-entropy nodes: just sample one outcome.

        Args:
            hidden_state: Afterstate representation
            chance_logits: Logits over chance outcomes
            top_k: Number of outcomes to enumerate for low-entropy
            entropy_threshold: Threshold for enumeration vs sampling

        Returns:
            chance_indices: List of expanded chance outcomes
            enumerated: Whether we enumerated (True) or sampled (False)
        """
        self.hidden_state = hidden_state

        probs = torch.softmax(chance_logits, dim=-1).cpu().numpy()

        # Compute entropy
        log_probs = np.log(probs + 1e-10)
        entropy = -np.sum(probs * log_probs)
        self.chance_entropy = entropy

        if entropy < entropy_threshold:
            # Low entropy: enumerate top-k outcomes
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]

            # Renormalize
            top_probs = top_probs / top_probs.sum()

            for idx, prob in zip(top_indices, top_probs):
                child = Node(
                    is_chance_node=False,
                    prior=float(prob),
                    parent=self,
                    action_from_parent=int(idx),
                )
                self.children[int(idx)] = child

            return list(top_indices), True
        else:
            # High entropy: sample one outcome
            sampled = np.random.choice(len(probs), p=probs)
            child = Node(
                is_chance_node=False,
                prior=1.0,
                parent=self,
                action_from_parent=int(sampled),
            )
            self.children[int(sampled)] = child

            return [int(sampled)], False

    def add_exploration_noise(
        self,
        dirichlet_alpha: float,
        exploration_fraction: float,
    ) -> None:
        """
        Add Dirichlet noise to priors for exploration (at root).

        Args:
            dirichlet_alpha: Dirichlet concentration parameter
            exploration_fraction: Weight of noise vs prior
        """
        if not self.children:
            return

        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))

        for i, action in enumerate(actions):
            child = self.children[action]
            child.prior = (
                child.prior * (1 - exploration_fraction)
                + noise[i] * exploration_fraction
            )

    def select_child(
        self,
        pb_c_base: float,
        pb_c_init: float,
        discount: float,
        min_max_stats: "MinMaxStats",
    ) -> Tuple[int, "Node"]:
        """
        Select best child according to PUCT formula.

        UCB score = Q(s,a) + C(s) * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        where:
        - Q(s,a) is the normalized action value
        - P(s,a) is the prior probability
        - N(s) is parent visit count
        - N(s,a) is child visit count
        - C(s) = log((N(s) + pb_c_base + 1) / pb_c_base) + pb_c_init

        Args:
            pb_c_base: Base for exploration coefficient
            pb_c_init: Initial exploration coefficient
            discount: Discount factor for value backup
            min_max_stats: Min-max normalization statistics

        Returns:
            (best_action, best_child)
        """
        best_score = float("-inf")
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = self._ucb_score(
                child, pb_c_base, pb_c_init, discount, min_max_stats
            )
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _ucb_score(
        self,
        child: "Node",
        pb_c_base: float,
        pb_c_init: float,
        discount: float,
        min_max_stats: "MinMaxStats",
    ) -> float:
        """Compute UCB score for a child."""
        # Exploration coefficient
        pb_c = (
            np.log((self.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        )

        # Prior score (exploration)
        prior_score = (
            pb_c * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
        )

        # Value score (exploitation)
        if child.visit_count > 0:
            # Normalize value to [0, 1]
            value = child.reward + discount * child.value
            value_score = min_max_stats.normalize(value)
        else:
            value_score = 0.0

        return prior_score + value_score

    def select_action(self, temperature: float = 1.0) -> int:
        """
        Select action based on visit counts.

        Args:
            temperature: Controls exploration (0 = greedy, higher = more random)

        Returns:
            Selected action index
        """
        actions = list(self.children.keys())
        visit_counts = np.array([self.children[a].visit_count for a in actions], dtype=np.float64)

        if visit_counts.sum() == 0:
            # No visits — uniform random over children
            return int(np.random.choice(actions))

        if temperature == 0:
            # Greedy selection
            return actions[np.argmax(visit_counts)]

        # Temperature-based selection
        counts_temp = visit_counts ** (1.0 / temperature)
        total = counts_temp.sum()
        if total == 0:
            return int(np.random.choice(actions))
        probs = counts_temp / total
        return int(np.random.choice(actions, p=probs))

    def get_policy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get policy distribution from visit counts.

        Returns:
            (actions, probabilities)
        """
        actions = np.array(list(self.children.keys()))
        visit_counts = np.array([self.children[a].visit_count for a in actions], dtype=np.float64)
        total = visit_counts.sum()
        if total == 0:
            # No visits — uniform distribution
            probs = np.ones(len(actions)) / len(actions)
        else:
            probs = visit_counts / total
        return actions, probs


class MinMaxStats:
    """
    Track min and max values for normalization.

    Used to normalize Q-values to [0, 1] range for UCB calculation.
    """

    def __init__(self):
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, value: float) -> None:
        """Update statistics with new value."""
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1]."""
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
