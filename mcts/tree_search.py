"""Stochastic MCTS with macro-operator support.

This implements Monte Carlo Tree Search for Stochastic MuZero with:
1. Alternating decision and chance nodes
2. Entropy-based chance node expansion (enumerate vs sample)
3. Macro-operator lookup and usage during planning
4. Proper value backup through stochastic branches
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .node import Node, MinMaxStats
from .macro_cache import MacroCache, MacroOperator


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""

    num_simulations: int = 50
    discount: float = 0.997
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25

    # Chance node handling
    entropy_threshold: float = 0.5
    top_k_chances: int = 5

    # Macro support
    use_macros: bool = True
    macro_verification_threshold: float = 0.1


class StochasticMCTS:
    """
    Monte Carlo Tree Search for Stochastic MuZero.

    Handles:
    - Decision nodes: Agent chooses action
    - Chance nodes: Environment samples outcome
    - Macro-operators: Skip deterministic segments
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: MCTSConfig,
        macro_cache: Optional[MacroCache] = None,
    ):
        self.model = model
        self.config = config
        self.macro_cache = macro_cache
        self.min_max_stats = MinMaxStats()

    @torch.no_grad()
    def search(
        self,
        observation: torch.Tensor,
        legal_actions: List[int],
        add_exploration_noise: bool = True,
    ) -> Node:
        """
        Run MCTS from the given observation.

        Args:
            observation: Root observation (batch_size=1, observation_dim)
            legal_actions: List of legal actions at root
            add_exploration_noise: Whether to add Dirichlet noise at root

        Returns:
            Root node after search
        """
        # Ensure batch dimension
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Initial inference at root
        initial = self.model.initial_inference(observation)

        # Create root node
        root = Node()
        root.expand(
            hidden_state=initial.state,
            policy_logits=initial.policy_logits.squeeze(0),
            legal_actions=legal_actions,
            is_chance_node=False,  # Root is decision node
        )

        # Add exploration noise
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree until leaf
            while node.expanded:
                # Check for applicable macros
                if (
                    self.config.use_macros
                    and self.macro_cache is not None
                    and not node.is_chance_node
                ):
                    macro = self._try_macro(node, search_path)
                    if macro is not None:
                        node = macro
                        continue

                # Normal selection
                action, child = node.select_child(
                    pb_c_base=self.config.pb_c_base,
                    pb_c_init=self.config.pb_c_init,
                    discount=self.config.discount,
                    min_max_stats=self.min_max_stats,
                )
                search_path.append(child)
                node = child

            # Expansion
            parent = search_path[-2] if len(search_path) > 1 else None
            value = self._expand(node, parent)

            # Backpropagation
            self._backpropagate(search_path, value)

        return root

    def _try_macro(
        self,
        node: Node,
        search_path: List[Node],
    ) -> Optional[Node]:
        """
        Try to use a macro-operator from this node.

        Returns the node reached after applying the macro,
        or None if no applicable macro was found.
        """
        if node.hidden_state is None:
            return None

        # Get legal actions (actions with children)
        legal_actions = list(node.children.keys())

        # Find applicable macros
        macros = self.macro_cache.get_applicable_macros(
            state=node.hidden_state.squeeze(0),
            legal_actions=legal_actions,
        )

        if not macros:
            return None

        # Try the highest-confidence macro
        macro = macros[0]

        # Verify macro is still valid (low entropy)
        current_state = node.hidden_state
        total_reward = 0.0
        max_entropy = 0.0

        for action in macro.action_sequence:
            # Ensure action is still legal
            if action not in node.children:
                self.macro_cache.update_macro(macro.id, success=False, entropy=1.0)
                return None

            action_tensor = torch.tensor([action], device=current_state.device)

            # Get dynamics
            dynamics_out = self.model.recurrent_inference(
                current_state, action_tensor
            )

            max_entropy = max(max_entropy, dynamics_out.chance_entropy.item())

            # Check if still deterministic
            if max_entropy > self.config.macro_verification_threshold:
                self.macro_cache.update_macro(macro.id, success=False, entropy=max_entropy)
                return None

            # Most likely chance outcome
            chance = torch.argmax(dynamics_out.chance_logits, dim=-1)
            next_state, reward_logits = self.model.dynamics(
                dynamics_out.afterstate, chance
            )

            # Get scalar reward
            reward_probs = F.softmax(reward_logits, dim=-1)
            reward = self._support_to_scalar(reward_probs).item()
            total_reward += reward * (self.config.discount ** len(search_path))

            current_state = next_state

            # Update search path through macro
            # Create virtual nodes for the path
            virtual_node = Node(
                hidden_state=dynamics_out.afterstate,
                is_chance_node=True,
                reward=reward,
                parent=search_path[-1],
                action_from_parent=action,
                macro_id=macro.id,
            )
            search_path.append(virtual_node)

        # Success - update macro statistics
        self.macro_cache.update_macro(macro.id, success=True, entropy=max_entropy)

        # Create final node after macro
        final_node = Node(
            hidden_state=current_state,
            is_chance_node=False,
            reward=total_reward,
            parent=search_path[-1],
            macro_id=macro.id,
            macro_confidence=macro.confidence,
        )
        search_path.append(final_node)

        return final_node

    def _expand(self, node: Node, parent: Optional[Node]) -> float:
        """
        Expand a leaf node and return its value.

        Args:
            node: Leaf node to expand
            parent: Parent node (needed for dynamics)

        Returns:
            Value estimate for backpropagation
        """
        if parent is None:
            # Root node already expanded in search()
            return 0.0

        action = node.action_from_parent
        parent_state = parent.hidden_state

        if node.is_chance_node:
            # This is a chance node (after action, before environment response)
            # Expand with chance outcomes

            action_tensor = torch.tensor([action], device=parent_state.device)
            dynamics_out = self.model.recurrent_inference(parent_state, action_tensor)

            # Expand chance node
            chance_indices, enumerated = node.expand_chance(
                hidden_state=dynamics_out.afterstate,
                chance_logits=dynamics_out.chance_logits.squeeze(0),
                top_k=self.config.top_k_chances,
                entropy_threshold=self.config.entropy_threshold,
            )

            # Get afterstate value
            _, value_logits = self.model.predict_afterstate(dynamics_out.afterstate)
            value_probs = F.softmax(value_logits, dim=-1)
            value = self._support_to_scalar(value_probs).item()

            return value
        else:
            # This is a decision node (after chance resolved)
            # Need to compute state from parent's afterstate + chance

            # Get chance outcome that led here
            chance = node.action_from_parent
            chance_tensor = torch.tensor([chance], device=parent_state.device)

            # Parent is afterstate, compute next state
            next_state, reward_logits = self.model.dynamics(parent_state, chance_tensor)

            # Get reward
            reward_probs = F.softmax(reward_logits, dim=-1)
            reward = self._support_to_scalar(reward_probs).item()
            node.reward = reward

            # Get policy and value at next state
            policy_logits, value_logits = self.model.predict_state(next_state)

            # Expand with all actions (no legal action filtering in latent space)
            # In practice, you'd track legal actions or use the policy network
            node.expand(
                hidden_state=next_state,
                policy_logits=policy_logits.squeeze(0),
                legal_actions=list(range(policy_logits.shape[-1])),
                is_chance_node=True,  # Children will be chance nodes
            )

            value_probs = F.softmax(value_logits, dim=-1)
            value = self._support_to_scalar(value_probs).item()

            return value

    def _backpropagate(self, search_path: List[Node], value: float) -> None:
        """
        Backpropagate value through the search path.

        Args:
            search_path: Path from root to leaf
            value: Value at leaf node
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value

            # Update min-max stats
            self.min_max_stats.update(node.reward + self.config.discount * value)

            # Discount value for next level up
            value = node.reward + self.config.discount * value

    def _support_to_scalar(self, probs: torch.Tensor) -> torch.Tensor:
        """Convert categorical support to scalar value."""
        support_size = (probs.shape[-1] - 1) // 2
        support = torch.arange(
            -support_size, support_size + 1,
            device=probs.device, dtype=probs.dtype
        )
        expected = (probs * support).sum(dim=-1)

        # Inverse transformation
        eps = 0.001
        sign = torch.sign(expected)
        abs_expected = torch.abs(expected)
        return sign * ((abs_expected + 1).square() - 1) / (1 + 2 * eps * (abs_expected + 1))

    def get_action_policy(
        self, root: Node, temperature: float = 1.0
    ) -> Tuple[int, np.ndarray]:
        """
        Get action and policy from search results.

        Args:
            root: Root node after search
            temperature: Temperature for action selection

        Returns:
            (selected_action, policy_distribution)
        """
        actions, probs = root.get_policy()

        # Create full policy array
        policy = np.zeros(len(root.children))
        for action, prob in zip(actions, probs):
            policy[action] = prob

        # Select action
        selected = root.select_action(temperature=temperature)

        return selected, policy


def run_mcts(
    model: torch.nn.Module,
    observation: torch.Tensor,
    legal_actions: List[int],
    config: Optional[MCTSConfig] = None,
    macro_cache: Optional[MacroCache] = None,
    add_noise: bool = True,
) -> Tuple[int, np.ndarray, float, Node]:
    """
    Convenience function to run MCTS and get results.

    Args:
        model: MuZero network
        observation: Current observation
        legal_actions: Legal actions
        config: MCTS configuration (uses defaults if None)
        macro_cache: Optional macro cache
        add_noise: Whether to add exploration noise

    Returns:
        (action, policy, root_value, root_node)
    """
    if config is None:
        config = MCTSConfig()

    mcts = StochasticMCTS(model, config, macro_cache)
    root = mcts.search(observation, legal_actions, add_exploration_noise=add_noise)

    action, policy = mcts.get_action_policy(root, temperature=1.0)
    root_value = root.value

    return action, policy, root_value, root
