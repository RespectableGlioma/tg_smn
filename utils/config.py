"""Configuration management for Stochastic MuZero."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
from pathlib import Path


@dataclass
class Config:
    """Configuration for Stochastic MuZero with macro-operator discovery."""

    # Game settings
    game: str = "2048"
    action_space_size: int = 4
    chance_space_size: int = 33  # 2048: 16 positions * 2 values + 1 (no spawn)

    # Network architecture
    state_dim: int = 256
    hidden_dim: int = 128
    num_layers: int = 2
    observation_dim: int = 496  # 2048: 31 bits * 16 tiles

    # Support for scalar predictions (value/reward)
    support_size: int = 31  # Support from -support_size to +support_size

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    num_unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.997

    # Self-play
    num_simulations: int = 50
    num_actors: int = 4
    max_moves: int = 10000
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_decay_steps: int = 10000

    # Replay buffer
    replay_buffer_size: int = 100000
    priority_alpha: float = 1.0
    priority_beta: float = 1.0

    # MCTS
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25

    # Macro-operator discovery (key innovation)
    entropy_threshold: float = 0.1  # Below this, transition is "deterministic"
    composition_threshold: float = 0.01  # Max error for macro validity
    min_macro_length: int = 2
    max_macro_length: int = 8
    macro_confidence_decay: float = 0.9
    macro_confidence_boost: float = 1.05
    max_macros: int = 1000

    # Chance node handling
    chance_entropy_threshold: float = 0.5  # Below: enumerate, above: sample
    top_k_chances: int = 5

    # Logging
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500

    # Device
    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration."""
        assert self.state_dim > 0
        assert self.hidden_dim > 0
        assert self.num_simulations > 0
        assert 0 <= self.entropy_threshold <= 1
        assert self.min_macro_length >= 2
        assert self.max_macro_length >= self.min_macro_length

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


def load_config(config_path: Optional[str] = None, **overrides) -> Config:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.
        **overrides: Key-value pairs to override config values.

    Returns:
        Config object with loaded and overridden values.
    """
    config_dict = {}

    # Load from file if provided
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f) or {}

    # Apply overrides
    config_dict.update(overrides)

    return Config.from_dict(config_dict)


def save_config(config: Config, path: str) -> None:
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
