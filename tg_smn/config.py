from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional


# =====================
# Data / training config
# =====================

@dataclass
class DataCfg:
    seq_len: int = 64
    batch_size: int = 32
    num_workers: int = 2


@dataclass
class TrainCfgLM:
    seed: int = 0
    epochs_per_task: int = 1

    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    fisher_every: int = 100
    delta_rho_samples: int = 3

    replay_max_seqs: int = 20000
    log_every: int = 20

    # If set, caps training steps per task for speed.
    max_steps_per_task: Optional[int] = None

    # Progress / logging
    # NOTE: We deliberately avoid printing time-to-completion estimates. Progress is shown
    # via counts + elapsed time only.
    show_progress: bool = True
    progress_steps: bool = False
    progress_postfix_every: int = 50


# ============
# Model config
# ============

@dataclass
class ModelCfgLM:
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    # Sparse memory
    n_experts: int = 256
    rank: int = 16
    max_k: int = 2
    group_size: int = 32


# =====================
# Controller configurations
# =====================

@dataclass
class FixedCtrlCfg:
    """Simple fixed (non-learned) controller parameters used for baselines."""

    k: int = 2
    replay_ratio: float = 0.10
    router_noise: float = 0.30
    router_temp: float = 1.0


@dataclass
class LearnedCtrlCfgLM:
    """Learned (RNN) TIUR controller configuration."""

    # Action bounds
    k_min: int = 1
    k_max: int = 2

    replay_max: float = 0.5
    noise_max: float = 0.5
    temp_min: float = 0.7
    temp_max: float = 1.3

    # RL
    hidden_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    rollout_len: int = 50
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    # Optional warm start (not used by default)
    warmup_steps: int = 0


# =====================
# Environment configurations
# =====================

EnvType = Literal["wt2", "multidomain"]


@dataclass
class WT2EnvCfg:
    env_type: EnvType = "wt2"
    name: str = "wt2_perm_unique_10"

    n_tasks: int = 10
    permuted_vocab: bool = True
    perm_mode: Literal["unique", "repeat", "drift"] = "unique"

    # For perm_mode="repeat": number of unique permutations to cycle
    repeat_k: int = 4

    # For perm_mode="drift": number of random swaps per step (controls correlation of shifts)
    drift_swaps: int = 150

    # Limits for speed
    max_docs_total: Optional[int] = None
    val_frac_per_task: float = 0.1


@dataclass
class MultiDomainEnvCfg:
    env_type: EnvType = "multidomain"
    name: str = "md_rr_40"

    n_tasks: int = 40
    schedule_mode: Literal["round_robin", "blocks", "custom"] = "round_robin"
    block_size: int = 10

    # If schedule_mode="custom", provide a list like ["wt2","ptb","agnews",...]
    schedule: Optional[List[str]] = None

    # How many distinct domains to mix into each task.
    # 1 => pure domain task. 2/3 => mixture (harder).
    mix_n_domains_per_task: int = 1
    mix_seed: int = 0

    # Docs sampled per task per split
    train_docs_per_task: int = 800
    val_docs_per_task: int = 200
    test_docs_per_task: int = 200

    # Vocab control (multi-domain vocab can explode)
    min_freq: int = 2
    max_vocab_size: int = 60000

    # Optional: cap dataset sizes for quick experiments
    max_docs_per_domain: Optional[int] = None


# =====================
# Helper
# =====================

def to_jsonable(dc_obj) -> Dict[str, Any]:
    """dataclass -> JSON-serializable dict"""
    return asdict(dc_obj)
