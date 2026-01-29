"""
Stochastic MuZero-style World Model

Implements world models with explicit afterstate/chance separation:

    state ─→ Dynamics(state, action) ─→ afterstate ─→ Chance(afterstate) ─→ next_state
                    │                        │                │
              DETERMINISTIC            where rules         STOCHASTIC  
               (rule core)              live here        (chance events)

Two input modalities:
- Grid: Takes symbolic board state directly (fast, clear diagnostics)
- Pixel: Takes raw images (the real test - must learn perception + rules)

Usage:
    # Grid-based training (fast, for debugging)
    python -m world_models.stoch_muzero.train --game 2048 --train_steps 30000 --w_policy 0 --w_value 0
    
    # Pixel-based training (THE REAL TEST)
    python -m world_models.stoch_muzero.train_pixel --game 2048 --train_steps 10000
    
    # Evaluation
    python -m world_models.stoch_muzero.eval --ckpt outputs/ckpt_final.pt

Key Metrics:
    - cell_acc: Per-cell prediction accuracy (the right metric)
    - changed_acc: Accuracy on cells that changed (rule learning test)
    - pred_entropy: Model's predicted entropy for chance outcomes
    - oracle_entropy: True entropy (perfect model target)
    
Expected Results:
    - 2048: Bimodal entropy (deterministic slides + stochastic spawns)
    - Othello: All near-zero entropy (fully deterministic)
"""

__all__ = [
    # Grid-based models
    'CausalWorldModel',
    'ModelConfig',
    'create_model',
    'TrainConfig',
    'MacroCache',
    'MacroOperator',
    'HierarchicalMCTS',
    # Games (grid)
    'Game2048Env',
    'OthelloEnv',
    # Pixel-based (lazy import to avoid torch dependency)
    'get_pixel_model',
    'get_pixel_games',
]


# ============================================================================
# Grid-based model imports (symbolic input)
# ============================================================================
from .model import CausalWorldModel, ModelConfig, create_model
from .train import TrainConfig
from .macro_cache import MacroCache, MacroOperator, HierarchicalMCTS
from .games import Game2048Env, OthelloEnv


# ============================================================================
# Pixel-based model imports (lazy to avoid torch at import time)
# ============================================================================
def get_pixel_model():
    """Get pixel-based model classes.
    
    Returns:
        Tuple of (PixelWorldModel, PixelModelConfig)
        
    Example:
        PixelWorldModel, PixelModelConfig = get_pixel_model()
        config = PixelModelConfig(img_size=64, state_dim=256)
        model = PixelWorldModel(config)
    """
    from .pixel_model import PixelWorldModel, PixelModelConfig
    return PixelWorldModel, PixelModelConfig


def get_pixel_games():
    """Get pixel-wrapped game environments.
    
    Returns:
        Tuple of (PixelGame2048, PixelOthello)
        
    Example:
        PixelGame2048, PixelOthello = get_pixel_games()
        game = PixelGame2048(img_size=64)
        obs = game.reset()  # Returns 64x64 grayscale image
    """
    from .pixel_games import PixelGame2048, PixelOthello
    return PixelGame2048, PixelOthello
