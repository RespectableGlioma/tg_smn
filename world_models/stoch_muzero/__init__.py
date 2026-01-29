"""
Stochastic MuZero-style World Model

Implements world models with explicit afterstate/chance separation:

    state ─→ Dynamics(state, action) ─→ afterstate ─→ Chance(afterstate) ─→ next_state
                    │                        │                │
              DETERMINISTIC            where rules         STOCHASTIC  
               (rule core)              live here        (chance events)

Three approaches:
- Grid: Takes symbolic board state directly (fast, clear diagnostics)
- Pixel: Takes raw images (original approach - had entropy calibration issues)
- VQ-VAE: Discrete codes from images (RECOMMENDED - entropy emerges from data)

Usage:
    # Grid-based training (fast, for debugging)
    python -m world_models.stoch_muzero.train --game 2048 --train_steps 30000 --w_policy 0 --w_value 0
    
    # VQ-VAE v2 pixel training (RECOMMENDED - with dead code reset!)
    python -m world_models.stoch_muzero.train_vq_v2 --game 2048 --train_steps 20000
    python -m world_models.stoch_muzero.train_vq_v2 --game othello --train_steps 20000
    
    # Legacy pixel training
    python -m world_models.stoch_muzero.train_pixel --game 2048 --train_steps 10000
    
    # Evaluation
    python -m world_models.stoch_muzero.eval --ckpt outputs/ckpt_final.pt

Key Insight (VQ-VAE approach):
    Discrete latent codes make entropy LEARNABLE FROM DATA:
    - Same (code, action) → same next_code = deterministic (entropy → 0)
    - Same (code, action) → varied next_codes = stochastic (entropy > 0)
    
    No oracle entropy bonuses/penalties needed! The model discovers which
    transitions are stochastic from the data itself.

Rule Extraction:
    After training, use rule_extraction module to:
    - Visualize what each code represents
    - Extract transition rules (deterministic) vs chance (stochastic)
    - Generate interpretable summaries
    
    from world_models.stoch_muzero import get_rule_extractor
    RuleExtractor, analyze_model = get_rule_extractor()
    extractor = analyze_model(model, obs, actions, device)
    
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
    # VQ-VAE based (RECOMMENDED - entropy learned from data)
    'get_vq_model',
    # Rule extraction and visualization
    'get_rule_extractor',
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


def get_vq_model(version: int = 2):
    """Get VQ-VAE based world model classes (RECOMMENDED).
    
    This is the recommended approach for pixel-based world modeling.
    Unlike the original pixel model which required oracle entropy bonuses,
    the VQ-VAE model learns calibrated uncertainty FROM DATA:
    
    - Discrete codebook → categorical transition distribution
    - Cross-entropy loss naturally calibrates uncertainty
    - Deterministic transitions → peaked distribution (low entropy)
    - Stochastic transitions → spread distribution (high entropy)
    
    Args:
        version: 1 for original, 2 for improved with dead code reset (default)
    
    Returns:
        Tuple of (VQWorldModel, VQWorldModelConfig)
        
    Example:
        VQWorldModel, VQWorldModelConfig = get_vq_model()
        cfg = VQWorldModelConfig(img_size=64, n_actions=4, codebook_size=512)
        model = VQWorldModel(cfg)
        
        # Encode image to discrete codes
        enc = model.encode(img)  # z_q, indices, vq_loss
        
        # Step dynamics + chance
        result = model.step(enc['z_q'], action)  # entropy is DATA-DRIVEN!
        
    Training:
        python -m world_models.stoch_muzero.train_vq_v2 --game 2048
        python -m world_models.stoch_muzero.train_vq_v2 --game othello
    """
    if version == 2:
        from .vq_model_v2 import VQWorldModel, VQWorldModelConfig
    else:
        from .vq_model import VQWorldModel, VQWorldModelConfig
    return VQWorldModel, VQWorldModelConfig


def get_rule_extractor():
    """Get rule extraction and visualization tools.
    
    Use this to analyze trained VQ models and extract interpretable rules.
    
    Returns:
        Tuple of (RuleExtractor, analyze_model)
        
    Example:
        RuleExtractor, analyze_model = get_rule_extractor()
        
        # Full analysis with visualizations
        extractor = analyze_model(model, obs, actions, device, 
                                  game_name='2048', save_dir='analysis/')
        
        # Or manual analysis
        extractor = RuleExtractor(model, device)
        extractor.analyze_codebook(obs_samples)
        extractor.extract_transitions(obs, actions)
        print(extractor.get_rule_summary())
    """
    from .rule_extraction import RuleExtractor, analyze_model
    return RuleExtractor, analyze_model
