"""
World Models for Causal Structure Learning

This package contains world models that learn to separate:
1. CAUSAL CORE: Deterministic rule-governed transitions (low entropy)
2. STOCHASTIC ENVELOPE: Irreducible randomness (matches oracle entropy)

Domains:
- Board Games (stoch_muzero/): 2048, Othello with grid or pixel input

The key insight: Rules are not symbols — they are compressible causal structure.

Usage:
    # Pixel-based training (the real test)
    python -m world_models.stoch_muzero.train_pixel --game 2048 --train_steps 10000
    
    # Grid-based training (fast debugging)  
    python -m world_models.stoch_muzero.train --game 2048 --train_steps 30000
    
    # Evaluation
    python -m world_models.stoch_muzero.eval --ckpt outputs/ckpt_final.pt
"""

__all__ = ['stoch_muzero']
