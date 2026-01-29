# World Models: Causal Structure Learning

This package implements world models that learn to separate **causal rules** from **stochastic noise**.

## Core Insight

> **Rules are not symbols — they are compressible causal structure.**

A well-trained world model will have:
- **Low entropy** for deterministic transitions (learned rules)
- **Matching entropy** for stochastic transitions (chance events like tile spawns)

## Architecture

```
                         PERCEPTION                    DYNAMICS
                         ──────────                    ────────
                         
Observation ──→ Encoder ──→ state ──→ Dynamics(s,a) ──→ afterstate ──→ Chance ──→ next_state
    │              │          │            │                │            │
    │              │          │            │                │            │
  pixels       CNN/MLP     latent    DETERMINISTIC    deterministic   STOCHASTIC
  or grid                   repr      (rule core)       result       (chance node)
```

The key is the **afterstate/chance split**:
- `Dynamics(state, action) → afterstate` is DETERMINISTIC (the rules)
- `Chance(afterstate) → next_state` models STOCHASTIC outcomes

## Quick Start

### Train (Pixel-based - THE REAL TEST)

```bash
# 2048 from pixels - must learn perception + rules
python -m world_models.stoch_muzero.train_pixel \
    --game 2048 \
    --collect_episodes 500 \
    --train_steps 10000

# Othello from pixels (fully deterministic)
python -m world_models.stoch_muzero.train_pixel \
    --game othello \
    --train_steps 15000
```

### Train (Grid-based - fast debugging)

```bash
# 2048 with symbolic input (faster, clearer diagnostics)
python -m world_models.stoch_muzero.train \
    --game 2048 \
    --collect_episodes 2000 \
    --train_steps 30000 \
    --w_policy 0 --w_value 0
```

### Evaluate

```bash
python -m world_models.stoch_muzero.eval \
    --ckpt outputs_stoch_muzero/2048/ckpt_final.pt
```

## Two Input Modalities

### Grid Input (`model.py`) - Training Wheels

Takes symbolic board state directly:
```python
board = [[2, 4, 0, 0],
         [2, 0, 8, 0], ...]
```

**Pros**: Fast, clear diagnostics, easy debugging  
**Cons**: Sidesteps the perception problem

### Pixel Input (`pixel_model.py`) - The Real Test

Takes rendered images (64×64 grayscale):
```python
pixels = render_board(board)  # → [64, 64] float array
```

**This is what we need for generalization.** Must learn both perception AND rules.

## Games

### 2048 (Perfect Test Case)

Has both components:
1. **Deterministic Afterstate**: slide+merge is fully predictable
2. **Stochastic Chance**: tile spawns randomly (90% 2-tile, 10% 4-tile)

Expected results (well-trained):
```
Board prediction:
  cell_acc: 0.95+
  changed_acc: 0.80+
  
Chance prediction:
  pred_entropy ≈ oracle_entropy (within 0.2 bits)
```

### Othello (Fully Deterministic)

No randomness - every transition is deterministic.

Expected results:
```
All transitions: entropy ≈ 0
```

## Key Metrics

### Board Prediction
- `cell_acc`: Per-cell accuracy (**the right metric**)
- `nonempty_acc`: Accuracy on non-empty cells
- `changed_acc`: Accuracy on cells that changed (**rule learning test**)
- `exact_acc`: Fully correct boards (strict, expected low)

### Chance Prediction  
- `pred_entropy`: Model's predicted entropy
- `oracle_entropy`: True entropy (perfect model target)
- `entropy_error`: |predicted - oracle|

### Entropy Distribution
- Should be **bimodal** for 2048 (deterministic cluster + stochastic)
- Should be **near-zero** for Othello (fully deterministic)

## Files

```
world_models/
├── __init__.py
├── README.md
│
└── stoch_muzero/                       # Board game models
    ├── __init__.py
    │
    │  # Grid input (symbolic state)
    ├── model.py              # CausalWorldModel
    ├── train.py              # Grid training
    ├── eval.py               # Evaluation metrics
    │
    │  # Pixel input (the real test)
    ├── pixel_model.py        # PixelWorldModel  
    ├── pixel_games.py        # Board→pixel renderers
    ├── train_pixel.py        # Pixel training
    │
    │  # Macro caching
    ├── macro_cache.py        # Temporal compression
    │
    └── games/
        ├── game2048.py       # 2048 environment
        └── othello.py        # Othello environment
```

## Training Tips

### 1. Class Imbalance (Critical for 2048)

Empty cells dominate → model predicts "empty everywhere"

**Fix**: Use weighted loss
```bash
--empty_weight 0.2
--changed_cell_bonus 3.0
```

### 2. World Model Pretraining

Don't train policy/value until dynamics work

**Fix**: Disable RL losses initially
```bash
--w_policy 0 --w_value 0
```

### 3. Valid Actions Only

Random actions often don't change the board

**Fix**: Sample only board-changing actions (done automatically in data collection)

## Macro Caching (Hierarchical Planning)

Once the model learns rules, we can cache deterministic patterns:

```python
from world_models.stoch_muzero import MacroCache, get_pixel_model

PixelWorldModel, PixelModelConfig = get_pixel_model()
model = PixelWorldModel(config)
cache = MacroCache(entropy_threshold=0.1)

# During planning, cache low-entropy transitions
result = model.step(state, action)
if result['chance_entropy'] < 0.1:
    cache.store(state, action, result['next_state'])
```

This enables:
- **Faster planning**: Skip known outcomes
- **Temporal abstraction**: Multi-step macros
- **Transfer**: Reuse learned rules across contexts

## Initial Experiments to Run

### Experiment 1: Grid-based 2048 (Sanity Check)

Verify the architecture works before adding perception:

```bash
python -m world_models.stoch_muzero.train \
    --game 2048 \
    --collect_episodes 2000 \
    --train_steps 30000 \
    --w_policy 0 --w_value 0 \
    --empty_weight 0.2 \
    --changed_cell_bonus 3.0 \
    --outdir outputs/grid_2048_baseline
```

**Expected**: cell_acc > 0.90, changed_acc > 0.70

### Experiment 2: Pixel-based 2048 (The Real Test)

```bash
python -m world_models.stoch_muzero.train_pixel \
    --game 2048 \
    --collect_episodes 500 \
    --train_steps 15000 \
    --img_size 64 \
    --state_dim 256 \
    --outdir outputs/pixel_2048_baseline
```

**Expected**: Lower accuracy initially, but entropy distribution should still become bimodal

### Experiment 3: Pixel-based Othello (Determinism Test)

```bash
python -m world_models.stoch_muzero.train_pixel \
    --game othello \
    --collect_episodes 300 \
    --train_steps 15000 \
    --outdir outputs/pixel_othello_baseline
```

**Expected**: All predicted entropies should approach 0 (fully deterministic)

## Research Questions

1. **Does entropy distribution become bimodal?** (Rules vs chance)
2. **Can we transfer learned rules?** (Same rules, different visuals)
3. **Do macros emerge from compression?** (Temporal abstraction)
4. **How does pixel vs grid performance compare?** (Perception overhead)
