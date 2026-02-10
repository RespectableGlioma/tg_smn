# Stochastic MuZero with Learned Temporal Abstractions

## Project Overview

This project implements **Stochastic MuZero** with a novel approach to discovering **rules as compressible causal structure** rather than explicit symbols.

### The Core Insight

> **Rules are not symbols — they are compressible causal structure in the transition dynamics.**

Traditional AI defines rules symbolically: `IF condition THEN action`. We instead discover rules operationally:

- If a sequence of transitions is **deterministic** (low entropy)
- And **repeatable** (occurs across different states)
- And **compositional** (intermediate states don't matter)

Then it can be collapsed into a **macro-operator** — a reusable temporal abstraction.

### Why This Matters

| Traditional Symbolic Rules | Our Approach |
|---------------------------|--------------|
| Fixed primitives | Representation-agnostic |
| Predefined operators | Discovered from data |
| Brittle boundaries | Graceful degradation |
| Hard to learn from pixels | Works in latent space |

The key question shifts from *"What is the rule?"* to *"Which transitions don't matter?"*

---

## Theoretical Framework

### Macro-Operator Discovery Criterion

A sequence of actions `a_t, ..., a_{t+k}` forms a macro if:

1. **Low Entropy**: `H(s_{t+k} | s_t, a_{t:t+k}) ≈ 0`
   - The model confidently predicts the outcome

2. **Composition Invariance**: `||rollout(s_t, actions) - direct_predict(s_t, actions)|| < ε`
   - Step-by-step and direct prediction agree

3. **Context Generality**: Pattern works from multiple initial states
   - Not just a single-context fluke

### Connection to Causal Abstraction

This is equivalent to learning **causal homomorphisms** in the transition graph:
- Low-level: single-step transitions
- High-level: multi-step macro transitions
- Validity: predictive equivalence + context invariance

---

## Current Implementation

### Architecture

```
stochastic_muzero_colab_v2.ipynb   # Main notebook (Atari-focused)
├── AtariGame                       # Deterministic Atari wrapper
├── MuZeroNetwork                   # Representation + Dynamics + Prediction + Entropy
├── MCTS                            # Tree search with entropy tracking
└── MacroCache                      # Dual-threshold macro discovery
```

### Key Components

#### 1. MuZero Network with Entropy Head
```python
class MuZeroNetwork:
    def recurrent_inference(self, state, action):
        # Returns: next_state, reward, policy, value, ENTROPY
        # Entropy = model's uncertainty about this transition
        # Low entropy = deterministic = macro candidate
```

#### 2. Dual-Threshold Macro Discovery
```python
class MacroCache:
    play_threshold = 0.2   # Tight for actual trajectories
    mcts_threshold = 0.4   # Loose for MCTS speculation

    # Tracks:
    # - Action patterns
    # - Average entropy across pattern
    # - State buckets (generality)
    # - Source (play vs MCTS)
```

#### 3. State-Conditioned Tracking
```python
def _get_state_bucket(self, hidden_state):
    # Hash state via random projection
    # Macros that work from many buckets = more general
    # state_generality = len(state_buckets)
```

#### 4. MCTS with Rollout Collection
```python
class MCTS:
    def search(self, obs, legal_actions):
        # Explores many paths through dynamics model
        # Collects (actions, entropies, rewards) for each path
        # ALL paths used for macro discovery, not just played
```

### Why Atari?

We focus on Atari games because they're:
- **Single-player**: Agent controls all actions (no opponent uncertainty)
- **Deterministic physics**: Ball bounces, gravity, collisions predictable
- **Pixel inputs**: Tests learning from raw observations

Two-player games (Chess, TTT) don't work well for macros because alternating turns mean "my action → opponent response" sequences aren't under agent control.

---

## Results and Findings

### What Works

| Metric | Observation |
|--------|-------------|
| **Entropy decreases** | 0.7 → 0.03 over training |
| **Deterministic rate** | 0% → 70%+ as model learns |
| **MCTS rollouts** | 500K+ paths explored for macro discovery |
| **Loss decreases** | Model learns dynamics |

### Key Insight Validated

The model learns that Breakout transitions are deterministic:
- Untrained: entropy ~0.7 (uncertain)
- Trained: entropy ~0.03 (confident)

This confirms the core hypothesis: **entropy tracks determinism**.

### Challenges Discovered

1. **MCTS vs Play Entropy**: MCTS explores speculative paths with higher entropy than actual gameplay. Solution: dual thresholds.

2. **Early Discovery Problem**: Macros discovered early in training have high entropy (model hadn't learned yet). Solution: track entropy, sort by lowest.

3. **Score Collapse**: Training too long caused score to drop. Solution: fewer iterations, monitor for overfit.

---

## File Structure

```
tg_smn/
├── CLAUDE.md                          # This file
├── stochastic_muzero_colab_v2.ipynb   # Main notebook (Atari)
├── stochastic_muzero_colab.ipynb      # Original notebook (2048, TTT, Chess)
├── requirements.txt                   # Dependencies
├── games/
│   ├── base.py                        # Abstract game interface
│   ├── game_2048.py                   # 2048 with binary encoding
│   ├── game_tictactoe.py              # Tic-tac-toe
│   ├── game_chess.py                  # Chess with AlphaZero encoding
│   └── game_atari.py                  # Atari wrapper (deterministic mode)
├── networks/
│   └── muzero_network.py              # MuZero with entropy head
├── mcts/
│   ├── tree_search.py                 # Stochastic MCTS
│   └── macro_cache.py                 # Macro discovery
├── training/
│   ├── replay_buffer.py               # Experience replay
│   ├── trainer.py                     # Training loop
│   └── self_play.py                   # Self-play with macro tracking
├── config/
│   ├── game_2048.yaml
│   ├── game_breakout.yaml
│   ├── game_pong.yaml
│   └── game_spaceinvaders.yaml
└── utils/
    ├── config.py
    └── support.py                     # Value support encoding
```

---

## Next Steps

### Immediate Improvements

1. **Use Macros in Planning**
   - Currently: discover macros, don't use them
   - Next: MCTS expansion considers macro actions
   - Skip k steps when macro is applicable

2. **Macro Preconditions**
   - Learn WHEN a macro applies (not just the action sequence)
   - Neural network: state → applicable_macros

3. **Composition Error Tracking**
   - Compare: rollout(s, a1, a2, a3) vs direct_predict(s, [a1,a2,a3])
   - Low composition error = true macro

### Research Directions

1. **Transfer Learning**
   - Learn macros on Breakout
   - Test if they transfer to similar games (Pong?)
   - General macros should transfer

2. **Hierarchical Planning**
   - MCTS at multiple time scales
   - Primitive actions + macro actions as candidates
   - Automatic temporal abstraction

3. **Evaluation Metrics**
   - Planning speedup from macro use
   - Macro reuse rate across episodes
   - Generalization to novel states

4. **Theoretical Analysis**
   - Connection to information bottleneck
   - Minimum description length (MDL) perspective
   - Formal definition of "rule" via predictive equivalence

---

## Running the Code

### In Google Colab

1. Open `stochastic_muzero_colab_v2.ipynb`
2. Run cells 1-2 (install dependencies, ROMs)
3. Run cells 3-17 (define components, train)
4. Run cells 18+ (analyze macros)

### Locally

```bash
pip install -r requirements.txt
pip install gymnasium[atari] ale-py autorom
AutoROM --accept-license
jupyter notebook stochastic_muzero_colab_v2.ipynb
```

---

## Key References

- **Stochastic MuZero**: [Antonoglou et al., 2021](https://arxiv.org/abs/2104.06303)
- **MuZero**: [Schrittwieser et al., 2020](https://www.nature.com/articles/s41586-020-03051-4)
- **Causal Abstraction**: [Beckers & Halpern, Rubenstein et al.]
- **DreamCoder**: [Ellis et al., 2021](https://arxiv.org/abs/2006.08381) - program induction via compression

---

## Summary

This project demonstrates that:

1. **Rules can be discovered operationally** — as low-entropy transition sequences
2. **Entropy tracks determinism** — model uncertainty reveals what's predictable
3. **State generality reveals true rules** — patterns that work across contexts
4. **Single-player games are ideal** — agent controls all actions

The core contribution: **Learning hierarchical causal abstractions via predictive compression in world models.**
