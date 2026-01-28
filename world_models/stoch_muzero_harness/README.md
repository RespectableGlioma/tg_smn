# Stochastic MuZero Harness: Rule Core vs Chance + Nuisance

This is a compact research harness to test two ideas:

1) **Separate deterministic rules from stochastic transitions** using a Stochastic-MuZero-style factorization:
- Rule core (afterstate): `a_s = phi(s, a)` (deterministic, cheap)
- Chance: `c ~ sigma(c | a_s)` (stochastic)
- Apply chance: `s' = g(a_s, c)` (deterministic given `c`)

2) **Learn symbolic-like rules from pixels**, by adding *mechanistic* auxiliary heads:
- Decode true underlying state (Othello board / 2048 tiles) from the latent state `s`
- Decode the *afterstate* symbolic state from `a_s` (direct supervision of the rule core)
- Force observation nuisance into a separate latent `u` via style-classification

Benchmarks included:
- **Othello** (deterministic rules; chance is degenerate)
- **2048** (deterministic slide/merge afterstate + stochastic tile spawn)

## Install (Colab)

```bash
pip -q install torch numpy pillow tqdm
```

## Train

**Important:** run as a module so relative imports work:

### Othello
```bash
python -m world_models.stoch_muzero_harness.train \
  --game othello \
  --collect_episodes 200 \
  --train_steps 10000 \
  --eval_every 2000
```

### 2048
```bash
python -m world_models.stoch_muzero_harness.train \
  --game 2048 \
  --collect_episodes 200 \
  --train_steps 10000 \
  --eval_every 2000
```

Outputs:
- `outputs_stoch_muzero_harness/{game}/rollout_gt_vs_pred_stepXXXX.png`
- checkpoints `ckpt_stepXXXX.pt`, `ckpt_final.pt`

## Evaluate

Prediction metrics on fresh random episodes:
```bash
python -m world_models.stoch_muzero_harness.eval \
  --game 2048 \
  --ckpt outputs_stoch_muzero_harness/2048/ckpt_final.pt \
  --episodes 50
```

Optional: run **MCTS in the learned latent model** (2048 only):
```bash
python -m world_models.stoch_muzero_harness.eval \
  --game 2048 \
  --ckpt outputs_stoch_muzero_harness/2048/ckpt_final.pt \
  --episodes 50 \
  --mcts_sims 64 \
  --entropy_thr 0.5
```

## What to look for

- Othello:
  - `exact_after` and `exact_next` should approach 1.0 as it learns rules.
  - `chance_acc` is trivially 1.0 (chance size = 1).

- 2048:
  - `exact_after` tests **rule-core** learning (slide+merge).
  - `chance_acc` tests **stochastic modeling** (spawn position/value).
  - `exact_next` is harder (depends on both).

If `exact_after` becomes high while `chance_acc` remains calibrated, that’s strong evidence you’ve learned the deterministic rule core separately from the stochastic envelope.
