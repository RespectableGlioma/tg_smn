"""Stochastic MuZero-style harness with explicit rule-core (afterstate) vs chance + nuisance u.

Benchmarks:
- Othello (deterministic rules; chance is degenerate)
- 2048 (deterministic slide/merge afterstate + stochastic tile spawn chance)

Designed to assess:
1) Separating deterministic rules from stochastic transitions
2) Learning symbolic-like rules from pixels

Entry points:
- train.py
- eval.py
"""
