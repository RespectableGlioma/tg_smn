"""TG-SMN: Thermodynamically-governed Sparse Memory Networks.

This package contains:
- environment builders (continual LM tasks)
- dense + sparse Transformer LM models
- fixed and learned controllers
- training harness and sweep utilities

The main entry points intended for notebook use:

- tg_smn.envs.build_env(...)
- tg_smn.sweep.run_grid(...)
- tg_smn.analysis.load_grid_results(...)
"""

from . import config
from .sweep import run_grid

