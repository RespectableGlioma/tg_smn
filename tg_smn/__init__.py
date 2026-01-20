"""TG-SMN: Thermodynamically-governed Sparse Memory Networks.

This package contains:
- environment builders (continual LM tasks)
- dense + sparse Transformer LM models
- fixed and learned controllers
- training harness and sweep utilities

Note: keep package import side-effects minimal. Heavy dependencies (e.g. Hugging Face `datasets`)
are imported inside specific modules/functions. This allows notebooks to configure cache locations
(HF_HOME, HF_DATASETS_CACHE, etc.) before importing those heavy modules.
"""

__all__ = ["config"]
