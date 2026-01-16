from __future__ import annotations

from typing import Union

from ..config import MultiDomainEnvCfg, WT2EnvCfg
from .base import EnvData
from .multidomain import build_multidomain_env
from .wt2 import build_wt2_env


def build_env(cfg: Union[WT2EnvCfg, MultiDomainEnvCfg]) -> EnvData:
    if cfg.env_type == "wt2":
        return build_wt2_env(cfg)  # type: ignore[arg-type]
    if cfg.env_type == "multidomain":
        return build_multidomain_env(cfg)  # type: ignore[arg-type]
    raise ValueError(f"Unknown env_type: {getattr(cfg, 'env_type', None)}")
