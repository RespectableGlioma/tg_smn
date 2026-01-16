from __future__ import annotations

from dataclasses import asdict

from ..config import FixedCtrlCfg


class FixedController:
    """A minimal controller that always returns the same actions."""

    def __init__(self, cfg: FixedCtrlCfg):
        self.cfg = cfg

    def act(self):
        return {
            "k": int(self.cfg.k),
            "replay_ratio": float(self.cfg.replay_ratio),
            "noise": float(self.cfg.router_noise),
            "temp": float(self.cfg.router_temp),
        }

    def state_dict(self):
        return {"cfg": asdict(self.cfg)}
