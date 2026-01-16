from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path: str) -> str:
    """Create (if needed) and return a normalized directory path.

    - Expands `~` for Colab/local friendliness.
    - Leaves absolute paths (e.g. `/content/drive/...`) untouched.
    """
    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj: Any) -> None:
    def default(o):
        if is_dataclass(o):
            return asdict(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
