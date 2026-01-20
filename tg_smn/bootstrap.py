from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def configure_runtime(
    cache_root: str,
    *,
    set_tmpdir: bool = False,
    tmp_root: Optional[str] = None,
    set_pip_cache: bool = False,
    pip_cache_root: Optional[str] = None,
    set_torch_home: bool = True,
) -> Dict[str, str]:
    """Configure cache locations to avoid filling Colab's local disk.

    This should be called **before importing** Hugging Face `datasets` / `transformers`
    (or any module that imports them), so the libraries pick up these cache dirs.

    Args:
        cache_root: Base directory for caches (recommend a Google Drive path in Colab).
        set_tmpdir: If True, set TMPDIR/TEMP/TMP under cache_root (can be slower on Drive).
        tmp_root: Optional override for temp dir base (used if set_tmpdir=True).
        set_pip_cache: If True, set PIP_CACHE_DIR (useful if pip downloads are large).
        pip_cache_root: Optional override for pip cache base (used if set_pip_cache=True).
        set_torch_home: If True, set TORCH_HOME under cache_root (torch hub cache).
    Returns:
        Dict of paths actually configured.
    """

    cache_root = os.path.expanduser(cache_root)
    Path(cache_root).mkdir(parents=True, exist_ok=True)

    # Hugging Face caches
    hf_root = os.path.join(cache_root, "hf")
    hf_datasets = os.path.join(hf_root, "datasets")
    hf_hub = os.path.join(hf_root, "hub")
    hf_transformers = os.path.join(hf_root, "transformers")
    for p in (hf_root, hf_datasets, hf_hub, hf_transformers):
        Path(p).mkdir(parents=True, exist_ok=True)

    # Only set if not already set by the user.
    os.environ.setdefault("HF_HOME", hf_root)
    os.environ.setdefault("HF_DATASETS_CACHE", hf_datasets)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_transformers)
    os.environ.setdefault("XDG_CACHE_HOME", hf_root)

    configured = {
        "HF_HOME": os.environ["HF_HOME"],
        "HF_DATASETS_CACHE": os.environ["HF_DATASETS_CACHE"],
        "HUGGINGFACE_HUB_CACHE": os.environ["HUGGINGFACE_HUB_CACHE"],
        "TRANSFORMERS_CACHE": os.environ["TRANSFORMERS_CACHE"],
        "XDG_CACHE_HOME": os.environ["XDG_CACHE_HOME"],
    }

    if set_torch_home:
        torch_home = os.path.join(cache_root, "torch")
        Path(torch_home).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", torch_home)
        configured["TORCH_HOME"] = os.environ["TORCH_HOME"]

    if set_pip_cache:
        pc = pip_cache_root or os.path.join(cache_root, "pip")
        Path(pc).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PIP_CACHE_DIR", pc)
        configured["PIP_CACHE_DIR"] = os.environ["PIP_CACHE_DIR"]

    if set_tmpdir:
        td = tmp_root or os.path.join(cache_root, "tmp")
        Path(td).mkdir(parents=True, exist_ok=True)
        # Warning: TMPDIR on Drive can be slower; prefer num_workers=0 instead.
        os.environ.setdefault("TMPDIR", td)
        os.environ.setdefault("TEMP", td)
        os.environ.setdefault("TMP", td)
        configured["TMPDIR"] = os.environ["TMPDIR"]

    return configured
