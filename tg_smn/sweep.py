from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import (
    DataCfg,
    FixedCtrlCfg,
    LearnedCtrlCfgLM,
    ModelCfgLM,
    MultiDomainEnvCfg,
    TrainCfgLM,
    WT2EnvCfg,
)
from .envs import build_env
from .train.lm_trainer import run_lm_experiment
from .utils import ensure_dir, get_device, load_json


@dataclass
class LearnedAblation:
    name: str
    fixed_k: Optional[int] = None
    fixed_replay: Optional[float] = None
    drop_obs_kl: bool = False
    drop_reward_kl: bool = False


def _summary_exists(path: str) -> bool:
    return os.path.exists(path)


def run_grid(
    env_cfgs: Sequence[Union[WT2EnvCfg, MultiDomainEnvCfg]],
    experts_list: Sequence[int],
    seeds: Sequence[int],
    *,
    out_root: str,
    variants: Sequence[str] = ("dense_baseline", "sparse_fixed", "tg_smn_learned"),
    data_cfg: Optional[DataCfg] = None,
    model_cfg: Optional[ModelCfgLM] = None,
    train_cfg: Optional[TrainCfgLM] = None,
    fixed_ctrl_cfg: Optional[FixedCtrlCfg] = None,
    learned_ctrl_cfg: Optional[LearnedCtrlCfgLM] = None,
    learned_ablations: Optional[Sequence[LearnedAblation]] = None,
    device: Optional[str] = None,
    skip_existing: bool = True,
    show_inner_progress: bool = False,
) -> pd.DataFrame:
    """Run a grid of experiments and write `grid_results.csv` under out_root.

    The notebook-friendly entry point.

    Directory structure under out_root:
        {env_name}/{variant}/{ablation}/experts{E}/seed{S}/...
    """

    device = get_device(device)
    out_root = ensure_dir(out_root)

    # Fallback logging location (local tmp dir) used only if the primary out_root
    # becomes temporarily unavailable (e.g., Drive mount issues). This prevents
    # long sweeps from crashing due to transient filesystem errors.
    fallback_root = ensure_dir(os.path.join(tempfile.gettempdir(), "tg_smn_fallback"))

    # Cache built environments under the sweep root so long sweeps are resumable
    # across runtime resets.
    env_cache_dir = ensure_dir(os.path.join(out_root, "_env_cache"))

    data_cfg = data_cfg or DataCfg()
    model_cfg = model_cfg or ModelCfgLM()
    train_cfg = train_cfg or TrainCfgLM()
    fixed_ctrl_cfg = fixed_ctrl_cfg or FixedCtrlCfg()

    rows: List[Dict[str, Any]] = []

    # Write incremental JSONL so long sweeps don't lose progress if the runtime resets.
    jsonl_path = os.path.join(out_root, "grid_results.jsonl")

    # Human-readable progress snapshot for monitoring from Drive.
    progress_path = os.path.join(out_root, "grid_progress.json")

    # Progress bar without ETA (time-to-completion estimates are intentionally omitted).
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {rate_fmt}"

    # Count total combinations (including those that may be skipped).
    per_variant = 0
    for v in variants:
        if v == "tg_smn_learned" and learned_ablations:
            per_variant += len(learned_ablations)
        else:
            per_variant += 1
    total_runs = len(env_cfgs) * len(experts_list) * len(seeds) * per_variant

    started_at = time.time()

    # Fallback log file paths (written only if out_root becomes unavailable).
    # The tag avoids collisions if multiple sweeps run in the same runtime.
    _tag = f"{os.path.basename(out_root.rstrip(os.sep))}_{int(started_at)}"
    fallback_jsonl_path = os.path.join(fallback_root, f"{_tag}_grid_results.jsonl")
    fallback_progress_path = os.path.join(fallback_root, f"{_tag}_grid_progress.json")

    done = 0
    skipped = 0
    failed = 0

    def _safe_append_jsonl(row: Dict[str, Any]) -> None:
        line = json.dumps(row) + "\n"
        # Try primary location.
        try:
            ensure_dir(out_root)
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(line)
            return
        except Exception:
            # Try to recreate parent dirs and retry once.
            try:
                ensure_dir(os.path.dirname(jsonl_path) or out_root)
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(line)
                return
            except Exception:
                # Fall back to local tmp dir so the sweep doesn't crash.
                ensure_dir(os.path.dirname(fallback_jsonl_path) or fallback_root)
                with open(fallback_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(line)

    def _safe_write_progress(snap: Dict[str, Any]) -> None:
        # Atomic-ish write with fallback.
        def _write_atomic(path: str) -> None:
            ensure_dir(os.path.dirname(path) or out_root)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snap, f, indent=2)
            os.replace(tmp, path)

        try:
            _write_atomic(progress_path)
        except Exception:
            try:
                _write_atomic(fallback_progress_path)
            except Exception:
                # Last resort: don't crash the sweep for progress logging.
                return

    def write_progress(current: Optional[Dict[str, Any]] = None) -> None:
        snap = {
            "total": int(total_runs),
            "done": int(done),
            "skipped": int(skipped),
            "failed": int(failed),
            "elapsed_sec": float(time.time() - started_at),
            "current": current,
            "fallback_jsonl_path": fallback_jsonl_path,
            "fallback_progress_path": fallback_progress_path,
        }
        _safe_write_progress(snap)

    pbar = tqdm(total=total_runs, desc="TG-SMN sweep", bar_format=bar_format)
    write_progress(current=None)

    for env_cfg in env_cfgs:
        # Environment build can be expensive (tokenization, dataset loads).
        # Cache by env_cfg signature so restarts don't redo it.
        cfg_dict = asdict(env_cfg)
        cfg_sig = hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        cache_path = os.path.join(env_cache_dir, f"{env_cfg.name}_{cfg_sig}.pt")

        if os.path.exists(cache_path):
            try:
                env = torch.load(cache_path)
            except Exception:
                # If the cache is corrupted/incomplete, rebuild.
                env = build_env(env_cfg)
        else:
            env = build_env(env_cfg)

        # Best-effort write of env cache.
        try:
            ensure_dir(os.path.dirname(cache_path) or env_cache_dir)
            torch.save(env, cache_path)
        except Exception:
            pass

        for E in experts_list:
            # Update model cfg for this run
            mc = replace(model_cfg, n_experts=int(E))
            # Basic validity check
            if mc.n_experts % mc.group_size != 0:
                raise ValueError(f"n_experts={mc.n_experts} must be divisible by group_size={mc.group_size}")

            for seed in seeds:
                tc = replace(train_cfg, seed=int(seed))

                for variant in variants:
                    if variant == "tg_smn_learned" and learned_ablations:
                        # Run learned controller ablations
                        for abl in learned_ablations:
                            exp_name = os.path.join(
                                env.name,
                                variant,
                                abl.name,
                                f"experts{E}",
                                f"seed{seed}",
                            )
                            summary_path = os.path.join(out_root, exp_name, "summary.json")
                            current = {"env": env.name, "variant": variant, "ablation": abl.name, "n_experts": int(E), "seed": int(seed)}
                            pbar.set_postfix(current)
                            write_progress(current=current)

                            if skip_existing and _summary_exists(summary_path):
                                s = load_json(summary_path)
                                row = {"env": env.name, "variant": variant, "ablation": abl.name, "n_experts": E, "seed": seed, "status": "skipped", **s}
                                rows.append(row)
                                _safe_append_jsonl(row)
                                skipped += 1
                                done += 1
                                pbar.update(1)
                                continue

                            # Disable noisy inner progress bars during sweeps by default.
                            tc_run = replace(tc, show_progress=bool(show_inner_progress))
                            try:
                                s = run_lm_experiment(
                                    exp_name=exp_name,
                                    variant=variant,
                                    env=env,
                                    data_cfg=data_cfg,
                                    model_cfg=mc,
                                    train_cfg=tc_run,
                                    out_dir=out_root,
                                    device=device,
                                    fixed_ctrl_cfg=fixed_ctrl_cfg,
                                    learned_ctrl_cfg=learned_ctrl_cfg,
                                    learned_fixed_k=abl.fixed_k,
                                    learned_fixed_replay=abl.fixed_replay,
                                    learned_drop_obs_kl=abl.drop_obs_kl,
                                    learned_drop_reward_kl=abl.drop_reward_kl,
                                )
                                row = {"env": env.name, "variant": variant, "ablation": abl.name, "n_experts": E, "seed": seed, "status": "ok", **s}
                            except Exception as e:
                                failed += 1
                                row = {
                                    "env": env.name,
                                    "variant": variant,
                                    "ablation": abl.name,
                                    "n_experts": E,
                                    "seed": seed,
                                    "status": "error",
                                    "error_type": type(e).__name__,
                                    "error_message": str(e),
                                    "exp_name": exp_name,
                                }
                                pbar.write(f"[ERROR] {current} -> {type(e).__name__}: {e}")
                            rows.append(row)
                            _safe_append_jsonl(row)
                            done += 1
                            pbar.update(1)
                            write_progress(current=None)
                    else:
                        exp_name = os.path.join(env.name, variant, "none", f"experts{E}", f"seed{seed}")
                        summary_path = os.path.join(out_root, exp_name, "summary.json")
                        current = {"env": env.name, "variant": variant, "ablation": "none", "n_experts": int(E), "seed": int(seed)}
                        pbar.set_postfix(current)
                        write_progress(current=current)

                        if skip_existing and _summary_exists(summary_path):
                            s = load_json(summary_path)
                            row = {"env": env.name, "variant": variant, "ablation": "none", "n_experts": E, "seed": seed, "status": "skipped", **s}
                            rows.append(row)
                            _safe_append_jsonl(row)
                            skipped += 1
                            done += 1
                            pbar.update(1)
                            continue

                        tc_run = replace(tc, show_progress=bool(show_inner_progress))
                        try:
                            s = run_lm_experiment(
                                exp_name=exp_name,
                                variant=variant,
                                env=env,
                                data_cfg=data_cfg,
                                model_cfg=mc,
                                train_cfg=tc_run,
                                out_dir=out_root,
                                device=device,
                                fixed_ctrl_cfg=fixed_ctrl_cfg,
                                learned_ctrl_cfg=learned_ctrl_cfg,
                            )
                            row = {"env": env.name, "variant": variant, "ablation": "none", "n_experts": E, "seed": seed, "status": "ok", **s}
                        except Exception as e:
                            failed += 1
                            row = {
                                "env": env.name,
                                "variant": variant,
                                "ablation": "none",
                                "n_experts": E,
                                "seed": seed,
                                "status": "error",
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "exp_name": exp_name,
                            }
                            pbar.write(f"[ERROR] {current} -> {type(e).__name__}: {e}")
                        rows.append(row)
                        _safe_append_jsonl(row)
                        done += 1
                        pbar.update(1)
                        write_progress(current=None)

    pbar.close()
    write_progress(current=None)

    df = pd.DataFrame(rows)
    # Best-effort write. If out_root is temporarily unavailable, write to fallback.
    try:
        ensure_dir(out_root)
        df.to_csv(os.path.join(out_root, "grid_results.csv"), index=False)
    except Exception:
        try:
            df.to_csv(os.path.join(fallback_root, f"{_tag}_grid_results.csv"), index=False)
        except Exception:
            pass
    return df
