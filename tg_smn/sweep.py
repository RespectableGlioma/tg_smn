from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

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
) -> pd.DataFrame:
    """Run a grid of experiments and write `grid_results.csv` under out_root.

    The notebook-friendly entry point.

    Directory structure under out_root:
        {env_name}/{variant}/{ablation}/experts{E}/seed{S}/...
    """

    device = get_device(device)
    out_root = ensure_dir(out_root)

    data_cfg = data_cfg or DataCfg()
    model_cfg = model_cfg or ModelCfgLM()
    train_cfg = train_cfg or TrainCfgLM()
    fixed_ctrl_cfg = fixed_ctrl_cfg or FixedCtrlCfg()

    rows: List[Dict[str, Any]] = []

    for env_cfg in env_cfgs:
        env = build_env(env_cfg)

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
                            if skip_existing and _summary_exists(summary_path):
                                s = load_json(summary_path)
                                rows.append({"env": env.name, "variant": variant, "ablation": abl.name, "n_experts": E, "seed": seed, **s})
                                continue

                            s = run_lm_experiment(
                                exp_name=exp_name,
                                variant=variant,
                                env=env,
                                data_cfg=data_cfg,
                                model_cfg=mc,
                                train_cfg=tc,
                                out_dir=out_root,
                                device=device,
                                fixed_ctrl_cfg=fixed_ctrl_cfg,
                                learned_ctrl_cfg=learned_ctrl_cfg,
                                learned_fixed_k=abl.fixed_k,
                                learned_fixed_replay=abl.fixed_replay,
                                learned_drop_obs_kl=abl.drop_obs_kl,
                                learned_drop_reward_kl=abl.drop_reward_kl,
                            )
                            rows.append({"env": env.name, "variant": variant, "ablation": abl.name, "n_experts": E, "seed": seed, **s})
                    else:
                        exp_name = os.path.join(env.name, variant, "none", f"experts{E}", f"seed{seed}")
                        summary_path = os.path.join(out_root, exp_name, "summary.json")
                        if skip_existing and _summary_exists(summary_path):
                            s = load_json(summary_path)
                            rows.append({"env": env.name, "variant": variant, "ablation": "none", "n_experts": E, "seed": seed, **s})
                            continue

                        s = run_lm_experiment(
                            exp_name=exp_name,
                            variant=variant,
                            env=env,
                            data_cfg=data_cfg,
                            model_cfg=mc,
                            train_cfg=tc,
                            out_dir=out_root,
                            device=device,
                            fixed_ctrl_cfg=fixed_ctrl_cfg,
                            learned_ctrl_cfg=learned_ctrl_cfg,
                        )
                        rows.append({"env": env.name, "variant": variant, "ablation": "none", "n_experts": E, "seed": seed, **s})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "grid_results.csv"), index=False)
    return df
