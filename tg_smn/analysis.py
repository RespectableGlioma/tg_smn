from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def load_grid_results(out_root: str) -> pd.DataFrame:
    path = os.path.join(out_root, "grid_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def aggregate_results(
    df: pd.DataFrame,
    metric: str,
    group_cols: Sequence[str] = ("env", "variant", "ablation", "n_experts"),
) -> pd.DataFrame:
    g = df.groupby(list(group_cols))[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    return out


def plot_scaling(
    df: pd.DataFrame,
    *,
    env: str,
    metric: str,
    variants: Optional[Sequence[str]] = None,
    ablation: str = "none",
    title: Optional[str] = None,
):
    """Plot metric vs number of experts for one environment."""
    sub = df[df["env"] == env]
    sub = sub[sub["ablation"] == ablation]
    if variants is not None:
        sub = sub[sub["variant"].isin(list(variants))]

    agg = aggregate_results(sub, metric)

    plt.figure()
    for v in agg["variant"].unique():
        a = agg[agg["variant"] == v].sort_values("n_experts")
        plt.errorbar(a["n_experts"], a["mean"], yerr=a["std"], marker="o", label=v)

    plt.xlabel("n_experts")
    plt.ylabel(metric)
    plt.xscale("log", base=2)
    if title is None:
        title = f"{env} | {metric}"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
