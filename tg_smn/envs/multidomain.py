from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset

from ..config import MultiDomainEnvCfg
from ..tokenization import build_vocab, encode_docs, split_into_docs
from .base import EnvData


def _chunks(lst: List[str], n: int) -> List[str]:
    return [" ".join(lst[i:i+n]) for i in range(0, len(lst), n)]


def _load_wt2_docs(max_docs: Optional[int] = None) -> Tuple[List[str], List[str]]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_docs = split_into_docs(ds["train"]["text"])
    test_docs = split_into_docs(ds["test"]["text"])
    if max_docs is not None:
        train_docs = train_docs[:max_docs]
        test_docs = test_docs[:max_docs]
    return train_docs, test_docs


def _load_ptb_docs(max_docs: Optional[int] = None, concat_n: int = 20) -> Tuple[List[str], List[str]]:
    """Penn Treebank text-only docs.

    NOTE: As of `datasets>=4.0.0`, Hugging Face removed support for loading datasets that
    require executing a dataset loading script ("dataset scripts are no longer supported").
    The legacy `ptb_text_only` dataset relies on a script and will fail to load.

    We instead load PTB from a parquet-only mirror on the Hub.
    """

    # Parquet-only mirror (avoids dataset scripts):
    # https://huggingface.co/datasets/FALcon6/ptb_text_only
    repo = "FALcon6/ptb_text_only"
    base = "penn_treebank"
    # Use hf:// paths so the parquet loader doesn't touch the repo's Python script.
    data_files = {
        "train": f"hf://datasets/{repo}@main/{base}/train/*.parquet",
        "validation": f"hf://datasets/{repo}@main/{base}/validation/*.parquet",
        "test": f"hf://datasets/{repo}@main/{base}/test/*.parquet",
    }
    ds = load_dataset("parquet", data_files=data_files)

    def pick_text_column(split) -> str:
        # Prefer common names; otherwise choose the first string column.
        for name in ("text", "sentence", "content"):
            if name in split.column_names:
                return name
        # Fall back to any string feature
        try:
            for k, v in split.features.items():
                if getattr(v, "dtype", None) == "string":
                    return k
        except Exception:
            pass
        return split.column_names[0]

    # Merge validation into train pool to increase available training docs
    train_text_col = pick_text_column(ds["train"])
    valid_text_col = pick_text_column(ds["validation"])
    test_text_col = pick_text_column(ds["test"])

    train_sents = [t for t in ds["train"][train_text_col] if (t or "").strip()]
    valid_sents = [t for t in ds["validation"][valid_text_col] if (t or "").strip()]
    test_sents = [t for t in ds["test"][test_text_col] if (t or "").strip()]

    train_docs = _chunks(train_sents + valid_sents, concat_n)
    test_docs = _chunks(test_sents, concat_n)

    if max_docs is not None:
        train_docs = train_docs[:max_docs]
        test_docs = test_docs[:max_docs]
    return train_docs, test_docs


def _load_agnews_docs(max_docs: Optional[int] = None) -> Tuple[List[str], List[str]]:
    ds = load_dataset("ag_news")
    train_docs = [x["text"] for x in ds["train"] if (x.get("text") or "").strip()]
    test_docs = [x["text"] for x in ds["test"] if (x.get("text") or "").strip()]
    if max_docs is not None:
        train_docs = train_docs[:max_docs]
        test_docs = test_docs[:max_docs]
    return train_docs, test_docs


def _load_imdb_docs(max_docs: Optional[int] = None) -> Tuple[List[str], List[str]]:
    ds = load_dataset("imdb")
    train_docs = [x["text"] for x in ds["train"] if (x.get("text") or "").strip()]
    test_docs = [x["text"] for x in ds["test"] if (x.get("text") or "").strip()]
    if max_docs is not None:
        train_docs = train_docs[:max_docs]
        test_docs = test_docs[:max_docs]
    return train_docs, test_docs


DEFAULT_DOMAINS = {
    "wt2": _load_wt2_docs,
    "ptb": _load_ptb_docs,
    "agnews": _load_agnews_docs,
    "imdb": _load_imdb_docs,
}


def _build_schedule(cfg: MultiDomainEnvCfg, domain_keys: List[str]) -> List[str]:
    if cfg.schedule_mode == "round_robin":
        return [domain_keys[t % len(domain_keys)] for t in range(cfg.n_tasks)]

    if cfg.schedule_mode == "blocks":
        sched = []
        i = 0
        while len(sched) < cfg.n_tasks:
            dk = domain_keys[i % len(domain_keys)]
            sched.extend([dk] * cfg.block_size)
            i += 1
        return sched[: cfg.n_tasks]

    if cfg.schedule_mode == "custom":
        if not cfg.schedule:
            raise ValueError("schedule_mode='custom' requires cfg.schedule")
        if len(cfg.schedule) < cfg.n_tasks:
            # repeat if shorter
            sched = (cfg.schedule * ((cfg.n_tasks + len(cfg.schedule) - 1) // len(cfg.schedule)))[: cfg.n_tasks]
            return sched
        return cfg.schedule[: cfg.n_tasks]

    raise ValueError(f"Unknown schedule_mode: {cfg.schedule_mode}")


def build_multidomain_env(cfg: MultiDomainEnvCfg) -> EnvData:
    """Build a multi-domain continual LM environment.

    Tasks follow a domain schedule (round-robin, blocks, or custom), optionally mixing multiple
    domains per task.

    Evaluation is per-task (each task has its own test stream), suitable for forgetting metrics.
    """

    # Load docs per domain
    train_docs_by_domain: Dict[str, List[str]] = {}
    test_docs_by_domain: Dict[str, List[str]] = {}

    for key, loader in DEFAULT_DOMAINS.items():
        try:
            tr, te = loader(max_docs=cfg.max_docs_per_domain)
        except Exception as e:
            # Keep sweeps resilient: if a single domain fails to load on a given
            # machine/version, skip it rather than crashing the entire grid.
            print(f"[multidomain] WARNING: skipping domain '{key}' due to load error: {type(e).__name__}: {e}")
            continue
        train_docs_by_domain[key] = tr
        test_docs_by_domain[key] = te

    if len(train_docs_by_domain) < 2:
        raise RuntimeError(
            "Multi-domain env needs at least 2 successfully loaded domains. "
            f"Loaded domains: {list(train_docs_by_domain.keys())}"
        )

    domain_keys = list(train_docs_by_domain.keys())
    schedule = _build_schedule(cfg, domain_keys)

    rng = random.Random(cfg.mix_seed)

    task_train_docs: List[List[str]] = []
    task_val_docs: List[List[str]] = []
    task_test_docs: List[List[str]] = []

    # Helper: sample docs from a domain, with replacement if needed.
    def sample_docs(pool: List[str], n: int) -> List[str]:
        if len(pool) == 0:
            return [""] * n
        if n <= len(pool):
            return rng.sample(pool, n)
        return [rng.choice(pool) for _ in range(n)]

    for t in range(cfg.n_tasks):
        primary = schedule[t]
        domains = [primary]
        if cfg.mix_n_domains_per_task > 1:
            others = [d for d in domain_keys if d != primary]
            rng.shuffle(others)
            domains.extend(others[: cfg.mix_n_domains_per_task - 1])

        # split counts evenly
        def split_n(total: int, k: int) -> List[int]:
            base = total // k
            rem = total - base * k
            counts = [base] * k
            for i in range(rem):
                counts[i] += 1
            return counts

        tr_counts = split_n(cfg.train_docs_per_task, len(domains))
        va_counts = split_n(cfg.val_docs_per_task, len(domains))
        te_counts = split_n(cfg.test_docs_per_task, len(domains))

        tr_docs: List[str] = []
        va_docs: List[str] = []
        te_docs: List[str] = []

        for dom, n_tr, n_va, n_te in zip(domains, tr_counts, va_counts, te_counts):
            tr_pool = train_docs_by_domain[dom]
            te_pool = test_docs_by_domain[dom]

            tr_docs.extend(sample_docs(tr_pool, n_tr))
            # For validation, sample from train pool (simplifies; keeps distribution matched)
            va_docs.extend(sample_docs(tr_pool, n_va))
            te_docs.extend(sample_docs(te_pool, n_te))

        task_train_docs.append(tr_docs)
        task_val_docs.append(va_docs)
        task_test_docs.append(te_docs)

    # Build vocab from all sampled train docs
    vocab_docs = [d for docs in task_train_docs for d in docs]
    vocab = build_vocab(vocab_docs, min_freq=cfg.min_freq, max_vocab_size=cfg.max_vocab_size)
    vocab_size = len(vocab.itos)

    task_train_streams: List[torch.Tensor] = []
    task_val_streams: List[torch.Tensor] = []
    task_test_streams: List[torch.Tensor] = []

    for t in range(cfg.n_tasks):
        task_train_streams.append(encode_docs(task_train_docs[t], vocab))
        task_val_streams.append(encode_docs(task_val_docs[t], vocab))
        task_test_streams.append(encode_docs(task_test_docs[t], vocab))

    meta = {
        "cfg": asdict(cfg),
        "schedule": schedule,
    }

    return EnvData(
        name=cfg.name,
        vocab_size=vocab_size,
        pad_id=vocab.pad_id,
        unk_id=vocab.unk_id,
        eos_id=vocab.eos_id,
        task_train_streams=task_train_streams,
        task_val_streams=task_val_streams,
        task_test_streams=task_test_streams,
        meta=meta,
    )
