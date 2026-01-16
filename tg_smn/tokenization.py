from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def split_into_docs(lines: Sequence[str]) -> List[str]:
    """Split raw WikiText lines into doc-like chunks.

    - Splits on blank lines
    - Treats section headers like `= ... =` as boundaries
    """
    docs: List[str] = []
    cur: List[str] = []
    header_re = re.compile(r"^=+.*=+$")

    for ln in lines:
        s = (ln or "").strip()
        if s == "":
            if cur:
                docs.append(" ".join(cur))
                cur = []
            continue

        if header_re.match(s) and cur:
            docs.append(" ".join(cur))
            cur = [s]
        else:
            cur.append(s)

    if cur:
        docs.append(" ".join(cur))

    return docs


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int
    eos_id: int


def build_vocab(
    docs: Iterable[str],
    min_freq: int = 2,
    max_vocab_size: Optional[int] = None,
    specials: Tuple[str, str, str] = ("<pad>", "<unk>", "<eos>"),
) -> Vocab:
    ctr = Counter()
    for d in docs:
        ctr.update(tokenize(d))

    itos = list(specials)

    # Sort by frequency desc, then token for determinism
    items = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))
    for tok, c in items:
        if c < min_freq:
            continue
        if tok in specials:
            continue
        itos.append(tok)
        if max_vocab_size is not None and len(itos) >= max_vocab_size:
            break

    stoi = {t: i for i, t in enumerate(itos)}
    pad_id = stoi[specials[0]]
    unk_id = stoi[specials[1]]
    eos_id = stoi[specials[2]]
    return Vocab(stoi=stoi, itos=itos, pad_id=pad_id, unk_id=unk_id, eos_id=eos_id)


def encode_docs(docs: Iterable[str], vocab: Vocab) -> torch.Tensor:
    ids: List[int] = []
    for d in docs:
        toks = tokenize(d)
        for t in toks:
            ids.append(vocab.stoi.get(t, vocab.unk_id))
        ids.append(vocab.eos_id)
    return torch.tensor(ids, dtype=torch.long)
