from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelCfgLM


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class HierarchicalSparseExpertLayer(nn.Module):
    """Two-stage router:

    1) choose a group among G
    2) choose top-k experts within group among M

    Routing noise is applied ONLY to within-group (local) scores to avoid group thrash.

    Expert computation is low-rank:
        h -> relu(h @ W1) @ W2
    """

    def __init__(self, d: int, n_experts: int, rank: int, group_size: int):
        super().__init__()
        assert n_experts % group_size == 0, "n_experts must be divisible by group_size"
        self.d = d
        self.E = n_experts
        self.r = rank
        self.M = group_size
        self.G = n_experts // group_size

        self.group_router = nn.Linear(d, self.G)
        # group-specific local router weights: (G, d, M)
        self.local_router = nn.Parameter(torch.randn(self.G, d, self.M) * 0.02)

        self.W1 = nn.Parameter(torch.randn(self.E, d, rank) * 0.02)
        self.W2 = nn.Parameter(torch.randn(self.E, rank, d) * 0.02)

    def forward(
        self,
        h: torch.Tensor,
        k: int,
        router_temp: float = 1.0,
        router_noise: float = 0.0,
        stochastic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h: (N, d)
        returns:
          out: (N, d)
          expert_ids: (N, k)
          gate: (N, k)
          group_idx: (N,)
        """
        N, d = h.shape

        # --- stage 1: group ---
        g_scores = self.group_router(h)  # (N, G)
        g_topv, g_topi = torch.topk(g_scores, k=1, dim=-1)
        group_idx = g_topi.squeeze(-1)  # (N,)

        # --- stage 2: within-group ---
        Wloc = self.local_router[group_idx]  # (N, d, M)
        local_scores = torch.einsum("nd,ndm->nm", h, Wloc)  # (N, M)
        if stochastic and router_noise > 0:
            local_scores = local_scores + router_noise * torch.randn_like(local_scores)

        topv, topm = torch.topk(local_scores, k=k, dim=-1)  # (N,k)
        gate = F.softmax(topv / max(router_temp, 1e-6), dim=-1)

        expert_ids = group_idx.unsqueeze(-1) * self.M + topm  # (N,k)

        # gather experts
        W1_sel = self.W1[expert_ids]  # (N,k,d,r)
        W2_sel = self.W2[expert_ids]  # (N,k,r,d)

        tmp = torch.einsum("nd,nkdr->nkr", h, W1_sel)
        tmp = F.relu(tmp)
        out_k = torch.einsum("nkr,nkrd->nkd", tmp, W2_sel)
        out = (gate.unsqueeze(-1) * out_k).sum(dim=1)
        return out, expert_ids, gate, group_idx


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, dropout: float, mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mult * d_model)
        self.fc2 = nn.Linear(mult * d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        if stochastic and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)
        x = self.fc2(x)
        if stochastic and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)
        return x


class SparseFFN(nn.Module):
    def __init__(self, cfg: ModelCfgLM):
        super().__init__()
        self.cfg = cfg
        self.experts = HierarchicalSparseExpertLayer(cfg.d_model, cfg.n_experts, cfg.rank, cfg.group_size)
        self.dropout = cfg.dropout

        self.k = cfg.max_k
        self.router_temp = 1.0
        self.router_noise = 0.0

    def set_routing(self, k: Optional[int] = None, router_temp: Optional[float] = None, router_noise: Optional[float] = None) -> None:
        if k is not None:
            self.k = int(k)
        if router_temp is not None:
            self.router_temp = float(router_temp)
        if router_noise is not None:
            self.router_noise = float(router_noise)

    def forward(self, x: torch.Tensor, stochastic: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, d = x.shape
        h = x.reshape(B * T, d)
        out, expert_ids, gate, group_idx = self.experts(
            h,
            k=self.k,
            router_temp=self.router_temp,
            router_noise=self.router_noise,
            stochastic=stochastic,
        )
        out = out.reshape(B, T, d)
        if stochastic and self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=True)
        aux = {
            "topi": expert_ids,  # (N,k)
            "gate": gate,        # (N,k)
            "group": group_idx,  # (N,)
        }
        return out, aux


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelCfgLM, sparse: bool):
        super().__init__()
        self.cfg = cfg
        self.sparse = sparse

        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)

        if sparse:
            self.ffn = SparseFFN(cfg)
        else:
            self.ffn = DenseFFN(cfg.d_model, cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, stochastic: bool = True) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        if stochastic and self.cfg.dropout > 0:
            attn_out = F.dropout(attn_out, p=self.cfg.dropout, training=True)
        x = x + attn_out

        h2 = self.ln2(x)
        if self.sparse:
            ffn_out, aux = self.ffn(h2, stochastic=stochastic)
        else:
            ffn_out = self.ffn(h2, stochastic=stochastic)
            aux = None
        x = x + ffn_out
        return x, aux


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelCfgLM, vocab_size: int, sparse: bool):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.sparse = sparse

        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, sparse=sparse) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        self._sffn_layers: List[SparseFFN] = []
        if sparse:
            self._sffn_layers = [b.ffn for b in self.blocks]  # type: ignore

    def set_routing(self, k: Optional[int] = None, router_temp: Optional[float] = None, router_noise: Optional[float] = None) -> None:
        if not self.sparse:
            return
        for sffn in self._sffn_layers:
            sffn.set_routing(k=k, router_temp=router_temp, router_noise=router_noise)

    def forward(self, x: torch.Tensor, stochastic: bool = True) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        B, T = x.shape
        h = self.tok_emb(x)
        h = self.pos(h)
        if stochastic and self.cfg.dropout > 0:
            h = F.dropout(h, p=self.cfg.dropout, training=True)

        # Causal mask for MultiheadAttention
        attn_mask = torch.triu(torch.full((T, T), float("-inf"), device=h.device), diagonal=1)

        aux_all: List[Dict[str, torch.Tensor]] = []
        for blk in self.blocks:
            h, aux = blk(h, attn_mask=attn_mask, stochastic=stochastic)
            if aux is not None:
                aux_all.append(aux)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, aux_all


def lm_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over all positions."""
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
