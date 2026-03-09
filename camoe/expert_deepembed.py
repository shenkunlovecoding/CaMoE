"""Expertized RWKV-8-inspired DeepEmbed branches for the FFN market."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert_base import BaseExpert


class DeepEmbedExpert(BaseExpert):
    """
    RWKV-8-style token-conditioned DeepEmbed expert.

    This expert competes directly in the FFN market. It uses sparse token lookup
    and supports true sparse dispatch because the lookup only needs the selected
    token ids.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        expand: int = 4,
        mode: str = "1x",
        capital_init: float = 1.0,
        capital_floor: float = 0.01,
        capital_ceiling: float | None = None,
    ) -> None:
        super().__init__(
            capital_init=capital_init,
            capital_floor=capital_floor,
            capital_ceiling=capital_ceiling,
        )
        if mode not in {"1x", "4x"}:
            raise ValueError("DeepEmbedExpert mode must be '1x' or '4x'.")
        if expand <= 0:
            raise ValueError("DeepEmbedExpert expand must be positive.")

        hidden = dim * expand
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        self.key = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(hidden, dim, bias=False)
        self.deepemb = nn.Embedding(vocab_size, hidden if mode == "4x" else dim)
        nn.init.normal_(self.deepemb.weight, std=0.02)

    @property
    def expert_type(self) -> str:
        return "deepembed"

    @property
    def supports_sparse_dispatch(self) -> bool:
        return True

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None, **ctx) -> torch.Tensor:
        del ctx
        token_ids = self._require_token_ids(x, token_ids)
        h = self.norm(x)
        k = F.relu(self.key(h)).square()
        emb = self.deepemb(token_ids.to(device=x.device))
        if self.mode == "4x":
            return self.value(k * emb)
        return self.value(k) * emb

    @staticmethod
    def _require_token_ids(x: torch.Tensor, token_ids: torch.Tensor | None) -> torch.Tensor:
        if token_ids is None:
            raise ValueError("DeepEmbedExpert requires token_ids in expert context.")
        if token_ids.shape[: x.dim() - 1] != x.shape[: x.dim() - 1]:
            raise ValueError(
                f"token_ids shape {tuple(token_ids.shape)} is incompatible with x shape {tuple(x.shape)}."
            )
        return token_ids.long()


class SlimDeepEmbedExpert(BaseExpert):
    """
    Low-rank DeepEmbed-style modulator.

    This is a cheap per-layer alternative inspired by the later RWKV-8 low-rank
    variant. It competes in the FFN market without needing explicit token ids.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 32,
        expand: int = 4,
        capital_init: float = 1.0,
        capital_floor: float = 0.01,
        capital_ceiling: float | None = None,
    ) -> None:
        super().__init__(
            capital_init=capital_init,
            capital_floor=capital_floor,
            capital_ceiling=capital_ceiling,
        )
        if rank <= 0 or expand <= 0:
            raise ValueError("SlimDeepEmbedExpert requires positive rank and expand.")

        hidden = dim * expand
        self.norm = nn.LayerNorm(dim)
        self.key = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(hidden, dim, bias=False)
        self.s1 = nn.Linear(dim, rank, bias=False)
        self.semb = nn.Parameter(torch.randn(rank, rank) * 0.01)
        self.s2 = nn.Linear(rank, hidden, bias=False)
        self.s0 = nn.Parameter(torch.ones(hidden))

    @property
    def expert_type(self) -> str:
        return "slim_deepembed"

    @property
    def supports_sparse_dispatch(self) -> bool:
        return True

    def forward(self, x: torch.Tensor, **ctx) -> torch.Tensor:
        del ctx
        h = self.norm(x)
        k = F.relu(self.key(h)).square()
        ss = self.s1(h) @ self.semb
        k = k * (self.s2(ss) + self.s0)
        return self.value(k)
