"""RWKV feed-forward expert used by the pure market router."""

from __future__ import annotations

import torch
import torch.nn as nn

from .expert_base import BaseExpert


class RWKVExpert(BaseExpert):
    """
    Minimal sparse expert.

    Structure:
      x -> LayerNorm -> Linear(D, 4D) -> SiLU -> Linear(4D, D)
    """

    def __init__(
        self,
        dim: int,
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
        hidden = int(dim * expand)
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.act = nn.SiLU()

    @property
    def expert_type(self) -> str:
        return "rwkv"

    def forward(self, x: torch.Tensor, **ctx) -> torch.Tensor:
        del ctx
        h = self.norm(x)
        return self.w2(self.act(self.w1(h)))
