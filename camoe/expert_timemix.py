"""Sequence-market TimeMix expert wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import RWKV7_TimeMix
from .expert_base import BaseExpert


class TimeMixExpert(BaseExpert):
    """RWKV-7 TimeMix wrapped as a market participant with capital."""

    def __init__(
        self,
        dim: int,
        n_layers: int,
        layer_idx: int,
        head_size: int,
        capital_init: float = 1.0,
        capital_floor: float = 0.01,
        capital_ceiling: float | None = None,
    ) -> None:
        super().__init__(
            capital_init=capital_init,
            capital_floor=capital_floor,
            capital_ceiling=capital_ceiling,
        )
        self.input_norm = nn.LayerNorm(dim)
        self.timemix = RWKV7_TimeMix(
            n_embd=dim,
            n_layer=n_layers,
            layer_id=layer_idx,
            head_size=head_size,
        )

    @property
    def expert_type(self) -> str:
        return "timemix"

    @property
    def supports_sparse_dispatch(self) -> bool:
        return False

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None = None,
        **ctx,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del ctx
        normed = self.input_norm(x)
        return self.timemix(normed, v_first)
