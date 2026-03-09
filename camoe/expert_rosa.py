"""Slim Wind ROSA sequence expert."""

from __future__ import annotations

import torch
import torch.nn as nn

from .expert_base import BaseExpert
from .wind_rosa_adapter import wind_rosa


class ROSAExpert(BaseExpert):
    """
    Sequence-aware Slim Wind ROSA expert.

    The expert always processes the full sequence to preserve suffix state.
    Routing only decides which output tokens are kept.

    The symbolic branch is intentionally narrow: ``slim_heads * bits_per_symbol``
    can be far smaller than ``dim``. Wind ROSA returns discrete symbols which are
    decoded into the default signed bit semantics:
    matched ``1 -> +e``, matched ``0 -> -e``, unmatched ``-> 0``.
    """

    def __init__(
        self,
        dim: int,
        slim_heads: int,
        bits_per_symbol: int = 8,
        backend: str = "wind",
        truncation_length: int = 8,
        sequence_length: int | None = None,
        capital_init: float = 1.0,
        capital_floor: float = 0.01,
        capital_ceiling: float | None = None,
    ) -> None:
        super().__init__(
            capital_init=capital_init,
            capital_floor=capital_floor,
            capital_ceiling=capital_ceiling,
        )
        if backend != "wind":
            raise ValueError(f"ROSAExpert only supports backend='wind', got {backend!r}")
        if slim_heads <= 0 or bits_per_symbol <= 0:
            raise ValueError("ROSA expert requires positive slim_heads and bits_per_symbol.")

        self.dim = int(dim)
        self.slim_heads = int(slim_heads)
        self.bits_per_symbol = int(bits_per_symbol)
        self.truncation_length = int(truncation_length)
        self.sequence_length = None if sequence_length is None else int(sequence_length)
        hidden = self.slim_heads * self.bits_per_symbol

        self.norm = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, hidden, bias=False)
        self.wk = nn.Linear(dim, hidden, bias=False)
        self.wv = nn.Linear(dim, hidden, bias=False)
        self.symbol_scale = nn.Parameter(torch.ones(1, 1, hidden))
        self.wo = nn.Linear(hidden, dim, bias=False)

    @property
    def expert_type(self) -> str:
        return "rosa"

    @property
    def supports_sparse_dispatch(self) -> bool:
        return False

    def forward(self, x: torch.Tensor, **ctx) -> torch.Tensor:
        del ctx
        squeezed = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeezed = True
        if x.dim() != 3:
            raise ValueError(f"ROSAExpert expected [B, T, D] or [N, D], got {tuple(x.shape)}")

        _batch, steps, _dim = x.shape
        h = self.norm(x)
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)

        padded_q, padded_k, padded_v = self._maybe_pad_sequence(q, k, v)
        out = wind_rosa(
            padded_q,
            padded_k,
            padded_v,
            bits_per_symbol=self.bits_per_symbol,
            truncation_length=self.truncation_length,
        )
        out = out[:, :steps, :]
        out = out.to(h.dtype) * self.symbol_scale.to(h.dtype)
        out = self.wo(out)
        if squeezed:
            return out.squeeze(0)
        return out

    def _maybe_pad_sequence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steps = q.size(1)
        if self.sequence_length is None:
            return q, k, v
        if steps > self.sequence_length:
            raise ValueError(
                f"ROSAExpert received T={steps}, which exceeds configured sequence_length={self.sequence_length}."
            )
        if steps == self.sequence_length:
            return q, k, v
        pad = self.sequence_length - steps
        return (
            nn.functional.pad(q, (0, 0, 0, pad)),
            nn.functional.pad(k, (0, 0, 0, pad)),
            nn.functional.pad(v, (0, 0, 0, pad)),
        )
