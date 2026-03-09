"""Abstract base class for all CaMoE experts."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseExpert(nn.Module, ABC):
    """
    Shared interface for all market participants.

    Each expert owns a scalar capital buffer and exposes a unified settlement
    entrypoint. Concrete subclasses implement the actual forward pass.
    Experts can also spend capital on optional capabilities such as future
    expansion rights.
    """

    def __init__(
        self,
        capital_init: float = 1.0,
        capital_floor: float = 0.01,
        capital_ceiling: float | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("capital", torch.tensor(float(capital_init), dtype=torch.float32))
        self.capital_floor = float(capital_floor)
        self.capital_ceiling = None if capital_ceiling is None else float(capital_ceiling)

    @abstractmethod
    def forward(self, x: torch.Tensor, **ctx) -> torch.Tensor:
        """Run the expert on the provided hidden states."""

    @property
    @abstractmethod
    def expert_type(self) -> str:
        """Return the expert family name."""

    @property
    def is_routable(self) -> bool:
        """Whether this expert participates in the token auction."""
        return True

    @property
    def supports_sparse_dispatch(self) -> bool:
        """Whether the block can pass only the winner token subset to this expert."""
        return True

    def settle(self, profit: torch.Tensor) -> None:
        """Update capital with a scalar profit signal."""
        value = torch.as_tensor(
            profit,
            device=self.capital.device,
            dtype=self.capital.dtype,
        )
        updated = self.capital + value
        if self.capital_ceiling is None:
            updated = updated.clamp(min=self.capital_floor)
        else:
            updated = updated.clamp(min=self.capital_floor, max=self.capital_ceiling)
        self.capital.copy_(updated)

    def spend_capital(
        self,
        cost: float | torch.Tensor,
        reserve: float | None = None,
    ) -> bool:
        """
        Spend capital without crossing the protected floor.

        Returns ``True`` when the spend succeeds and ``False`` when the expert
        cannot afford the requested cost while respecting the minimum reserve.
        """
        value = torch.as_tensor(
            cost,
            device=self.capital.device,
            dtype=self.capital.dtype,
        )
        if torch.any(value < 0):
            raise ValueError("Capital spend cost must be non-negative.")

        min_capital = self.capital_floor if reserve is None else max(self.capital_floor, float(reserve))
        updated = self.capital - value
        if torch.any(updated < min_capital):
            return False

        if self.capital_ceiling is None:
            updated = updated.clamp(min=min_capital)
        else:
            updated = updated.clamp(min=min_capital, max=self.capital_ceiling)
        self.capital.copy_(updated)
        return True
