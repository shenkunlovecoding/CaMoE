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
    """

    def __init__(self, capital_init: float = 1.0, capital_floor: float = 0.01) -> None:
        super().__init__()
        self.register_buffer("capital", torch.tensor(float(capital_init), dtype=torch.float32))
        self.capital_floor = float(capital_floor)

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

    def settle(self, profit: torch.Tensor) -> None:
        """Update capital with a scalar profit signal."""
        value = torch.as_tensor(
            profit,
            device=self.capital.device,
            dtype=self.capital.dtype,
        )
        updated = (self.capital + value).clamp(min=self.capital_floor)
        self.capital.copy_(updated)
