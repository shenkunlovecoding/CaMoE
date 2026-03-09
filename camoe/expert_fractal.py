"""Fractal CaMoE placeholders for future nested-submarket expansion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import torch
import torch.nn as nn

from .expert_base import BaseExpert


@dataclass
class FractalBlueprint:
    """
    Static metadata describing a potential nested CaMoE branch.

    The blueprint is only a placeholder. It does not build a child market on
    its own. Expansion must be explicitly funded and attached later.
    """

    depth: int = 0
    max_depth: int = 0
    slot_name: str = "fractal"
    child_kind: str = "camoe"
    notes: str = "placeholder"


class ExpansionRightLedger(nn.Module):
    """Bookkeeping for paid expansion rights owned by a fractal placeholder."""

    def __init__(self, default_cost: float = 1.0) -> None:
        super().__init__()
        self.default_cost = float(default_cost)
        self.register_buffer("rights_owned", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("rights_spent", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("total_spend", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("last_purchase_cost", torch.tensor(0.0, dtype=torch.float32))

    def purchase(
        self,
        owner: BaseExpert,
        price: float | torch.Tensor | None = None,
        quantity: int = 1,
        reserve: float | None = None,
    ) -> bool:
        if quantity <= 0:
            raise ValueError("Expansion right quantity must be positive.")
        unit_price = self.default_cost if price is None else price
        total_cost = torch.as_tensor(
            unit_price,
            device=owner.capital.device,
            dtype=owner.capital.dtype,
        ) * float(quantity)
        if not owner.spend_capital(total_cost, reserve=reserve):
            return False
        self.rights_owned.add_(float(quantity))
        self.total_spend.add_(total_cost.detach().to(self.total_spend.device, self.total_spend.dtype))
        self.last_purchase_cost.copy_(
            torch.as_tensor(unit_price, device=self.last_purchase_cost.device, dtype=self.last_purchase_cost.dtype)
        )
        return True

    def grant(self, quantity: int = 1) -> None:
        if quantity <= 0:
            raise ValueError("Expansion right quantity must be positive.")
        self.rights_owned.add_(float(quantity))

    def consume(self, quantity: int = 1) -> None:
        if quantity <= 0:
            raise ValueError("Expansion right quantity must be positive.")
        if self.rights_owned.item() < quantity:
            raise RuntimeError("Not enough expansion rights to consume.")
        self.rights_owned.sub_(float(quantity))
        self.rights_spent.add_(float(quantity))


class FractalCaMoEPlaceholder(BaseExpert):
    """
    Placeholder expert for future fractal CaMoE nesting.

    By default the placeholder is inert and returns zeros, so adding it to the
    codebase has no behavioral effect. Once a child market is attached it simply
    forwards to that child and treats the nested CaMoE as a sub-expert.

    Expansion is gated by paid rights. A future parent market can let a child
    buy one or more rights, then later exercise a right to attach a nested
    CaMoE block or model.
    """

    def __init__(
        self,
        dim: int,
        *,
        blueprint: FractalBlueprint | None = None,
        depth: int = 0,
        max_depth: int = 0,
        slot_name: str = "fractal",
        expansion_right_cost: float = 1.0,
        child_config: Any | None = None,
        fallback_mode: str = "zero",
        capital_init: float = 1.0,
        capital_floor: float = 0.01,
        capital_ceiling: float | None = None,
    ) -> None:
        super().__init__(
            capital_init=capital_init,
            capital_floor=capital_floor,
            capital_ceiling=capital_ceiling,
        )
        self.dim = int(dim)
        self.blueprint = blueprint or FractalBlueprint(
            depth=int(depth),
            max_depth=int(max_depth),
            slot_name=slot_name,
        )
        self.fallback_mode = str(fallback_mode)
        if self.fallback_mode not in {"zero", "identity"}:
            raise ValueError("fallback_mode must be 'zero' or 'identity'.")

        self.child_config = self._snapshot_config(child_config)
        self.expansion_rights = ExpansionRightLedger(default_cost=expansion_right_cost)
        self.child_market: nn.Module | None = None

    @property
    def expert_type(self) -> str:
        return "fractal"

    @property
    def supports_sparse_dispatch(self) -> bool:
        return False

    @property
    def depth(self) -> int:
        return int(self.blueprint.depth)

    @property
    def max_depth(self) -> int:
        return int(self.blueprint.max_depth)

    @property
    def slot_name(self) -> str:
        return str(self.blueprint.slot_name)

    @property
    def has_child_market(self) -> bool:
        return self.child_market is not None

    @property
    def can_expand(self) -> bool:
        return (
            self.depth < self.max_depth
            and not self.has_child_market
            and self.expansion_rights.rights_owned.item() >= 1.0
        )

    def grant_expansion_rights(self, quantity: int = 1) -> None:
        self.expansion_rights.grant(quantity)

    def buy_expansion_right(
        self,
        price: float | torch.Tensor | None = None,
        quantity: int = 1,
        reserve: float | None = None,
    ) -> bool:
        return self.expansion_rights.purchase(self, price=price, quantity=quantity, reserve=reserve)

    def attach_child_market(self, child_market: nn.Module) -> None:
        if self.depth >= self.max_depth:
            raise RuntimeError(
                f"Fractal placeholder {self.slot_name!r} is already at max_depth={self.max_depth}."
            )
        if self.child_market is not None:
            raise RuntimeError(f"Fractal placeholder {self.slot_name!r} already has a child market attached.")
        self.expansion_rights.consume(1)
        self.child_market = child_market

    def detach_child_market(self) -> None:
        self.child_market = None

    def describe(self) -> dict[str, Any]:
        return {
            "expert_type": self.expert_type,
            "slot_name": self.slot_name,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "has_child_market": self.has_child_market,
            "rights_owned": float(self.expansion_rights.rights_owned.item()),
            "rights_spent": float(self.expansion_rights.rights_spent.item()),
            "total_spend": float(self.expansion_rights.total_spend.item()),
            "child_config": self.child_config,
        }

    def forward(self, x: torch.Tensor, **ctx) -> torch.Tensor:
        if self.child_market is None:
            return self._fallback_output(x)
        result = self.child_market(x, **ctx)
        return self._coerce_child_output(result)

    def _fallback_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.fallback_mode == "identity":
            return x
        return torch.zeros_like(x)

    @staticmethod
    def _coerce_child_output(result: Any) -> torch.Tensor:
        if isinstance(result, torch.Tensor):
            return result
        if isinstance(result, tuple) and result and isinstance(result[0], torch.Tensor):
            return result[0]
        raise TypeError(
            "Fractal child market must return a Tensor or a tuple whose first element is a Tensor."
        )

    @staticmethod
    def _snapshot_config(config: Any | None) -> dict[str, Any] | None:
        if config is None:
            return None
        if is_dataclass(config):
            return asdict(config)
        if isinstance(config, Mapping):
            return dict(config)
        return {"repr": repr(config)}
