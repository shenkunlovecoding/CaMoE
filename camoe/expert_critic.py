"""Critic experts for pure-market CaMoE."""

from __future__ import annotations

import torch
import torch.nn as nn

from .expert_base import BaseExpert


class CriticExpert(BaseExpert):
    """
    Market critic that takes long/short positions on routable experts.

    The critic never participates in token routing and is trained only through
    the separate REINFORCE-style objective.
    """

    def __init__(
        self,
        dim: int,
        n_routable: int,
        hidden_dim: int | None = None,
        capital_init: float = 0.5,
        capital_floor: float = 0.01,
    ) -> None:
        super().__init__(capital_init=capital_init, capital_floor=capital_floor)
        hidden_dim = hidden_dim or max(1, dim // 4)
        self.n_routable = int(n_routable)
        self.position_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_routable),
            nn.Tanh(),
        )

    @property
    def expert_type(self) -> str:
        return "critic"

    @property
    def is_routable(self) -> bool:
        return False

    def forward(self, x: torch.Tensor, **ctx) -> torch.Tensor:
        del ctx
        return self.position_net(x)

    def compute_pnl(
        self,
        positions: torch.Tensor,
        expert_profits: torch.Tensor,
    ) -> torch.Tensor:
        per_token = (positions * expert_profits.view(1, 1, -1)).sum(dim=-1)
        return per_token.mean()


class CriticPair(nn.Module):
    """Two critics that use each other as variance-reduction baselines."""

    def __init__(self, dim: int, n_routable: int, **critic_kwargs) -> None:
        super().__init__()
        self.critic_a = CriticExpert(dim, n_routable, **critic_kwargs)
        self.critic_b = CriticExpert(dim, n_routable, **critic_kwargs)

    def get_positions(self, x: torch.Tensor) -> torch.Tensor:
        pos_a = self.critic_a(x)
        pos_b = self.critic_b(x)
        return (pos_a + pos_b) / 2

    def settle_both(
        self,
        x_detached: torch.Tensor,
        expert_profits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_a = self.critic_a(x_detached)
        pos_b = self.critic_b(x_detached)

        pnl_a = self.critic_a.compute_pnl(pos_a, expert_profits)
        pnl_b = self.critic_b.compute_pnl(pos_b, expert_profits)

        self.critic_a.settle(pnl_a.detach())
        self.critic_b.settle(pnl_b.detach())

        adv_a = pnl_a - pnl_b.detach()
        adv_b = pnl_b - pnl_a.detach()
        return adv_a.detach(), adv_b.detach()

    def reinforce_loss(
        self,
        x_detached: torch.Tensor,
        expert_profits: torch.Tensor,
        adv_a: torch.Tensor,
        adv_b: torch.Tensor,
    ) -> torch.Tensor:
        pos_a = self.critic_a(x_detached)
        pos_b = self.critic_b(x_detached)

        pnl_a = self.critic_a.compute_pnl(pos_a, expert_profits)
        pnl_b = self.critic_b.compute_pnl(pos_b, expert_profits)
        return -(adv_a * pnl_a + adv_b * pnl_b)
