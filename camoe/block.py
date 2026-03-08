"""Single pure-market CaMoE block."""

from __future__ import annotations

import torch
import torch.nn as nn

from .auction import VickreyAuctionHouse
from .expert_base import BaseExpert
from .expert_critic import CriticPair


class CaMoE_Block(nn.Module):
    """A routable expert pool plus critic pair and zero-parameter auction."""

    def __init__(
        self,
        experts: list[BaseExpert],
        critic_pair: CriticPair,
        top_k: int = 2,
        auction_noise: float = 0.01,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.critic_pair = critic_pair
        self.n_routable = len(experts)
        self.top_k = int(top_k)
        self.auction = VickreyAuctionHouse(top_k=self.top_k, noise_std=auction_noise)
        self._cache: dict[str, torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        critic_alpha: float = 1.0,
        training: bool = True,
        uniform: bool = False,
        **expert_ctx,
    ) -> torch.Tensor:
        if uniform:
            self._cache = {}
            return self._forward_uniform(x, **expert_ctx)

        positions = self.critic_pair.get_positions(x.detach()).detach()
        expert_caps = torch.stack([expert.capital for expert in self.experts]).to(x.device)
        winners, weights, prices = self.auction(
            expert_caps,
            positions,
            critic_alpha=critic_alpha,
            training=training,
        )
        output = self._dispatch(x, winners, weights, **expert_ctx)

        self._cache = {
            "winners": winners.detach(),
            "weights": weights.detach(),
            "prices": prices.detach(),
            "positions": positions.detach(),
            "x_detached": x.detach(),
        }
        return output

    def _dispatch(
        self,
        x: torch.Tensor,
        winners: torch.Tensor,
        weights: torch.Tensor,
        **expert_ctx,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)

        for expert_idx, expert in enumerate(self.experts):
            match = winners == expert_idx
            selected = match.any(dim=-1)
            if not bool(selected.any()):
                continue

            expert_weight = (weights * match.to(weights.dtype)).sum(dim=-1)
            expert_out = expert(x, **expert_ctx)
            compensated = (
                expert_out * expert_weight.unsqueeze(-1)
                + expert_out.detach() * (1.0 / self.top_k - expert_weight).unsqueeze(-1)
            )
            output = output + compensated * selected.unsqueeze(-1).to(x.dtype)

        return output

    def _forward_uniform(self, x: torch.Tensor, **expert_ctx) -> torch.Tensor:
        outputs = [expert(x, **expert_ctx) for expert in self.experts]
        return sum(outputs) / len(outputs)

    def get_cache(self) -> dict[str, torch.Tensor]:
        return self._cache
