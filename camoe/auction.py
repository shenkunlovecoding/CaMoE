"""Zero-parameter Vickrey auction used for routing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class VickreyAuctionHouse:
    """
    Stateless second-price auction.

    Bid definition:
      bid = expert_capital + critic_alpha * critic_position
    """

    def __init__(self, top_k: int = 2, noise_std: float = 0.01) -> None:
        self.top_k = int(top_k)
        self.noise_std = float(noise_std)

    def __call__(
        self,
        expert_capitals: torch.Tensor,
        critic_positions: torch.Tensor,
        critic_alpha: float = 1.0,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_experts = expert_capitals.numel()
        if self.top_k >= n_experts:
            raise ValueError(
                f"top_k={self.top_k} requires at least top_k + 1 experts, got {n_experts}."
            )

        with torch.no_grad():
            bids = expert_capitals.detach().view(1, 1, -1).to(critic_positions.dtype)
            bids = bids + float(critic_alpha) * critic_positions.detach()
            if training and self.noise_std > 0:
                bids = bids + torch.randn_like(bids) * self.noise_std

            top_vals, top_idxs = bids.topk(self.top_k + 1, dim=-1)
            winners = top_idxs[:, :, : self.top_k]
            winner_bids = top_vals[:, :, : self.top_k]
            prices = top_vals[:, :, self.top_k]
            weights = F.softmax(winner_bids, dim=-1)

        return winners, weights, prices
