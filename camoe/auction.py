"""Zero-parameter Vickrey auction used for routing."""

from __future__ import annotations

import torch


class VickreyAuctionHouse:
    """
    Stateless second-price auction.

    Bid definition:
      bid = expert_capital + critic_alpha * critic_position

    v22 now uses strict Top-1 winner-takes-all routing.
    """

    def __init__(self, noise_std: float = 0.01) -> None:
        self.noise_std = float(noise_std)

    def __call__(
        self,
        expert_capitals: torch.Tensor,
        critic_positions: torch.Tensor,
        critic_alpha: float = 1.0,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_experts = expert_capitals.numel()
        if n_experts < 2:
            raise ValueError(
                f"winner-takes-all Vickrey auction requires at least 2 experts, got {n_experts}."
            )

        with torch.no_grad():
            bids = expert_capitals.detach().view(1, 1, -1).to(critic_positions.dtype)
            bids = bids + float(critic_alpha) * critic_positions.detach()
            if training and self.noise_std > 0:
                bids = bids + torch.randn_like(bids) * self.noise_std

            top_vals, top_idxs = bids.topk(2, dim=-1)
            winners = top_idxs[:, :, 0]
            prices = top_vals[:, :, 1]

        return winners, prices, bids
