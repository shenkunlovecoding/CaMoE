"""Reward critics for prediction-market CaMoE."""

from __future__ import annotations

import torch
import torch.nn as nn


class RewardCritic(nn.Module):
    """Predict per-token realized reward for each routable expert."""

    def __init__(
        self,
        dim: int,
        n_routable: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(1, dim // 4)
        self.n_routable = int(n_routable)
        self.reward_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_routable),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reward_net(x)

    def predict_reward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def supervised_loss(
        self,
        x_detached: torch.Tensor,
        winners: torch.Tensor,
        realized_reward: torch.Tensor,
        token_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_reward = self.predict_reward(x_detached)
        chosen_reward = torch.gather(pred_reward, dim=-1, index=winners.unsqueeze(-1)).squeeze(-1)
        weight = (
            torch.ones_like(realized_reward, dtype=chosen_reward.dtype, device=chosen_reward.device)
            if token_weight is None
            else torch.as_tensor(token_weight, device=chosen_reward.device, dtype=chosen_reward.dtype)
        )
        loss = (chosen_reward - realized_reward).pow(2) * weight
        return loss.sum() / weight.sum().clamp(min=1e-8)
