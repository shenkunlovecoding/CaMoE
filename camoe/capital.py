"""Vectorized capital accounting for pure-market CaMoE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertCapitalManager(nn.Module):
    """
    Manage expert capital and running losses as layer-wise buffers.

    The settlement path is vectorized over all experts within a layer.
    """

    def __init__(
        self,
        n_layers: int,
        n_experts_per_layer: int,
        capital_init: float = 1.0,
        ema_decay: float = 0.95,
        capital_floor: float = 0.01,
        capital_ceiling: float | None = None,
        depreciation: float = 0.001,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.n_experts = int(n_experts_per_layer)
        self.ema_decay = float(ema_decay)
        self.capital_floor = float(capital_floor)
        self.capital_ceiling = None if capital_ceiling is None else float(capital_ceiling)
        self.depreciation = float(depreciation)

        self.register_buffer(
            "capitals",
            torch.full((self.n_layers, self.n_experts), float(capital_init), dtype=torch.float32),
        )
        self.register_buffer(
            "running_loss",
            torch.zeros(self.n_layers, self.n_experts, dtype=torch.float32),
        )

    def settle_layer(
        self,
        layer_idx: int,
        winners: torch.Tensor,
        token_loss: torch.Tensor,
        prices: torch.Tensor,
        token_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Settle one layer and return expert profits shaped ``[E]``.
        """
        num_experts = self.n_experts
        caps = self.capitals[layer_idx]
        dtype = token_loss.dtype

        winner_mask = F.one_hot(winners, num_classes=num_experts).to(dtype=dtype)
        if token_weight is not None:
            token_weight = torch.as_tensor(token_weight, device=token_loss.device, dtype=dtype)
            weighted_mask = winner_mask * token_weight.unsqueeze(-1)
        else:
            weighted_mask = winner_mask

        total_selected = weighted_mask.sum(dim=(0, 1)).clamp(min=1e-8)
        weighted_loss = (token_loss.unsqueeze(-1) * weighted_mask).sum(dim=(0, 1))
        mean_loss = weighted_loss / total_selected

        active = total_selected > 0.01
        running = self.running_loss[layer_idx].to(dtype=dtype)
        bootstrap = active & running.abs().lt(1e-8)
        baseline = torch.where(bootstrap, mean_loss, running)
        relative = torch.where(
            active,
            baseline - mean_loss,
            torch.zeros_like(mean_loss),
        )

        updated_running = torch.where(
            active,
            self.ema_decay * baseline + (1.0 - self.ema_decay) * mean_loss,
            running,
        )
        self.running_loss[layer_idx].copy_(updated_running.to(self.running_loss.dtype))

        revenue = relative * caps.to(dtype=dtype)
        if token_weight is None:
            total_tokens = max(token_loss.shape[0] * token_loss.shape[1], 1)
            cost = (prices.unsqueeze(-1) * winner_mask).sum(dim=(0, 1)) / float(total_tokens)
        else:
            cost = (prices.unsqueeze(-1) * weighted_mask).sum(dim=(0, 1)) / total_selected
        cost = torch.where(active, cost, torch.zeros_like(cost))
        depreciation = self.depreciation * caps.to(dtype=dtype)
        profit = revenue - cost - depreciation

        next_caps = caps.to(dtype=dtype) + profit
        if self.capital_ceiling is None:
            next_caps = next_caps.clamp(min=self.capital_floor)
        else:
            next_caps = next_caps.clamp(min=self.capital_floor, max=self.capital_ceiling)
        self.capitals[layer_idx].copy_(next_caps.to(self.capitals.dtype))
        return profit.detach()

    def sync_to_experts(self, layer_idx: int, experts: list[nn.Module]) -> None:
        for expert_idx, expert in enumerate(experts):
            expert.capital.fill_(float(self.capitals[layer_idx, expert_idx].item()))

    def sync_from_experts(self, layer_idx: int, experts: list[nn.Module]) -> None:
        for expert_idx, expert in enumerate(experts):
            self.capitals[layer_idx, expert_idx] = float(expert.capital.item())
