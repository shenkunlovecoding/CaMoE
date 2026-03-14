"""Market-state accounting for prediction-market CaMoE."""

from __future__ import annotations

import torch
import torch.nn as nn

from .auction import PredictionMarketRouter


class MarketStateManager(nn.Module):
    """Manage per-layer wallets, prices, and shared market reward baselines."""

    def __init__(
        self,
        n_layers: int,
        n_experts_per_layer: int,
        capital_init: float = 1.0,
        ema_decay: float = 0.95,
        capital_floor: float = 0.01,
        capital_ceiling: float | None = None,
        price_lr: float = 0.02,
        price_temperature: float = 1.0,
        liquidity_floor: float = 0.02,
        exploration_epsilon: float = 0.02,
        reward_scale: float = 5.0,
        reward_eps: float = 1e-8,
        bet_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.n_experts = int(n_experts_per_layer)
        self.ema_decay = float(ema_decay)
        self.capital_floor = float(capital_floor)
        self.capital_ceiling = None if capital_ceiling is None else float(capital_ceiling)
        self.price_lr = float(price_lr)
        self.price_temperature = float(price_temperature)
        self.liquidity_floor = float(liquidity_floor)
        self.exploration_epsilon = float(exploration_epsilon)
        self.force_winner: int | None = None
        self.reward_scale = float(reward_scale)
        self.reward_eps = float(reward_eps)
        self.bet_fraction = float(bet_fraction)

        self.register_buffer(
            "capitals",
            torch.full((self.n_layers, self.n_experts), float(capital_init), dtype=torch.float32),
        )
        self.register_buffer(
            "q",
            torch.zeros(self.n_layers, self.n_experts, dtype=torch.float32),
        )
        self.register_buffer(
            "loss_ema",
            torch.zeros(self.n_layers, dtype=torch.float32),
        )

    def prices(self, layer_idx: int) -> torch.Tensor:
        return PredictionMarketRouter.compute_prices(
            self.q[layer_idx],
            temperature=self.price_temperature,
            liquidity_floor=self.liquidity_floor,
        )

    def route_state(self, layer_idx: int) -> dict[str, torch.Tensor | float]:
        return {
            "capital": self.capitals[layer_idx],
            "q": self.q[layer_idx],
            "bet_fraction": self.bet_fraction,
            "price_temperature": self.price_temperature,
            "liquidity_floor": self.liquidity_floor,
            "exploration_epsilon": getattr(self, "exploration_epsilon", 0.0),
            "force_winner": self.force_winner,
        }

    def settle_layer(
        self,
        layer_idx: int,
        winners: torch.Tensor,
        token_loss: torch.Tensor,
        prices: torch.Tensor,
        shares: torch.Tensor,
        token_weight: torch.Tensor | None = None,
        update_state: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Settle one layer and return token-level and expert-level market statistics.

        The caller passes `shares` computed once at batch-start from the current
        market prices and wallets; settlement reuses those same shares for every
        token in the batch.
        """
        dtype = token_loss.dtype
        device = token_loss.device
        num_experts = self.n_experts

        weights = (
            torch.ones_like(token_loss, dtype=dtype, device=device)
            if token_weight is None
            else torch.as_tensor(token_weight, device=device, dtype=dtype)
        )
        valid_mask = weights > 0
        total_valid = weights.sum().clamp(min=1e-8)

        market_mean_loss = (token_loss * weights).sum() / total_valid
        ema = self.loss_ema[layer_idx].to(device=device, dtype=dtype)
        market_baseline = ema if ema.abs().ge(1e-8) else market_mean_loss
        prices = prices.to(device=device, dtype=dtype)
        shares = shares.to(device=device, dtype=dtype)
        winners = winners.to(device=device, dtype=torch.long)

        baseline_denom = market_baseline.clamp(min=self.reward_eps)
        improvement = (market_baseline - token_loss) / baseline_denom
        reward_all = torch.sigmoid(improvement * self.reward_scale)
        selected_prices = prices[winners]
        selected_shares = shares[winners]
        selected_profit = selected_shares * (reward_all - selected_prices)

        realized_reward = torch.where(valid_mask, reward_all, torch.zeros_like(reward_all))
        token_profit = torch.where(valid_mask, selected_profit, torch.zeros_like(selected_profit))

        flat_valid = valid_mask.reshape(-1)
        flat_winners = winners.reshape(-1)[flat_valid]
        flat_weights = weights.reshape(-1)[flat_valid]
        flat_reward = reward_all.reshape(-1)[flat_valid]
        flat_profit = selected_profit.reshape(-1)[flat_valid]

        expert_weight = torch.zeros(num_experts, dtype=dtype, device=device)
        expert_weight.scatter_add_(0, flat_winners, flat_weights)

        profit = torch.zeros(num_experts, dtype=dtype, device=device)
        profit.scatter_add_(0, flat_winners, flat_profit * flat_weights)
        profit = profit / total_valid

        reward_sum = torch.zeros(num_experts, dtype=dtype, device=device)
        reward_sum.scatter_add_(0, flat_winners, flat_reward * flat_weights)

        avg_reward = reward_sum / expert_weight.clamp(min=1e-8)
        active = expert_weight > 0

        if update_state:
            next_caps = self.capitals[layer_idx].to(device=device, dtype=dtype) + profit
            if self.capital_ceiling is None:
                next_caps = next_caps.clamp(min=self.capital_floor)
            else:
                next_caps = next_caps.clamp(min=self.capital_floor, max=self.capital_ceiling)
            self.capitals[layer_idx].copy_(next_caps.to(self.capitals.dtype))

            next_q = self.q[layer_idx].to(device=device, dtype=dtype).clone()
            next_q[active] = next_q[active] + self.price_lr * (avg_reward[active] - prices[active])

            next_ema = self.ema_decay * market_baseline + (1.0 - self.ema_decay) * market_mean_loss
            self.loss_ema[layer_idx].copy_(next_ema.to(self.loss_ema.dtype))
            self.q[layer_idx].copy_(next_q.to(self.q.dtype))

        winner_share = expert_weight / total_valid

        return {
            "profit": profit.detach(),
            "realized_reward": realized_reward.detach(),
            "token_profit": token_profit.detach(),
            "avg_reward": avg_reward.detach(),
            "winner_share": winner_share.detach(),
            "winner_mask": valid_mask.detach(),
        }

    def sync_to_experts(self, layer_idx: int, experts: list[nn.Module]) -> None:
        for expert_idx, expert in enumerate(experts):
            expert.capital.fill_(float(self.capitals[layer_idx, expert_idx].item()))

    def sync_from_experts(self, layer_idx: int, experts: list[nn.Module]) -> None:
        for expert_idx, expert in enumerate(experts):
            self.capitals[layer_idx, expert_idx] = float(expert.capital.item())
