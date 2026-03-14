"""Prediction-market router used for CaMoE routing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PredictionMarketRouter:
    """
    Stateless Top-1 router for Polymarket-style expert selection.

    Routing score:
      expected_profit = shares * (pred_reward - price)
    """

    def __init__(self, noise_std: float = 0.01) -> None:
        self.noise_std = float(noise_std)

    @staticmethod
    def compute_prices(
        q: torch.Tensor,
        temperature: float = 1.0,
        liquidity_floor: float = 0.02,
    ) -> torch.Tensor:
        if q.ndim != 1:
            raise ValueError(f"Expected 1D q, got shape={tuple(q.shape)}")
        if q.numel() < 2:
            raise ValueError(f"Prediction-market routing requires at least 2 experts, got {q.numel()}.")
        if temperature <= 0:
            raise ValueError(f"price_temperature must be positive, got {temperature}.")
        if not 0.0 <= liquidity_floor < 1.0:
            raise ValueError(f"liquidity_floor must be in [0, 1), got {liquidity_floor}.")

        base = F.softmax(q / float(temperature), dim=-1)
        return (1.0 - float(liquidity_floor)) * base + float(liquidity_floor) / q.numel()

    def __call__(
        self,
        expert_capitals: torch.Tensor,
        q: torch.Tensor,
        reward_logits: torch.Tensor,
        bet_fraction: float = 0.05,
        price_temperature: float = 1.0,
        liquidity_floor: float = 0.02,
        exploration_epsilon: float = 0.02,
        force_winner: int | None = None,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        if expert_capitals.ndim != 1:
            raise ValueError(f"Expected 1D expert_capitals, got shape={tuple(expert_capitals.shape)}")
        if q.shape != expert_capitals.shape:
            raise ValueError(f"q shape {tuple(q.shape)} must match capitals {tuple(expert_capitals.shape)}")
        if reward_logits.ndim != 3 or reward_logits.size(-1) != expert_capitals.numel():
            raise ValueError(
                "reward_logits must have shape [B, T, E] with E matching expert_capitals."
            )
        if not 0.0 <= bet_fraction:
            raise ValueError(f"bet_fraction must be non-negative, got {bet_fraction}.")
        if not 0.0 <= exploration_epsilon <= 1.0:
            raise ValueError(f"exploration_epsilon must be in [0, 1], got {exploration_epsilon}.")
        if force_winner is not None and not 0 <= int(force_winner) < expert_capitals.numel():
            raise ValueError(f"force_winner must be in [0, {expert_capitals.numel() - 1}], got {force_winner}.")

        with torch.no_grad():
            prices = self.compute_prices(q.detach(), temperature=price_temperature, liquidity_floor=liquidity_floor)
            capitals = expert_capitals.detach().to(dtype=reward_logits.dtype)
            prices = prices.to(device=reward_logits.device, dtype=reward_logits.dtype)
            stakes = capitals * float(bet_fraction)
            # Shares are priced once at batch-start and then reused for every token in the batch.
            shares = stakes / prices.clamp(min=1e-8)

            pred_reward = torch.sigmoid(reward_logits.detach())
            expected_profit = shares.view(1, 1, -1) * (pred_reward - prices.view(1, 1, -1))

            score = expected_profit
            if training and self.noise_std > 0:
                score = score + torch.randn_like(score) * self.noise_std

            winners = score.argmax(dim=-1)
            if training and exploration_epsilon > 0:
                exploration_mask = torch.rand_like(winners, dtype=torch.float32) < float(exploration_epsilon)
                random_winners = torch.randint(0, expert_capitals.numel(), winners.shape, device=winners.device)
                winners = torch.where(exploration_mask, random_winners, winners)
            else:
                exploration_mask = torch.zeros_like(winners, dtype=torch.bool)
            if force_winner is not None:
                winners = torch.full_like(winners, int(force_winner))
                exploration_mask = torch.zeros_like(winners, dtype=torch.bool)

        return {
            "winners": winners,
            "prices": prices,
            "stakes": stakes,
            "shares": shares,
            "score": score,
            "expected_profit": expected_profit,
            "pred_reward": pred_reward,
            "exploration_mask": exploration_mask,
        }
