"""CaMoE block with prediction-market routing."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .auction import PredictionMarketRouter
from .expert_base import BaseExpert
from .expert_critic import RewardCritic
from .expert_rosa import ROSAExpert
from .expert_timemix import TimeMixExpert


class CaMoE_Block(nn.Module):
    """Single layer with sequence-market routing and FFN-market routing."""

    def __init__(
        self,
        timemix_expert: TimeMixExpert,
        rosa_expert: ROSAExpert | None,
        ffn_experts: list[BaseExpert],
        sequence_reward_critic: RewardCritic | None,
        ffn_reward_critic: RewardCritic,
        routing_noise_std: float = 0.01,
        use_gradient_checkpointing: bool = True,
        use_routing_ste: bool = True,
        ste_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.timemix_expert = timemix_expert
        self.rosa_expert = rosa_expert
        self.ffn_experts = nn.ModuleList(ffn_experts)
        self.experts = self.ffn_experts
        self.sequence_reward_critic = sequence_reward_critic
        self.ffn_reward_critic = ffn_reward_critic
        self.n_routable = len(ffn_experts)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_routing_ste = bool(use_routing_ste)
        self.ste_temperature = float(ste_temperature)
        self.sequence_router = PredictionMarketRouter(noise_std=routing_noise_std)
        self.ffn_router = PredictionMarketRouter(noise_std=routing_noise_std)
        self._cache: dict[str, dict[str, torch.Tensor]] = {}

    @property
    def has_sequence_market(self) -> bool:
        return self.rosa_expert is not None and self.sequence_reward_critic is not None

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None = None,
        critic_alpha: float = 1.0,
        ste_temperature: float | None = None,
        training: bool = True,
        uniform: bool = False,
        market_weight: float = 1.0,
        sequence_state: dict[str, torch.Tensor | float] | None = None,
        ffn_state: dict[str, torch.Tensor | float] | None = None,
        **expert_ctx,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del critic_alpha
        ste_temp = self.ste_temperature if ste_temperature is None else float(ste_temperature)
        seq_out, next_v_first, seq_cache = self._forward_sequence_market(
            x,
            v_first=v_first,
            ste_temperature=ste_temp,
            training=training,
            uniform=uniform,
            market_weight=market_weight,
            sequence_state=sequence_state,
            **expert_ctx,
        )
        x = x + seq_out

        ffn_out, ffn_cache = self._forward_ffn_market(
            x,
            ste_temperature=ste_temp,
            training=training,
            uniform=uniform,
            market_weight=market_weight,
            ffn_state=ffn_state,
            **expert_ctx,
        )
        x = x + ffn_out

        self._cache = {
            "sequence": seq_cache,
            "ffn": ffn_cache,
        }
        return x, next_v_first

    def _forward_sequence_market(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        ste_temperature: float,
        training: bool,
        uniform: bool,
        market_weight: float,
        sequence_state: dict[str, torch.Tensor | float] | None,
        **expert_ctx,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        timemix_out, next_v_first, timemix_state = self.timemix_expert(
            x,
            v_first=v_first,
            **expert_ctx,
        )
        if not self.has_sequence_market:
            return timemix_out, next_v_first, {}

        assert self.rosa_expert is not None
        assert self.sequence_reward_critic is not None
        rosa_out = self.rosa_expert(x, **expert_ctx)
        if uniform:
            return (timemix_out + rosa_out) * 0.5, next_v_first, {}

        if sequence_state is None:
            raise ValueError("sequence_state is required when sequence prediction-market routing is enabled.")

        reward_logits = self.sequence_reward_critic(x.detach())
        route = self.sequence_router(
            expert_capitals=sequence_state["capital"],
            q=sequence_state["q"],
            reward_logits=reward_logits,
            bet_fraction=float(sequence_state["bet_fraction"]),
            price_temperature=float(sequence_state["price_temperature"]),
            liquidity_floor=float(sequence_state["liquidity_floor"]),
            exploration_epsilon=float(sequence_state["exploration_epsilon"]),
            force_winner=sequence_state.get("force_winner"),
            training=training,
        )

        winner_mask = route["winners"].unsqueeze(-1).eq(1)
        hard_output = torch.where(winner_mask, rosa_out, timemix_out)
        if training and self.use_routing_ste:
            probs = torch.softmax(route["score"] / ste_temperature, dim=-1)
            soft_output = timemix_out * probs[:, :, 0:1] + rosa_out * probs[:, :, 1:2]
            market_output = soft_output + (hard_output - soft_output).detach()
        else:
            market_output = hard_output

        if market_weight < 1.0:
            uniform_output = (timemix_out + rosa_out) * 0.5
            output = market_weight * market_output + (1.0 - market_weight) * uniform_output
        else:
            output = market_output

        cache = {
            **{key: value.detach() for key, value in route.items()},
            "x_detached": x.detach(),
            "timemix_state": timemix_state.detach(),
        }
        return output, next_v_first, cache

    def _forward_ffn_market(
        self,
        x: torch.Tensor,
        ste_temperature: float,
        training: bool,
        uniform: bool,
        market_weight: float,
        ffn_state: dict[str, torch.Tensor | float] | None,
        **expert_ctx,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if uniform:
            return self._forward_uniform_ffn(x, **expert_ctx), {}

        if ffn_state is None:
            raise ValueError("ffn_state is required when FFN prediction-market routing is enabled.")

        reward_logits = self.ffn_reward_critic(x.detach())
        route = self.ffn_router(
            expert_capitals=ffn_state["capital"],
            q=ffn_state["q"],
            reward_logits=reward_logits,
            bet_fraction=float(ffn_state["bet_fraction"]),
            price_temperature=float(ffn_state["price_temperature"]),
            liquidity_floor=float(ffn_state["liquidity_floor"]),
            exploration_epsilon=float(ffn_state["exploration_epsilon"]),
            force_winner=ffn_state.get("force_winner"),
            training=training,
        )

        if training and self.use_routing_ste:
            market_output = self._dispatch_ffn_ste(
                x,
                route["winners"],
                route["score"],
                ste_temperature=ste_temperature,
                **expert_ctx,
            )
        elif self.training and self.use_gradient_checkpointing:
            market_output = checkpoint(
                self._dispatch_ffn,
                x,
                route["winners"],
                use_reentrant=False,
                **expert_ctx,
            )
        else:
            market_output = self._dispatch_ffn(x, route["winners"], **expert_ctx)

        if market_weight < 1.0:
            uniform_output = self._forward_uniform_ffn(x, **expert_ctx)
            output = market_weight * market_output + (1.0 - market_weight) * uniform_output
        else:
            output = market_output

        cache = {
            **{key: value.detach() for key, value in route.items()},
            "x_detached": x.detach(),
        }
        return output, cache

    def _dispatch_ffn(
        self,
        x: torch.Tensor,
        winners: torch.Tensor,
        **expert_ctx,
    ) -> torch.Tensor:
        batch, steps, dim = x.shape
        flat_x = x.reshape(batch * steps, dim)
        flat_winners = winners.reshape(batch * steps)
        total_tokens = batch * steps
        flat_ctx = self._flatten_sparse_ctx(batch, steps, expert_ctx)
        output = flat_x.new_zeros(flat_x.shape)

        for expert_idx, expert in enumerate(self.ffn_experts):
            token_indices = (flat_winners == expert_idx).nonzero(as_tuple=False).squeeze(-1)
            if token_indices.numel() == 0:
                continue

            if expert.supports_sparse_dispatch:
                expert_input = flat_x.index_select(0, token_indices)
                expert_out = expert(
                    expert_input,
                    **self._select_sparse_ctx(flat_ctx, token_indices, total_tokens),
                )
                output.index_copy_(0, token_indices, expert_out)
                continue

            full_out = expert(x, **expert_ctx).reshape(batch * steps, dim)
            selected_out = full_out.index_select(0, token_indices)
            output.index_copy_(0, token_indices, selected_out)

        return output.view(batch, steps, dim)

    def _forward_uniform_ffn(self, x: torch.Tensor, **expert_ctx) -> torch.Tensor:
        outputs = [expert(x, **expert_ctx) for expert in self.ffn_experts]
        return sum(outputs) / len(outputs)

    def _dispatch_ffn_ste(
        self,
        x: torch.Tensor,
        winners: torch.Tensor,
        score: torch.Tensor,
        ste_temperature: float,
        **expert_ctx,
    ) -> torch.Tensor:
        outputs = [expert(x, **expert_ctx) for expert in self.ffn_experts]
        all_out = torch.stack(outputs, dim=2)
        hard_index = winners.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, x.size(-1))
        hard_out = torch.gather(all_out, dim=2, index=hard_index).squeeze(2)
        probs = torch.softmax(score / ste_temperature, dim=-1).unsqueeze(-1)
        soft_out = (all_out * probs).sum(dim=2)
        return soft_out + (hard_out - soft_out).detach()

    def sequence_experts(self) -> list[BaseExpert]:
        if self.rosa_expert is None:
            return [self.timemix_expert]
        return [self.timemix_expert, self.rosa_expert]

    def get_cache(self) -> dict[str, dict[str, torch.Tensor]]:
        return self._cache

    @staticmethod
    def _flatten_sparse_ctx(
        batch: int,
        steps: int,
        expert_ctx: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        flat_ctx: dict[str, torch.Tensor] = {}
        token_count = batch * steps
        for key, value in expert_ctx.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and tuple(value.shape[:2]) == (batch, steps):
                flat_ctx[key] = value.reshape(token_count, *value.shape[2:])
            else:
                flat_ctx[key] = value
        return flat_ctx

    @staticmethod
    def _select_sparse_ctx(
        expert_ctx: dict[str, torch.Tensor],
        token_indices: torch.Tensor,
        total_tokens: int,
    ) -> dict[str, torch.Tensor]:
        selected: dict[str, torch.Tensor] = {}
        for key, value in expert_ctx.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.size(0) == total_tokens:
                selected[key] = value.index_select(0, token_indices)
            else:
                selected[key] = value
        return selected
