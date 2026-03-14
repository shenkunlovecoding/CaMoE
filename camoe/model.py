"""Prediction-market CaMoE model."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import CaMoE_Block
from .capital import MarketStateManager
from .config import CONFIG_0_4B, CaMoEConfig
from .expert_critic import RewardCritic
from .expert_base import BaseExpert
from .expert_deepembed import DeepEmbedExpert, SlimDeepEmbedExpert
from .expert_fractal import FractalBlueprint, FractalCaMoEPlaceholder
from .expert_rosa import ROSAExpert
from .expert_rwkv import RWKVExpert
from .expert_timemix import TimeMixExpert


class CaMoE_Model(nn.Module):
    """Stacked dual-market CaMoE with prediction-market routing."""

    def __init__(self, config: CaMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.dim)

        self.blocks = nn.ModuleList([self._build_block(layer_idx) for layer_idx in range(config.n_layers)])

        self.ln_out = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.emb.weight

        self.sequence_capital_manager = (
            MarketStateManager(
                n_layers=config.n_layers,
                n_experts_per_layer=2,
                capital_init=config.expert_capital_init,
                ema_decay=config.ema_decay,
                capital_floor=config.capital_floor,
                capital_ceiling=config.capital_ceiling,
                price_lr=config.price_lr,
                price_temperature=config.price_temperature,
                liquidity_floor=config.liquidity_floor,
                exploration_epsilon=config.exploration_epsilon,
                reward_scale=config.reward_scale,
                reward_eps=config.reward_eps,
                bet_fraction=config.bet_fraction,
            )
            if config.n_rosa_experts > 0
            else None
        )
        self.ffn_capital_manager = MarketStateManager(
            n_layers=config.n_layers,
            n_experts_per_layer=config.total_ffn_experts,
            capital_init=config.expert_capital_init,
            ema_decay=config.ema_decay,
            capital_floor=config.capital_floor,
            capital_ceiling=config.capital_ceiling,
            price_lr=config.price_lr,
            price_temperature=config.price_temperature,
            liquidity_floor=config.liquidity_floor,
            exploration_epsilon=config.exploration_epsilon,
            reward_scale=config.reward_scale,
            reward_eps=config.reward_eps,
            bet_fraction=config.bet_fraction,
        )
        self.capital_manager = self.ffn_capital_manager
        self._last_settle_results: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        self._sync_all_capitals_to_experts()
        self._maybe_compile_modules()

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        critic_alpha: float = 1.0,
        ste_temperature: float | None = None,
        training: bool = True,
        uniform: bool = False,
        market_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        del critic_alpha
        batch, steps = input_ids.shape
        x = self.emb(input_ids)

        v_first = None
        for layer_idx, block in enumerate(self.blocks):
            seq_state = (
                self.sequence_capital_manager.route_state(layer_idx)
                if self.sequence_capital_manager is not None and block.has_sequence_market
                else None
            )
            ffn_state = self.ffn_capital_manager.route_state(layer_idx)
            x, v_first = block(
                x,
                v_first=v_first,
                ste_temperature=ste_temperature,
                training=training,
                uniform=uniform,
                market_weight=market_weight,
                sequence_state=seq_state,
                ffn_state=ffn_state,
                token_ids=input_ids,
            )

        x = self.ln_out(x)
        logits = self.lm_head(x)
        result: dict[str, torch.Tensor] = {"logits": logits}

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.ignore_index,
                reduction="none",
            ).view(batch, steps)
            valid_mask = targets.ne(self.config.ignore_index)
            valid_count = valid_mask.sum().clamp(min=1)
            result["loss"] = loss
            main_loss = (loss * valid_mask.to(loss.dtype)).sum() / valid_count
            result["loss_main_scalar"] = main_loss
            result["loss_scalar"] = main_loss
            result["loss_mask"] = valid_mask

        return result

    def settle_all_layers(
        self,
        token_loss: torch.Tensor,
        token_weight: torch.Tensor | None = None,
        update_state: bool = True,
    ) -> list[dict[str, torch.Tensor | int | str | dict[str, torch.Tensor]]]:
        all_results: list[dict[str, torch.Tensor | int | str | dict[str, torch.Tensor]]] = []
        self._last_settle_results = {}

        for layer_idx, block in enumerate(self.blocks):
            cache = block.get_cache()
            if not cache:
                continue

            seq_cache = cache.get("sequence", {})
            if self.sequence_capital_manager is not None and seq_cache:
                seq_result = self.sequence_capital_manager.settle_layer(
                    layer_idx=layer_idx,
                    winners=seq_cache["winners"],
                    token_loss=token_loss,
                    prices=seq_cache["prices"],
                    shares=seq_cache["shares"],
                    token_weight=token_weight,
                    update_state=update_state,
                )
                if update_state:
                    self.sequence_capital_manager.sync_to_experts(layer_idx, block.sequence_experts())
                packed = {
                    "layer": layer_idx,
                    "market": "sequence",
                    "cache": seq_cache,
                    **seq_result,
                }
                all_results.append(packed)
                self._last_settle_results[(layer_idx, "sequence")] = {
                    key: value.detach().clone() for key, value in seq_result.items()
                }

            ffn_cache = cache.get("ffn", {})
            if ffn_cache:
                ffn_result = self.ffn_capital_manager.settle_layer(
                    layer_idx=layer_idx,
                    winners=ffn_cache["winners"],
                    token_loss=token_loss,
                    prices=ffn_cache["prices"],
                    shares=ffn_cache["shares"],
                    token_weight=token_weight,
                    update_state=update_state,
                )
                if update_state:
                    self.ffn_capital_manager.sync_to_experts(layer_idx, list(block.experts))
                packed = {
                    "layer": layer_idx,
                    "market": "ffn",
                    "cache": ffn_cache,
                    **ffn_result,
                }
                all_results.append(packed)
                self._last_settle_results[(layer_idx, "ffn")] = {
                    key: value.detach().clone() for key, value in ffn_result.items()
                }

        return all_results

    def compute_critic_loss(
        self,
        settle_results: list[dict[str, torch.Tensor | int | str | dict[str, torch.Tensor]]],
        critic_alpha: float = 1.0,
        token_weight: torch.Tensor | None = None,
        entropy_reg: float | None = None,
    ) -> torch.Tensor:
        del critic_alpha, entropy_reg
        total_loss: torch.Tensor | None = None
        count = 0
        for result in settle_results:
            layer_idx = int(result["layer"])
            market = str(result["market"])
            block = self.blocks[layer_idx]
            reward_critic = block.sequence_reward_critic if market == "sequence" else block.ffn_reward_critic
            if reward_critic is None:
                continue
            cache = result["cache"]
            loss = reward_critic.supervised_loss(
                cache["x_detached"],
                cache["winners"],
                result["realized_reward"].detach(),
                token_weight=token_weight,
            )
            total_loss = loss if total_loss is None else (total_loss + loss)
            count += 1

        if total_loss is None:
            return self.emb.weight.new_zeros(())
        return total_loss / max(count, 1)

    def get_market_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
        for layer_idx, block in enumerate(self.blocks):
            cache = block.get_cache()
            if not cache:
                continue

            seq_cache = cache.get("sequence", {})
            if self.sequence_capital_manager is not None and seq_cache:
                diagnostics.append(
                    {
                        "layer": layer_idx,
                        "market": "sequence",
                        "expert_types": [expert.expert_type for expert in block.sequence_experts()],
                        "capitals": self.sequence_capital_manager.capitals[layer_idx].detach().clone(),
                        "q": self.sequence_capital_manager.q[layer_idx].detach().clone(),
                        "loss_ema": self.sequence_capital_manager.loss_ema[layer_idx].detach().clone(),
                        "prices": self.sequence_capital_manager.prices(layer_idx).detach().clone(),
                        "cache": {key: value.detach().clone() for key, value in seq_cache.items()},
                    }
                )

            ffn_cache = cache.get("ffn", {})
            if ffn_cache:
                diagnostics.append(
                    {
                        "layer": layer_idx,
                        "market": "ffn",
                        "expert_types": [expert.expert_type for expert in block.experts],
                        "capitals": self.ffn_capital_manager.capitals[layer_idx].detach().clone(),
                        "q": self.ffn_capital_manager.q[layer_idx].detach().clone(),
                        "loss_ema": self.ffn_capital_manager.loss_ema[layer_idx].detach().clone(),
                        "prices": self.ffn_capital_manager.prices(layer_idx).detach().clone(),
                        "cache": {key: value.detach().clone() for key, value in ffn_cache.items()},
                    }
                )
        return diagnostics

    def market_metrics(
        self,
        critic_alpha: float = 1.0,
        token_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        del critic_alpha
        metrics: dict[str, float] = {}
        diagnostics = self.get_market_diagnostics()
        for entry in diagnostics:
            layer_idx = int(entry["layer"])
            market = str(entry.get("market", "ffn"))
            caps = entry["capitals"].float()
            q = entry["q"].float()
            prices = entry["prices"].float()
            prefix = f"{market}/layer_{layer_idx}"

            metrics[f"{prefix}/wallet_mean"] = float(caps.mean().item())
            metrics[f"{prefix}/wallet_min"] = float(caps.min().item())
            metrics[f"{prefix}/wallet_max"] = float(caps.max().item())
            metrics[f"{prefix}/wallet_gini"] = float(self._gini(caps).item())
            metrics[f"{prefix}/price_min"] = float(prices.min().item())
            metrics[f"{prefix}/price_max"] = float(prices.max().item())

            cache = entry["cache"]
            winners = cache["winners"]
            pred_reward = cache.get("pred_reward")
            expected_profit = cache.get("expected_profit")

            answer_mask = self._resolve_mask(token_mask, winners)
            active_mask = self._resolve_mask(valid_mask, winners)
            if active_mask is None:
                active_mask = torch.ones_like(winners, dtype=torch.float32)
            answer_mask = answer_mask * active_mask if answer_mask is not None else active_mask

            winner_hist_all = self._winner_histogram(winners, caps.numel(), active_mask)
            winner_hist_answer = self._winner_histogram(winners, caps.numel(), answer_mask)
            entropy_all = self._entropy(winner_hist_all)
            entropy_answer = self._entropy(winner_hist_answer)
            metrics[f"{prefix}/routing_entropy"] = float(entropy_all.item())
            metrics[f"{prefix}/routing_entropy_answer"] = float(entropy_answer.item())

            if pred_reward is not None:
                chosen_pred = torch.gather(pred_reward, dim=-1, index=winners.unsqueeze(-1)).squeeze(-1)
                metrics[f"{prefix}/pred_reward_mean"] = float(self._masked_mean(chosen_pred, active_mask).item())

            if expected_profit is not None:
                chosen_profit = torch.gather(expected_profit, dim=-1, index=winners.unsqueeze(-1)).squeeze(-1)
                metrics[f"{prefix}/expected_profit_mean"] = float(self._masked_mean(chosen_profit, active_mask).item())
            exploration_mask = cache.get("exploration_mask")
            if exploration_mask is not None:
                metrics[f"{prefix}/exploration_rate"] = float(exploration_mask.float().mean().item())

            settlement = self._last_settle_results.get((layer_idx, market))
            if settlement is not None:
                metrics[f"{prefix}/realized_reward_mean"] = float(
                    self._masked_mean(settlement["realized_reward"], active_mask).item()
                )
                metrics[f"{prefix}/token_profit_mean"] = float(
                    self._masked_mean(settlement["token_profit"], active_mask).item()
                )

            expert_types = list(entry["expert_types"])
            type_to_indices: dict[str, list[int]] = {}
            for expert_idx, expert_type in enumerate(expert_types):
                type_to_indices.setdefault(expert_type, []).append(expert_idx)
                metrics[f"{prefix}/expert_{expert_idx}/wallet"] = float(caps[expert_idx].item())
                metrics[f"{prefix}/expert_{expert_idx}/price"] = float(prices[expert_idx].item())
                metrics[f"{prefix}/expert_{expert_idx}/q"] = float(q[expert_idx].item())
                metrics[f"{prefix}/expert_{expert_idx}/winner_share"] = float(winner_hist_all[expert_idx].item())
                metrics[f"{prefix}/expert_{expert_idx}/winner_share_answer"] = float(
                    winner_hist_answer[expert_idx].item()
                )

            for expert_type, indices in type_to_indices.items():
                idx_tensor = torch.tensor(indices, device=caps.device, dtype=torch.long)
                metrics[f"{prefix}/wallet_{expert_type}"] = float(caps.index_select(0, idx_tensor).mean().item())
                metrics[f"{prefix}/price_{expert_type}"] = float(prices.index_select(0, idx_tensor).mean().item())
                metrics[f"{prefix}/winner_share_{expert_type}"] = float(
                    winner_hist_answer.index_select(0, idx_tensor).mean().item()
                )

            if market == "ffn":
                metrics[f"layer_{layer_idx}/wallet_mean"] = metrics[f"{prefix}/wallet_mean"]
                metrics[f"layer_{layer_idx}/wallet_min"] = metrics[f"{prefix}/wallet_min"]
                metrics[f"layer_{layer_idx}/wallet_max"] = metrics[f"{prefix}/wallet_max"]
                metrics[f"layer_{layer_idx}/wallet_gini"] = metrics[f"{prefix}/wallet_gini"]
                metrics[f"layer_{layer_idx}/price_max"] = metrics[f"{prefix}/price_max"]
                metrics[f"layer_{layer_idx}/routing_entropy"] = metrics[f"{prefix}/routing_entropy"]
                metrics[f"L{layer_idx}/Gini"] = metrics[f"{prefix}/wallet_gini"]

        return metrics

    def _build_block(self, layer_idx: int) -> CaMoE_Block:
        reward_hidden = self.config.reward_hidden_dim or self.config.critic_hidden_dim
        timemix_expert = TimeMixExpert(
            dim=self.config.dim,
            n_layers=self.config.n_layers,
            layer_idx=layer_idx,
            head_size=self.config.head_size,
            capital_init=self.config.expert_capital_init,
            capital_floor=self.config.capital_floor,
            capital_ceiling=self.config.capital_ceiling,
        )
        rosa_expert = None
        sequence_reward_critic = None
        if self.config.n_rosa_experts > 0:
            rosa_expert = ROSAExpert(
                dim=self.config.dim,
                slim_heads=self.config.effective_slim_rosa_heads,
                bits_per_symbol=self.config.rosa_bits,
                backend=self.config.rosa_backend,
                truncation_length=self.config.rosa_truncation_length,
                sequence_length=self.config.seq_len,
                capital_init=self.config.expert_capital_init,
                capital_floor=self.config.capital_floor,
                capital_ceiling=self.config.capital_ceiling,
            )
            sequence_reward_critic = RewardCritic(
                dim=self.config.dim,
                n_routable=2,
                hidden_dim=reward_hidden,
            )

        ffn_experts: list[BaseExpert] = []
        for _ in range(self.config.n_experts):
            ffn_experts.append(
                RWKVExpert(
                    dim=self.config.dim,
                    expand=self.config.ffn_expand,
                    capital_init=self.config.expert_capital_init,
                    capital_floor=self.config.capital_floor,
                    capital_ceiling=self.config.capital_ceiling,
                )
            )
        for _ in range(self.config.n_deepembed_experts):
            ffn_experts.append(
                DeepEmbedExpert(
                    vocab_size=self.config.vocab_size,
                    dim=self.config.dim,
                    expand=self.config.deepembed_expand,
                    mode=self.config.deepembed_mode,
                    capital_init=self.config.expert_capital_init,
                    capital_floor=self.config.capital_floor,
                    capital_ceiling=self.config.capital_ceiling,
                )
            )
        for _ in range(self.config.n_slim_deepembed_experts):
            ffn_experts.append(
                SlimDeepEmbedExpert(
                    dim=self.config.dim,
                    rank=self.config.slim_deepembed_rank,
                    expand=self.config.deepembed_expand,
                    capital_init=self.config.expert_capital_init,
                    capital_floor=self.config.capital_floor,
                    capital_ceiling=self.config.capital_ceiling,
                )
            )
        ffn_reward_critic = RewardCritic(
            dim=self.config.dim,
            n_routable=self.config.total_ffn_experts,
            hidden_dim=reward_hidden,
        )
        return CaMoE_Block(
            timemix_expert=timemix_expert,
            rosa_expert=rosa_expert,
            ffn_experts=ffn_experts,
            sequence_reward_critic=sequence_reward_critic,
            ffn_reward_critic=ffn_reward_critic,
            routing_noise_std=self.config.routing_noise_std,
            use_gradient_checkpointing=self.config.enable_gradient_checkpointing,
            use_routing_ste=self.config.routing_ste,
            ste_temperature=self.config.ste_temperature_end,
        )

    def _maybe_compile_modules(self) -> None:
        if not self.config.enable_compile or not hasattr(torch, "compile"):
            return

        compile_mode = self.config.compile_mode
        for block in self.blocks:
            self._compile_forward(block.timemix_expert, compile_mode)
            self._compile_forward(block.rosa_expert, compile_mode)
            for expert in block.ffn_experts:
                self._compile_forward(expert, compile_mode)
            self._compile_forward(block.sequence_reward_critic, compile_mode)
            self._compile_forward(block.ffn_reward_critic, compile_mode)

    @staticmethod
    def _compile_forward(module: nn.Module | None, compile_mode: str) -> None:
        if module is None:
            return
        try:
            original_forward = module.forward
            compiled_forward = torch.compile(original_forward, mode=compile_mode)

            def compiled_with_fallback(*args, **kwargs):
                try:
                    return compiled_forward(*args, **kwargs)
                except Exception:
                    module.forward = original_forward
                    return original_forward(*args, **kwargs)

            module.forward = compiled_with_fallback
        except Exception:
            return

    def _sync_all_capitals_to_experts(self) -> None:
        for layer_idx, block in enumerate(self.blocks):
            if self.sequence_capital_manager is not None and block.has_sequence_market:
                self.sequence_capital_manager.sync_to_experts(layer_idx, block.sequence_experts())
            self.ffn_capital_manager.sync_to_experts(layer_idx, list(block.experts))

    def get_checkpoint_config(self) -> dict:
        return asdict(self.config)

    def make_fractal_blueprint(
        self,
        *,
        depth: int = 0,
        slot_name: str = "fractal",
        notes: str = "placeholder",
    ) -> FractalBlueprint:
        return FractalBlueprint(
            depth=int(depth),
            max_depth=int(self.config.fractal_max_depth),
            slot_name=slot_name,
            child_kind="camoe",
            notes=notes,
        )

    def make_fractal_child_config(self, depth: int) -> CaMoEConfig:
        child = self.config.copy()
        child.fractal_max_depth = max(int(self.config.fractal_max_depth) - int(depth), 0)
        return child

    def make_fractal_placeholder(
        self,
        *,
        depth: int = 0,
        slot_name: str = "fractal",
        notes: str = "placeholder",
        fallback_mode: str = "zero",
    ) -> FractalCaMoEPlaceholder:
        return FractalCaMoEPlaceholder(
            dim=self.config.dim,
            blueprint=self.make_fractal_blueprint(depth=depth, slot_name=slot_name, notes=notes),
            expansion_right_cost=self.config.fractal_expansion_cost,
            child_config=self.make_fractal_child_config(depth + 1),
            fallback_mode=fallback_mode,
            capital_init=self.config.expert_capital_init,
            capital_floor=self.config.capital_floor,
            capital_ceiling=self.config.capital_ceiling,
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._sync_all_capitals_to_experts()
        return result

    @staticmethod
    def _resolve_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
        if mask is None:
            return None
        resolved = torch.as_tensor(mask, device=reference.device, dtype=torch.float32)
        if resolved.shape != reference.shape:
            raise ValueError(f"Mask shape {tuple(resolved.shape)} does not match {tuple(reference.shape)}")
        return resolved

    @staticmethod
    def _winner_histogram(
        winners: torch.Tensor,
        num_experts: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        one_hot = F.one_hot(winners, num_classes=num_experts).to(dtype=torch.float32)
        weighted = one_hot * mask.unsqueeze(-1)
        denom = mask.sum().clamp(min=1e-8)
        return weighted.sum(dim=(0, 1)) / denom

    @staticmethod
    def _entropy(prob: torch.Tensor) -> torch.Tensor:
        return -(prob * torch.log(prob + 1e-9)).sum()

    @staticmethod
    def _gini(values: torch.Tensor) -> torch.Tensor:
        sorted_values, _ = torch.sort(values.float())
        n = sorted_values.numel()
        idx = torch.arange(1, n + 1, device=sorted_values.device, dtype=sorted_values.dtype)
        return ((2 * idx - n - 1) * sorted_values).sum() / (n * sorted_values.sum().clamp(min=1e-6))

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = torch.as_tensor(mask, device=values.device, dtype=values.dtype)
        denom = weights.sum().clamp(min=1e-8)
        return (values * weights).sum() / denom


def load_camoe_checkpoint(
    checkpoint_path: str,
    device: str | torch.device = "cpu",
    config: CaMoEConfig | dict | None = None,
    strict: bool = True,
) -> tuple[CaMoE_Model, CaMoEConfig, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if config is None and isinstance(checkpoint, dict) and checkpoint.get("config") is not None:
        config = CaMoEConfig.from_mapping(checkpoint["config"])
    elif config is None:
        config = CONFIG_0_4B.copy()
    elif not isinstance(config, CaMoEConfig):
        config = CaMoEConfig.from_mapping(config)

    model = CaMoE_Model(config)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    return model, config, checkpoint
