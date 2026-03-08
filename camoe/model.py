"""Pure-market CaMoE model."""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import RWKV7_TimeMix, SharedDeepEmbed
from .block import CaMoE_Block
from .capital import ExpertCapitalManager
from .config import CONFIG_0_4B, CaMoEConfig
from .expert_critic import CriticPair
from .expert_rwkv import RWKVExpert


class CaMoE_Model(nn.Module):
    """Stacked RWKV backbone plus pure-market expert blocks."""

    def __init__(self, config: CaMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.dim)
        self.deep_embed = SharedDeepEmbed(config.vocab_size, config.dim) if config.use_deep_embed else None
        self.backbone_norms = nn.ModuleList([nn.LayerNorm(config.dim) for _ in range(config.n_layers)])
        self.backbone_layers = nn.ModuleList(
            [
                RWKV7_TimeMix(
                    n_embd=config.dim,
                    n_layer=config.n_layers,
                    layer_id=layer_idx,
                    head_size=config.head_size,
                )
                for layer_idx in range(config.n_layers)
            ]
        )

        self.blocks = nn.ModuleList()
        for _layer_idx in range(config.n_layers):
            experts = [
                RWKVExpert(
                    dim=config.dim,
                    expand=config.ffn_expand,
                    capital_init=config.expert_capital_init,
                    capital_floor=config.capital_floor,
                )
                for _ in range(config.n_experts)
            ]
            critic_pair = CriticPair(
                dim=config.dim,
                n_routable=config.n_experts,
                hidden_dim=config.critic_hidden_dim,
                capital_init=config.critic_capital_init,
                capital_floor=config.capital_floor,
            )
            self.blocks.append(
                CaMoE_Block(
                    experts=experts,
                    critic_pair=critic_pair,
                    top_k=config.top_k,
                    auction_noise=config.auction_noise_std,
                    use_gradient_checkpointing=config.enable_gradient_checkpointing,
                )
            )

        self.ln_out = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.emb.weight

        self.capital_manager = ExpertCapitalManager(
            n_layers=config.n_layers,
            n_experts_per_layer=config.n_experts,
            capital_init=config.expert_capital_init,
            ema_decay=config.ema_decay,
            capital_floor=config.capital_floor,
            depreciation=config.depreciation,
        )
        self._sync_all_capitals_to_experts()
        self._maybe_compile_modules()

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        critic_alpha: float = 1.0,
        training: bool = True,
        uniform: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch, steps = input_ids.shape
        x = self.emb(input_ids)
        if self.deep_embed is not None:
            x = x + self.config.deep_embed_scale * self.deep_embed(input_ids)
        v_first = None

        for layer_idx in range(self.config.n_layers):
            att_out, v_first = self._forward_backbone_layer(
                x,
                v_first,
                layer_idx,
                v_first is not None,
            )
            x = x + att_out
            x = x + self.blocks[layer_idx](
                x,
                critic_alpha=critic_alpha,
                training=training,
                uniform=uniform,
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
            result["loss"] = loss
            result["loss_scalar"] = loss.mean()

        return result

    def settle_all_layers(
        self,
        token_loss: torch.Tensor,
    ) -> list[dict[str, torch.Tensor | int | dict[str, torch.Tensor]]]:
        all_profits: list[dict[str, torch.Tensor | int | dict[str, torch.Tensor]]] = []
        for layer_idx, block in enumerate(self.blocks):
            cache = block.get_cache()
            if not cache:
                continue

            profit = self.capital_manager.settle_layer(
                layer_idx=layer_idx,
                winners=cache["winners"],
                weights=cache["weights"],
                token_loss=token_loss,
                prices=cache["prices"],
            )
            adv_a, adv_b = block.critic_pair.settle_both(cache["x_detached"], profit)
            self.capital_manager.sync_to_experts(layer_idx, list(block.experts))

            all_profits.append(
                {
                    "layer": layer_idx,
                    "profit": profit,
                    "adv_a": adv_a,
                    "adv_b": adv_b,
                    "cache": cache,
                }
            )

        return all_profits

    def compute_critic_loss(
        self,
        settle_results: list[dict[str, torch.Tensor | int | dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        total_loss: torch.Tensor | None = None
        count = 0
        for result in settle_results:
            layer_idx = int(result["layer"])
            block = self.blocks[layer_idx]
            cache = result["cache"]
            loss = block.critic_pair.reinforce_loss(
                cache["x_detached"],
                result["profit"].detach(),
                result["adv_a"],
                result["adv_b"],
            )
            total_loss = loss if total_loss is None else (total_loss + loss)
            count += 1

        if total_loss is None:
            return self.emb.weight.new_zeros(())
        return total_loss / max(count, 1)

    def market_metrics(self, critic_alpha: float = 1.0) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for layer_idx, block in enumerate(self.blocks):
            caps = self.capital_manager.capitals[layer_idx]
            metrics[f"layer_{layer_idx}/capital_mean"] = float(caps.mean().item())
            metrics[f"layer_{layer_idx}/capital_min"] = float(caps.min().item())
            metrics[f"layer_{layer_idx}/capital_max"] = float(caps.max().item())
            sorted_caps, _ = torch.sort(caps.float())
            n = sorted_caps.numel()
            idx = torch.arange(1, n + 1, device=sorted_caps.device, dtype=sorted_caps.dtype)
            gini = ((2 * idx - n - 1) * sorted_caps).sum() / (n * sorted_caps.sum().clamp(min=1e-6))
            critic_cap = (
                block.critic_pair.critic_a.capital.float() + block.critic_pair.critic_b.capital.float()
            ) / 2.0
            metrics[f"L{layer_idx}/Gini"] = float(gini.item())
            metrics[f"L{layer_idx}/CriticCap"] = float(critic_cap.item())
            metrics[f"L{layer_idx}/MarketAlpha"] = float(critic_alpha)
            cache = block.get_cache()
            if cache:
                weights = cache["weights"].float()
                entropy = -(weights * torch.log(weights + 1e-9)).sum(dim=-1).mean()
                metrics[f"layer_{layer_idx}/routing_entropy"] = float(entropy.item())
                metrics[f"L{layer_idx}/WinnerFromAdjustedEntropy"] = float(entropy.item())
                metrics[f"L{layer_idx}/WeightEntropy"] = float(entropy.item())
        return metrics

    def _forward_backbone_layer(
        self,
        x: torch.Tensor,
        v_first_in: torch.Tensor | None,
        layer_idx: int,
        has_v_first: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.backbone_norms[layer_idx](x)
        v_first = v_first_in if has_v_first else None
        att_out, next_v_first, _ = self.backbone_layers[layer_idx](normed, v_first)
        if next_v_first is None:
            next_v_first = v_first_in if v_first_in is not None else x.new_zeros(x.shape)
        return att_out, next_v_first

    def _maybe_compile_modules(self) -> None:
        if not self.config.enable_compile or not hasattr(torch, "compile"):
            return

        compile_mode = self.config.compile_mode
        self._compile_forward(self.deep_embed, compile_mode)
        for layer in self.backbone_layers:
            self._compile_forward(layer, compile_mode)
        for block in self.blocks:
            for expert in block.experts:
                self._compile_forward(expert, compile_mode)
            self._compile_forward(block.critic_pair.critic_a.position_net, compile_mode)
            self._compile_forward(block.critic_pair.critic_b.position_net, compile_mode)

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
            self.capital_manager.sync_to_experts(layer_idx, list(block.experts))

    def get_checkpoint_config(self) -> dict:
        return asdict(self.config)

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._sync_all_capitals_to_experts()
        return result


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
