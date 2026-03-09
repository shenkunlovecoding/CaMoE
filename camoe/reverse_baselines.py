"""Small reverse-digits baselines built on top of the CaMoE backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import RWKV7_TimeMix
from .config import CaMoEConfig
from .expert_rosa import ROSAExpert
from .expert_rwkv import RWKVExpert


class _SingleExpertToyModel(nn.Module):
    """A tiny RWKV backbone with one residual expert branch per layer."""

    def __init__(self, config: CaMoEConfig, experts: list[nn.Module]) -> None:
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.dim)
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
        self.expert_layers = nn.ModuleList(experts)
        self.ln_out = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, steps = input_ids.shape
        x = self.emb(input_ids)
        v_first = None

        for layer_idx in range(self.config.n_layers):
            normed = self.backbone_norms[layer_idx](x)
            att_out, v_first, _state = self.backbone_layers[layer_idx](normed, v_first)
            x = x + att_out
            x = x + self.expert_layers[layer_idx](x)

        x = self.ln_out(x)
        logits = self.lm_head(x)
        result: dict[str, torch.Tensor] = {"logits": logits}

        if targets is None:
            return result

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.config.ignore_index,
            reduction="none",
        ).view(batch, steps)
        valid_mask = targets.ne(self.config.ignore_index)
        valid_count = valid_mask.sum().clamp(min=1)
        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets) & valid_mask

        result["loss"] = loss
        result["loss_mask"] = valid_mask
        result["loss_scalar"] = (loss * valid_mask.to(loss.dtype)).sum() / valid_count
        result["token_acc_masked"] = correct.sum().to(loss.dtype) / valid_count
        result["exact_match_masked"] = correct.logical_or(~valid_mask).all(dim=1).float().mean()
        return result


class SingleROSAReverseModel(_SingleExpertToyModel):
    """A tiny RWKV backbone with one ROSA residual branch per layer."""

    def __init__(self, config: CaMoEConfig) -> None:
        experts = [
            ROSAExpert(
                dim=config.dim,
                slim_heads=config.effective_slim_rosa_heads,
                bits_per_symbol=config.rosa_bits,
                backend=config.rosa_backend,
                truncation_length=config.rosa_truncation_length,
                sequence_length=config.seq_len,
                capital_init=config.expert_capital_init,
                capital_floor=config.capital_floor,
                capital_ceiling=config.capital_ceiling,
            )
            for _ in range(config.n_layers)
        ]
        super().__init__(config, experts)


class SingleRWKVReverseModel(_SingleExpertToyModel):
    """A tiny RWKV backbone with one RWKV FFN residual branch per layer."""

    def __init__(self, config: CaMoEConfig) -> None:
        experts = [
            RWKVExpert(
                dim=config.dim,
                expand=config.ffn_expand,
                capital_init=config.expert_capital_init,
                capital_floor=config.capital_floor,
                capital_ceiling=config.capital_ceiling,
            )
            for _ in range(config.n_layers)
        ]
        super().__init__(config, experts)
