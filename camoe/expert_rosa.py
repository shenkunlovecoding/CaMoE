"""Slim Wind ROSA sequence expert (Advanced Neuro-Symbolic Edition)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert_base import BaseExpert
from .config import normalize_rosa_backend
from .rosa_soft_adapter import rosa_soft
from .soft_rosa_adapter import (
    soft_rosa_exact,
    soft_rosa_exact_with_aux,
    soft_rosa_qkv1bit,
    soft_rosa_qkv1bit_with_aux,
)
from .wind_rosa_adapter import wind_rosa


class StraightThroughSign(torch.autograd.Function):
    """
    STE for binarizing symbolic channels.

    Forward emits hard +/-1 values for the symbolic matcher.
    Backward uses the sigmoid derivative so the symbolic branch stays trainable.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.where(x > 0, torch.ones_like(x), torch.full_like(x, -1.0))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        sig = torch.sigmoid(x)
        return grad_output * sig * (1.0 - sig) * 4.0


class ROSAExpert(BaseExpert):
    """
    Sequence-aware Slim Wind ROSA expert.

    Native mode follows a Pengbo-style minimal ROSA block:
    LayerNorm -> token shift -> q/k/v projections -> ROSA -> output projection.

    Non-native mode keeps an extra continuous fusion branch on top of the same
    token-shifted symbolic source.
    """

    _SUPPORTED_BACKENDS = {
        "hard_symbolic_multibit",
        "soft_match",
        "soft_suffix",
        "soft_suffix_scan",
        "soft_exact_dp",
        "soft_exact_dp_serial",
        "soft_exact_dp_cuda",
        "soft_exact_dp_triton",
        "soft_qkv_multibit",
        "soft_qkv_multibit_serial",
        "soft_qkv_multibit_cuda",
        "soft_qkv_multibit_triton",
        "soft_qkv_multibit_unmatched",
        "soft_qkv_multibit_unmatched_serial",
        "soft_qkv_multibit_unmatched_cuda",
        "soft_qkv_multibit_unmatched_triton",
        "hard_qkv_multibit",
        "soft_qkv_binary",
        "soft_qkv_binary_bipolar",
        "soft_qkv_binary_bipolar_cuda",
        "soft_qkv_binary_bipolar_triton",
        "soft_qkv_binary_triton",
        "soft_qkv_binary_cuda",
    }

    def __init__(
        self,
        dim: int,
        slim_heads: int,
        bits_per_symbol: int = 8,
        backend: str = "wind",
        truncation_length: int = 8,
        native_mode: bool = False,
        use_gate: bool = True,
        hard_backend: str | None = None,
        hard_switch_step: int | None = None,
        sequence_length: int | None = None,
        capital_init: float = 1.0,
        capital_floor: float = 0.01,
        capital_ceiling: float | None = None,
    ) -> None:
        super().__init__(
            capital_init=capital_init,
            capital_floor=capital_floor,
            capital_ceiling=capital_ceiling,
        )
        backend = normalize_rosa_backend(backend)
        hard_backend = None if hard_backend is None else normalize_rosa_backend(hard_backend)
        if backend not in self._SUPPORTED_BACKENDS:
            raise ValueError(
                f"ROSAExpert supports backend in {sorted(self._SUPPORTED_BACKENDS)!r}, got {backend!r}"
            )
        if slim_heads <= 0 or bits_per_symbol <= 0:
            raise ValueError("ROSA expert requires positive slim_heads and bits_per_symbol.")

        self.dim = int(dim)
        self.slim_heads = int(slim_heads)
        self.bits_per_symbol = int(bits_per_symbol)
        self.backend = str(backend)
        self.truncation_length = int(truncation_length)
        self.native_mode = bool(native_mode)
        self.use_gate = bool(use_gate)
        self.hard_backend = None if hard_backend is None else str(hard_backend)
        self.hard_switch_step = None if hard_switch_step is None else int(hard_switch_step)
        self.sequence_length = None if sequence_length is None else int(sequence_length)
        hidden = self.slim_heads * self.bits_per_symbol

        self.norm = nn.LayerNorm(dim)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_q = nn.Parameter(torch.zeros(1, 1, dim))
        self.x_k = nn.Parameter(torch.zeros(1, 1, dim))
        self.x_v = nn.Parameter(torch.zeros(1, 1, dim))

        self.wq = nn.Linear(dim, hidden, bias=False)
        self.wk = nn.Linear(dim, hidden, bias=False)
        self.wv = nn.Linear(dim, hidden, bias=False)
        self.symbol_scale = nn.Parameter(torch.ones(1, 1, hidden))
        self.wo = nn.Linear(hidden, dim, bias=False)
        self.continuous_proj = nn.Linear(dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)

    @property
    def expert_type(self) -> str:
        return "rosa"

    @property
    def supports_sparse_dispatch(self) -> bool:
        return False

    def _ensure_btd(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        squeezed = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeezed = True
        if x.dim() != 3:
            raise ValueError(f"ROSAExpert expected [B, T, D] or [N, D], got {tuple(x.shape)}")
        return x, squeezed

    def _compute_symbol_logits(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        x, _ = self._ensure_btd(x)
        _batch, steps, _dim = x.shape
        h = self.norm(x)
        xx = self.time_shift(h) - h
        q_source = h + xx * self.x_q
        k_source = h + xx * self.x_k
        v_source = h + xx * self.x_v

        q_logits = self.wq(q_source)
        k_logits = self.wk(k_source)
        v_logits = self.wv(v_source)
        return h, q_logits, k_logits, v_logits, xx, steps

    def effective_backend(self, current_step: int | None = None) -> str:
        if (
            self.hard_backend is not None
            and self.hard_switch_step is not None
            and current_step is not None
            and current_step >= self.hard_switch_step
        ):
            return self.hard_backend
        return self.backend

    def forward(self, x: torch.Tensor, **ctx) -> torch.Tensor:
        current_step = ctx.pop("current_step", None)
        x, squeezed = self._ensure_btd(x)
        h, q_logits, k_logits, v_logits, _xx, steps = self._compute_symbol_logits(x)
        backend = self.effective_backend(current_step)

        if backend in {"hard_symbolic_multibit", "hard_qkv_multibit"}:
            q = StraightThroughSign.apply(q_logits)
            k = StraightThroughSign.apply(k_logits)
            v = StraightThroughSign.apply(v_logits)
            padded_q, padded_k, padded_v = self._maybe_pad_sequence(q, k, v)
            out = wind_rosa(
                padded_q,
                padded_k,
                padded_v,
                bits_per_symbol=self.bits_per_symbol,
                truncation_length=self.truncation_length,
            )
        elif backend.startswith("soft_qkv_multibit"):
            ua_mode = backend.startswith("soft_qkv_multibit_unmatched")
            stem = "soft_qkv_multibit_unmatched" if ua_mode else "soft_qkv_multibit"
            exact_backend = backend.removeprefix(stem).lstrip("_") or "auto"
            q = StraightThroughSign.apply(q_logits)
            k = StraightThroughSign.apply(k_logits)
            v = StraightThroughSign.apply(v_logits)
            padded_q, padded_k, padded_v = self._maybe_pad_sequence(q, k, v)
            if ua_mode:
                out, aux = soft_rosa_exact_with_aux(
                    padded_q,
                    padded_k,
                    padded_v,
                    bits_per_symbol=self.bits_per_symbol,
                    truncation_length=self.truncation_length,
                    scan_backend=exact_backend,
                )
                # Make unmatched positions explicitly collapse toward 0 while keeping
                # soft, differentiable behavior for weak-but-valid matches.
                match_strength = 1.0 - torch.exp(-aux["chosen_ell"].clamp_min(0.0))
                match_strength = match_strength.repeat_interleave(self.bits_per_symbol, dim=-1)
                out = out * match_strength.to(dtype=out.dtype)
            else:
                out = soft_rosa_exact(
                    padded_q,
                    padded_k,
                    padded_v,
                    bits_per_symbol=self.bits_per_symbol,
                    truncation_length=self.truncation_length,
                    scan_backend=exact_backend,
                )
        elif backend.startswith("soft_exact_dp"):
            exact_backend = backend.removeprefix("soft_exact_dp").lstrip("_") or "auto"
            padded_q, padded_k, padded_v = self._maybe_pad_sequence(q_logits, k_logits, v_logits)
            out = soft_rosa_exact(
                padded_q,
                padded_k,
                padded_v,
                bits_per_symbol=self.bits_per_symbol,
                truncation_length=self.truncation_length,
                scan_backend=exact_backend,
            )
        elif backend.startswith("soft_qkv_binary"):
            if self.bits_per_symbol != 1:
                raise ValueError(
                    f"{backend!r} requires bits_per_symbol == 1, got {self.bits_per_symbol}."
                )
            qkv1bit_mode = "bipolar" if backend.startswith("soft_qkv_binary_bipolar") else "plain"
            stem = "soft_qkv_binary_bipolar" if qkv1bit_mode == "bipolar" else "soft_qkv_binary"
            backend_name = backend.removeprefix(stem).lstrip("_") or "auto"
            q = StraightThroughSign.apply(q_logits)
            k = StraightThroughSign.apply(k_logits)
            v = StraightThroughSign.apply(v_logits)
            padded_q, padded_k, padded_v = self._maybe_pad_sequence(q, k, v)
            if qkv1bit_mode == "bipolar":
                out, aux = soft_rosa_qkv1bit_with_aux(
                    padded_q,
                    padded_k,
                    padded_v,
                    truncation_length=self.truncation_length,
                    backend=backend_name,
                )
                matched = (aux["best_j"] >= 0).to(dtype=out.dtype)
                # Experimental RWKV8-style semantics:
                # matched 1 => +e, matched 0 => -e, unmatched => 0.
                out = torch.where(matched > 0, out * 2.0 - 1.0, torch.zeros_like(out))
            else:
                out = soft_rosa_qkv1bit(
                    padded_q,
                    padded_k,
                    padded_v,
                    truncation_length=self.truncation_length,
                    backend=backend_name,
                )
        else:
            padded_q, padded_k, padded_v = self._maybe_pad_sequence(q_logits, k_logits, v_logits)
            out = rosa_soft(
                padded_q,
                padded_k,
                padded_v,
                bits_per_symbol=self.bits_per_symbol,
                mode=backend,
                truncation_length=self.truncation_length,
            )
        out = out[:, :steps, :]
        out = out.to(h.dtype) * self.symbol_scale.to(h.dtype)

        symbolic = self.wo(out)
        if self.native_mode:
            out = symbolic
        else:
            continuous_v = self.continuous_proj(h)
            if self.use_gate:
                fused = F.silu(symbolic) * continuous_v + continuous_v
            else:
                fused = symbolic + continuous_v
            out = self.output_proj(fused)

        if squeezed:
            return out.squeeze(0)
        return out

    def summarize_symbol_language(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        max_tokens: int = 24,
        max_heads: int = 2,
        top_k: int = 5,
    ) -> dict[str, object]:
        x, _ = self._ensure_btd(x.detach())
        _, q_logits, k_logits, v_logits, _xx, steps = self._compute_symbol_logits(x)

        if valid_mask is None:
            valid = torch.ones(x.size(0), steps, dtype=torch.bool, device=x.device)
        else:
            valid = valid_mask.to(device=x.device).bool()
            valid = valid[:, :steps]

        q_ids = self._pack_symbol_ids(q_logits)
        k_ids = self._pack_symbol_ids(k_logits)
        v_ids = self._pack_symbol_ids(v_logits)
        q_bits = self._hard_bits(q_logits)

        head_limit = min(max_heads, q_ids.size(-1))
        sample_limit = min(max_tokens, steps)
        sample_valid = valid[0, :sample_limit]

        q_vocab_counts: list[int] = []
        q_entropy_values: list[float] = []
        q_repeat_values: list[float] = []
        head_top_symbols: dict[str, list[str]] = {}
        sample_q: dict[str, list[str]] = {}
        sample_k: dict[str, list[str]] = {}
        sample_v: dict[str, list[str]] = {}

        for head_idx in range(q_ids.size(-1)):
            ids = q_ids[:, :, head_idx][valid]
            if ids.numel() == 0:
                q_vocab_counts.append(0)
                q_entropy_values.append(0.0)
                q_repeat_values.append(0.0)
                continue

            unique, counts = torch.unique(ids, return_counts=True)
            probs = counts.float() / counts.sum().clamp(min=1)
            entropy = -(probs * probs.clamp(min=1e-12).log2()).sum().item()
            repeat_rate = 1.0 - (unique.numel() / float(ids.numel()))

            q_vocab_counts.append(int(unique.numel()))
            q_entropy_values.append(float(entropy))
            q_repeat_values.append(float(repeat_rate))

            if head_idx < head_limit:
                order = torch.argsort(counts, descending=True)[:top_k]
                head_top_symbols[f"head_{head_idx}"] = [
                    f"{self._format_symbol(int(unique[idx].item()))}:{int(counts[idx].item())}"
                    for idx in order
                ]
                sample_q[f"head_{head_idx}"] = [
                    self._format_symbol(int(token))
                    for token in q_ids[0, :sample_limit, head_idx][sample_valid].tolist()
                ]
                sample_k[f"head_{head_idx}"] = [
                    self._format_symbol(int(token))
                    for token in k_ids[0, :sample_limit, head_idx][sample_valid].tolist()
                ]
                sample_v[f"head_{head_idx}"] = [
                    self._format_symbol(int(token))
                    for token in v_ids[0, :sample_limit, head_idx][sample_valid].tolist()
                ]

        return {
            "backend": self.backend,
            "hard_backend": self.hard_backend,
            "hard_switch_step": self.hard_switch_step,
            "native_mode": self.native_mode,
            "use_gate": self.use_gate,
            "bits_per_symbol": self.bits_per_symbol,
            "heads": self.slim_heads,
            "valid_tokens": int(valid.sum().item()),
            "q_bit_balance": float(q_bits[valid.unsqueeze(-1).expand_as(q_bits)].float().mean().item()),
            "q_margin_mean": float(q_logits.abs()[valid.unsqueeze(-1).expand_as(q_logits)].mean().item()),
            "q_vocab_mean": float(sum(q_vocab_counts) / max(len(q_vocab_counts), 1)),
            "q_vocab_per_head": q_vocab_counts[:head_limit],
            "q_entropy_mean": float(sum(q_entropy_values) / max(len(q_entropy_values), 1)),
            "q_repeat_rate_mean": float(sum(q_repeat_values) / max(len(q_repeat_values), 1)),
            "top_q_symbols": head_top_symbols,
            "sample0_q": sample_q,
            "sample0_k": sample_k,
            "sample0_v": sample_v,
        }

    def _pack_symbol_ids(self, logits: torch.Tensor) -> torch.Tensor:
        batch, steps, hidden = logits.shape
        heads = hidden // self.bits_per_symbol
        bits = (logits.view(batch, steps, heads, self.bits_per_symbol) > 0).to(torch.int64)
        shifts = torch.arange(self.bits_per_symbol, device=logits.device, dtype=torch.int64)
        return (bits << shifts.view(1, 1, 1, -1)).sum(dim=-1)

    @staticmethod
    def _hard_bits(logits: torch.Tensor) -> torch.Tensor:
        return logits > 0

    def _format_symbol(self, value: int) -> str:
        width = max(1, (self.bits_per_symbol + 3) // 4)
        return f"{value:0{width}X}"

    def _maybe_pad_sequence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steps = q.size(1)
        if self.sequence_length is None:
            return q, k, v
        if steps > self.sequence_length:
            raise ValueError(
                f"ROSAExpert received T={steps}, which exceeds configured sequence_length={self.sequence_length}."
            )
        if steps == self.sequence_length:
            return q, k, v
        pad = self.sequence_length - steps
        return (
            F.pad(q, (0, 0, 0, pad)),
            F.pad(k, (0, 0, 0, pad)),
            F.pad(v, (0, 0, 0, pad)),
        )
