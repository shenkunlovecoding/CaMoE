"""Local adapter for the vendored Slim Wind ROSA kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT_DIR = Path(__file__).resolve().parents[1]
WIND_ROSA_DIR = ROOT_DIR / "wind_rosa"


@dataclass(frozen=True)
class WindRosaSpec:
    sequence_length: int
    bits_per_symbol: int
    truncation_length: int


_LOADED_SPEC: WindRosaSpec | None = None
_SCRATCH: torch.Tensor | None = None
_LOAD_WARNING_EMITTED = False
_CUDA_KERNEL_DISABLED = False


def _pack_bits(x: torch.Tensor, bits_per_symbol: int) -> torch.Tensor:
    batch, steps, hidden = x.shape
    if hidden % bits_per_symbol != 0:
        raise ValueError(
            f"Hidden size {hidden} must be divisible by bits_per_symbol={bits_per_symbol}."
        )
    heads = hidden // bits_per_symbol
    shifts = torch.arange(bits_per_symbol, device=x.device, dtype=torch.int64)
    bits = (x.view(batch, steps, heads, bits_per_symbol) > 0).to(torch.int64)
    return (bits << shifts.view(1, 1, 1, -1)).sum(dim=-1)


def _decode_signed_symbols(
    packed_symbols: torch.Tensor,
    match_length: torch.Tensor,
    bits_per_symbol: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    shifts = torch.arange(bits_per_symbol, device=packed_symbols.device, dtype=torch.int64)
    bits = ((packed_symbols.unsqueeze(-1) >> shifts.view(1, 1, 1, -1)) & 1).to(dtype)
    signed = bits.mul(2).sub(1)
    matched = (match_length > 0).unsqueeze(-1).to(dtype)
    return (signed * matched).reshape(
        packed_symbols.size(0),
        packed_symbols.size(1),
        packed_symbols.size(2) * bits_per_symbol,
    )


def _expand_match_mask(match_length: torch.Tensor, bits_per_symbol: int) -> torch.Tensor:
    return (match_length > 0).unsqueeze(-1).expand(*match_length.shape, bits_per_symbol)


def _reshape_match_mask(match_length: torch.Tensor, bits_per_symbol: int, dtype: torch.dtype) -> torch.Tensor:
    return _expand_match_mask(match_length, bits_per_symbol).reshape(
        match_length.size(0),
        match_length.size(1),
        match_length.size(2) * bits_per_symbol,
    ).to(dtype)


def _reshape_signed_grad(
    grad_output: torch.Tensor,
    match_length: torch.Tensor,
    bits_per_symbol: int,
) -> torch.Tensor:
    grad_bits = 2.0 * grad_output.float() * _reshape_match_mask(match_length, bits_per_symbol, torch.float32)
    batch, steps, hidden = grad_bits.shape
    heads = hidden // bits_per_symbol
    return grad_bits.view(batch, steps, heads, bits_per_symbol).transpose(1, 2).contiguous()


def _unpack_bits(x: torch.Tensor, bits_per_symbol: int, dtype: torch.dtype) -> torch.Tensor:
    shifts = torch.arange(bits_per_symbol, device=x.device, dtype=torch.int64)
    bits = ((x.unsqueeze(-1) >> shifts.view(1, 1, 1, -1)) & 1).to(dtype)
    return bits.reshape(x.size(0), x.size(1), x.size(2) * bits_per_symbol)


def _naive_symbolic_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    truncation_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, steps, heads = q.shape
    y = torch.empty_like(v)
    match_length = torch.zeros(batch, steps, heads, dtype=torch.int32)
    q_list = q.tolist()
    k_list = k.tolist()
    v_list = v.tolist()
    for batch_idx in range(batch):
        for head_idx in range(heads):
            q_head = [step[head_idx] for step in q_list[batch_idx]]
            k_head = [step[head_idx] for step in k_list[batch_idx]]
            v_head = [step[head_idx] for step in v_list[batch_idx]]
            for step_idx in range(steps):
                best_value = v_head[step_idx]
                best_length = 0
                max_length = min(truncation_length, step_idx + 1)
                for width in range(max_length, 0, -1):
                    target = tuple(q_head[step_idx + 1 - width : step_idx + 1])
                    found = False
                    for start in range(step_idx - width, -1, -1):
                        if tuple(k_head[start : start + width]) == target:
                            best_value = v_head[start + width]
                            best_length = width
                            found = True
                            break
                    if found:
                        break
                y[batch_idx, step_idx, head_idx] = best_value
                match_length[batch_idx, step_idx, head_idx] = best_length
    return y, match_length


class _WindRosaFallback(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bits_per_symbol: int,
        truncation_length: int,
    ) -> torch.Tensor:
        del ctx
        packed_q = _pack_bits(q.float(), bits_per_symbol).cpu()
        packed_k = _pack_bits(k.float(), bits_per_symbol).cpu()
        packed_v = _pack_bits(v.float(), bits_per_symbol).cpu()
        packed_y, match_length = _naive_symbolic_forward(packed_q, packed_k, packed_v, truncation_length)
        return _decode_signed_symbols(
            packed_y.to(q.device),
            match_length.to(q.device),
            bits_per_symbol,
            q.dtype,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        del ctx
        return (
            torch.zeros_like(grad_output),
            torch.zeros_like(grad_output),
            torch.zeros_like(grad_output),
            None,
            None,
        )


class _WindRosaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert _LOADED_SPEC is not None
        bits_per_symbol = _LOADED_SPEC.bits_per_symbol
        batch, steps, hidden = q.shape
        if steps != _LOADED_SPEC.sequence_length:
            raise ValueError(
                f"Wind ROSA kernel expects T={_LOADED_SPEC.sequence_length}, got T={steps}."
            )
        if hidden % bits_per_symbol != 0:
            raise ValueError(
                f"Hidden size {hidden} must be divisible by bits_per_symbol={bits_per_symbol}."
            )
        if not all(t.is_cuda for t in (q, k, v)):
            raise ValueError("Wind ROSA CUDA kernel requires CUDA tensors.")

        heads = hidden // bits_per_symbol
        packed_q = _pack_bits(q.float(), bits_per_symbol).to(torch.uint8).transpose(1, 2).contiguous()
        packed_k = _pack_bits(k.float(), bits_per_symbol).to(torch.uint8).transpose(1, 2).contiguous()
        packed_v = _pack_bits(v.float(), bits_per_symbol).to(torch.uint8).transpose(1, 2).contiguous()

        global _SCRATCH
        scratch_bytes = int(torch.ops.wind_rosa.scratch_size(batch, heads))
        if _SCRATCH is None or _SCRATCH.numel() < scratch_bytes or _SCRATCH.device != q.device:
            _SCRATCH = torch.empty(scratch_bytes, dtype=torch.uint8, device=q.device)

        y = torch.empty_like(packed_v)
        match_length = torch.empty(batch, heads, steps, dtype=torch.int32, device=q.device)
        torch.ops.wind_rosa.forward_with_match_len(packed_q, packed_k, packed_v, y, match_length, _SCRATCH)
        ctx.save_for_backward(packed_q, packed_k, packed_v, match_length)
        return _decode_signed_symbols(y.transpose(1, 2), match_length.transpose(1, 2), bits_per_symbol, torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        packed_q, packed_k, packed_v, match_length = ctx.saved_tensors
        assert _LOADED_SPEC is not None
        bits_per_symbol = _LOADED_SPEC.bits_per_symbol
        batch, heads, steps = packed_q.shape
        grad = _reshape_signed_grad(grad_output, match_length.transpose(1, 2), bits_per_symbol)
        dq = torch.empty(batch, heads, steps, bits_per_symbol, device=packed_q.device, dtype=torch.float32)
        dk = torch.empty_like(dq)
        dv = torch.empty_like(dq)
        assert _SCRATCH is not None
        torch.ops.wind_rosa.backward(packed_q, packed_k, packed_v, grad, dq, dk, dv, _SCRATCH)
        return (
            dq.transpose(1, 2).reshape(batch, steps, heads * bits_per_symbol),
            dk.transpose(1, 2).reshape(batch, steps, heads * bits_per_symbol),
            dv.transpose(1, 2).reshape(batch, steps, heads * bits_per_symbol),
        )


def load_wind_rosa(spec: WindRosaSpec) -> None:
    global _LOADED_SPEC
    if _LOADED_SPEC is not None:
        if _LOADED_SPEC != spec:
            raise ValueError(
                f"Wind ROSA kernel already loaded with {_LOADED_SPEC}, requested {spec}. "
                "Use one fixed (T, C, K) per process."
            )
        return
    if not WIND_ROSA_DIR.exists():
        raise FileNotFoundError(f"Vendored wind_rosa directory not found: {WIND_ROSA_DIR}")

    load(
        name=f"camoe_wind_rosa_t{spec.sequence_length}_c{spec.bits_per_symbol}_k{spec.truncation_length}",
        sources=[
            str(WIND_ROSA_DIR / "wind_rosa.cu"),
            str(WIND_ROSA_DIR / "wind_rosa.cpp"),
        ],
        is_python_module=False,
        verbose=False,
        extra_cuda_cflags=[
            f"-D_T_={spec.sequence_length}",
            f"-D_C_={spec.bits_per_symbol}",
            f"-D_K_={spec.truncation_length}",
        ],
    )
    _LOADED_SPEC = spec


def wind_rosa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bits_per_symbol: int,
    truncation_length: int,
) -> torch.Tensor:
    global _LOAD_WARNING_EMITTED, _CUDA_KERNEL_DISABLED
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"wind_rosa expects matching q/k/v shapes, got {q.shape}, {k.shape}, {v.shape}")

    if q.device.type != "cuda" or _CUDA_KERNEL_DISABLED:
        return _WindRosaFallback.apply(q, k, v, bits_per_symbol, truncation_length)

    spec = WindRosaSpec(
        sequence_length=q.size(1),
        bits_per_symbol=bits_per_symbol,
        truncation_length=truncation_length,
    )
    try:
        load_wind_rosa(spec)
        return _WindRosaFunction.apply(q.float(), k.float(), v.float())
    except Exception as exc:
        _CUDA_KERNEL_DISABLED = True
        if not _LOAD_WARNING_EMITTED:
            print(f"Warning: Wind ROSA CUDA kernel unavailable, falling back to symbolic CPU path: {exc}")
            _LOAD_WARNING_EMITTED = True
        return _WindRosaFallback.apply(q, k, v, bits_per_symbol, truncation_length)
