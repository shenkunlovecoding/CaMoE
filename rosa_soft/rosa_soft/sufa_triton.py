from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - exercised through availability checks
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: Exception | None = exc
else:
    _TRITON_IMPORT_ERROR = None


def is_triton_window_available() -> bool:
    return triton is not None and tl is not None


def triton_window_import_error() -> Exception | None:
    return _TRITON_IMPORT_ERROR


def _window_decay_weights(
    win_size: int,
    decay_factor: Optional[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if win_size <= 0:
        raise ValueError(f"win_size must be positive, got {win_size}.")
    if win_size == 1:
        return torch.ones(1, device=device, dtype=dtype)

    if decay_factor is None:
        base = max(0.1, 1.0 / win_size) ** (1.0 / win_size)
    else:
        base = max(0.1, min(float(decay_factor), 0.99))

    inds = torch.arange(win_size, device=device, dtype=torch.float32)
    powers = (win_size - 1) - inds
    weights = torch.pow(torch.tensor(base, device=device, dtype=torch.float32), powers)
    weights = weights / weights.sum()
    return torch.sqrt(weights).to(dtype=dtype)


if is_triton_window_available():

    @triton.jit
    def _suffix_window_forward_kernel(
        x_ptr,
        weights_ptr,
        out_ptr,
        seq_len,
        win_size,
        dim,
        shift,
        block_d: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)

        w = pid0 % win_size
        t = (pid0 // win_size) % seq_len
        bh = pid0 // (win_size * seq_len)
        src_t = t + w - shift

        d_offsets = pid1 * block_d + tl.arange(0, block_d)
        valid_src = (src_t >= 0) & (src_t < seq_len)
        mask = valid_src & (d_offsets < dim)
        safe_src_t = tl.where(valid_src, src_t, 0)

        x_offsets = ((bh * seq_len + safe_src_t) * dim) + d_offsets
        out_offsets = pid0 * dim + d_offsets

        values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        weight = tl.load(weights_ptr + w)
        values = tl.where(mask, values * weight, 0.0)
        tl.store(out_ptr + out_offsets, values, mask=d_offsets < dim)


    @triton.jit
    def _suffix_window_backward_kernel(
        grad_out_ptr,
        weights_ptr,
        grad_x_ptr,
        seq_len,
        win_size,
        dim,
        shift,
        block_d: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)

        w = pid0 % win_size
        t = (pid0 // win_size) % seq_len
        bh = pid0 // (win_size * seq_len)
        src_t = t + w - shift

        d_offsets = pid1 * block_d + tl.arange(0, block_d)
        valid_src = (src_t >= 0) & (src_t < seq_len)
        mask = valid_src & (d_offsets < dim)
        safe_src_t = tl.where(valid_src, src_t, 0)
        grad_offsets = pid0 * dim + d_offsets
        grad_values = tl.load(grad_out_ptr + grad_offsets, mask=mask, other=0.0)
        weight = tl.load(weights_ptr + w)
        grad_values = grad_values * weight

        x_offsets = ((bh * seq_len + safe_src_t) * dim) + d_offsets
        tl.atomic_add(grad_x_ptr + x_offsets, grad_values, mask=mask)


class _SuffixWindowTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        win_size: int,
        offset: int,
        decay_factor: Optional[float],
    ) -> torch.Tensor:
        if not is_triton_window_available():
            raise RuntimeError("Triton is not available in this environment.")
        if not x.is_cuda:
            raise RuntimeError("The Triton suffix window kernel requires CUDA tensors.")
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape [B, H, T, D], got {tuple(x.shape)}")

        x = x.contiguous()
        batch, heads, seq_len, dim = x.shape
        total_bh = batch * heads
        total_windows = total_bh * seq_len * win_size
        block_d = max(1, min(64, triton.next_power_of_2(dim)))

        weights = _window_decay_weights(
            win_size,
            decay_factor,
            device=x.device,
            dtype=x.dtype,
        ).contiguous()
        out = torch.empty(
            (total_bh, seq_len, win_size, dim),
            device=x.device,
            dtype=x.dtype,
        )

        grid = (total_windows, triton.cdiv(dim, block_d))
        _suffix_window_forward_kernel[grid](
            x.view(total_bh, seq_len, dim),
            weights,
            out,
            seq_len,
            win_size,
            dim,
            win_size - 1 + offset,
            block_d=block_d,
        )

        ctx.save_for_backward(weights)
        ctx.input_shape = (batch, heads, seq_len, dim)
        ctx.win_size = int(win_size)
        ctx.offset = int(offset)
        ctx.block_d = int(block_d)
        return out.view(batch, heads, seq_len, win_size, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weights,) = ctx.saved_tensors
        batch, heads, seq_len, dim = ctx.input_shape
        win_size = ctx.win_size
        total_bh = batch * heads
        total_windows = total_bh * seq_len * win_size

        grad_output = grad_output.contiguous().view(total_bh, seq_len, win_size, dim)
        grad_x = torch.zeros((total_bh, seq_len, dim), device=grad_output.device, dtype=grad_output.dtype)

        grid = (total_windows, triton.cdiv(dim, ctx.block_d))
        _suffix_window_backward_kernel[grid](
            grad_output,
            weights,
            grad_x,
            seq_len,
            win_size,
            dim,
            win_size - 1 + ctx.offset,
            block_d=ctx.block_d,
        )

        return grad_x.view(batch, heads, seq_len, dim), None, None, None


def apply_suffix_window_triton(
    x: torch.Tensor,
    *,
    win_size: int,
    offset: int,
    decay_factor: Optional[float],
) -> torch.Tensor:
    return _SuffixWindowTritonFunction.apply(x, win_size, offset, decay_factor)
