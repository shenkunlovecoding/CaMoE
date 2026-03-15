from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

Tensor = torch.Tensor


if HAS_TRITON:
    @triton.jit
    def _diag_combine(g0, a0, g1, a1):
        g_ = g1 * g0
        a_ = g1 * a0 + a1
        return g_, a_


    @triton.jit
    def _diag_scan_kernel(
        x_ptr,
        y_ptr,
        T,
        stride_x_b,
        stride_x_r,
        stride_x_c,
        stride_y_b,
        stride_y_r,
        stride_y_c,
        BLOCK_SIZE_T: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        x_ptr_b = x_ptr + pid_b * stride_x_b
        y_ptr_b = y_ptr + pid_b * stride_y_b

        carry_g = 1.0
        carry_a = 0.0

        for block_start in range(0, T - pid_d, BLOCK_SIZE_T):
            off_c = block_start + tl.arange(0, BLOCK_SIZE_T)
            off_r = off_c + pid_d
            mask = off_r < T

            g = tl.load(
                x_ptr_b + off_r * stride_x_r + off_c * stride_x_c,
                mask=mask,
                other=1.0,
            )
            a = tl.where(mask, g, 0.0)

            _g, _a = _diag_combine(carry_g, carry_a, g, a)
            _g = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _g, g)
            _a = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _a, a)

            _g, _a = tl.associative_scan((_g, _a), axis=0, combine_fn=_diag_combine)
            tl.store(y_ptr_b + off_r * stride_y_r + off_c * stride_y_c, _a, mask=mask)

            _g, _a = tl.reduce((g, a), axis=0, combine_fn=_diag_combine)
            carry_g, carry_a = _diag_combine(carry_g, carry_a, _g, _a)


    @triton.jit
    def _diag_scan_bwd_kernel(
        x_ptr,
        y_ptr,
        g_ptr,
        o_ptr,
        T,
        stride_x_b,
        stride_x_r,
        stride_x_c,
        stride_y_b,
        stride_y_r,
        stride_y_c,
        stride_g_b,
        stride_g_r,
        stride_g_c,
        stride_o_b,
        stride_o_r,
        stride_o_c,
        BLOCK_SIZE_T: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        x_ptr_b = x_ptr + pid_b * stride_x_b
        y_ptr_b = y_ptr + pid_b * stride_y_b
        g_ptr_b = g_ptr + pid_b * stride_g_b
        o_ptr_b = o_ptr + pid_b * stride_o_b

        carry_a = 1.0
        carry_b = 0.0

        M = tl.cdiv(T - pid_d, BLOCK_SIZE_T)
        for i in range(M):
            block_start = (M - 1 - i) * BLOCK_SIZE_T
            off_c = block_start + BLOCK_SIZE_T - 1 - tl.arange(0, BLOCK_SIZE_T)
            off_r = off_c + pid_d

            mask = off_r < T

            a = tl.load(
                x_ptr_b + (off_r + 1) * stride_x_r + (off_c + 1) * stride_x_c,
                mask=mask & (off_r < T - 1),
                other=1.0,
            )
            b = tl.load(
                g_ptr_b + off_r * stride_g_r + off_c * stride_g_c,
                mask=mask,
                other=0.0,
            )

            _a, _b = _diag_combine(carry_a, carry_b, a, b)
            _a = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _a, a)
            _b = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _b, b)

            _a, _b = tl.associative_scan((_a, _b), axis=0, combine_fn=_diag_combine)

            y = tl.load(
                y_ptr_b + (off_r - 1) * stride_y_r + (off_c - 1) * stride_y_c,
                mask=mask & (off_r > pid_d),
                other=0.0,
            )
            tl.store(o_ptr_b + off_r * stride_o_r + off_c * stride_o_c, (y + 1) * _b, mask=mask)

            _a, _b = tl.reduce((a, b), axis=0, combine_fn=_diag_combine)
            carry_a, carry_b = _diag_combine(carry_a, carry_b, _a, _b)


    class _DiagonalScanFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: Tensor) -> Tensor:
            x32 = x.contiguous().to(torch.float32)
            out = torch.zeros_like(x32)
            seq_len = x32.size(-1)
            block_size_t = triton.next_power_of_2(min(seq_len, 1024))
            _diag_scan_kernel[(x32.size(0), seq_len)](
                x32,
                out,
                seq_len,
                x32.stride(0),
                x32.stride(1),
                x32.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                BLOCK_SIZE_T=block_size_t,
            )
            ctx.save_for_backward(x32, out)
            return out

        @staticmethod
        def backward(ctx, grad_output: Tensor):
            x32, out = ctx.saved_tensors
            grad_output = grad_output.contiguous().to(torch.float32)
            grad_input = torch.zeros_like(x32)
            seq_len = x32.size(-1)
            block_size_t = triton.next_power_of_2(min(seq_len, 1024))
            _diag_scan_bwd_kernel[(x32.size(0), seq_len)](
                x32,
                out,
                grad_output,
                grad_input,
                seq_len,
                x32.stride(0),
                x32.stride(1),
                x32.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                grad_input.stride(0),
                grad_input.stride(1),
                grad_input.stride(2),
                BLOCK_SIZE_T=block_size_t,
            )
            return grad_input


def diagonal_affine_scan_triton(x: Tensor) -> Tensor:
    if not HAS_TRITON:
        raise ValueError("Triton is not available")
    if not x.is_cuda:
        raise ValueError("Triton diagonal scan requires CUDA tensors")
    return _DiagonalScanFunction.apply(x).to(x.dtype)


def triton_is_available() -> bool:
    return HAS_TRITON


def triton_module() -> Optional[object]:
    return triton
