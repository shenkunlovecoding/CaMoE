from __future__ import annotations

from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

Tensor = torch.Tensor


def _as_bits(x: Tensor) -> Tensor:
    if x.dtype == torch.bool:
        return x.to(torch.long)
    if x.is_floating_point():
        return (x > 0).to(torch.long)
    return x.to(torch.long)


@triton.jit
def _combine(g0, a0, g1, a1):
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

        _g, _a = _combine(carry_g, carry_a, g, a)
        _g = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _g, g)
        _a = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _a, a)

        _g, _a = tl.associative_scan((_g, _a), axis=0, combine_fn=_combine)
        tl.store(y_ptr_b + off_r * stride_y_r + off_c * stride_y_c, _a, mask=mask)

        _g, _a = tl.reduce((g, a), axis=0, combine_fn=_combine)
        carry_g, carry_a = _combine(carry_g, carry_a, _g, _a)


def _reshape_streams(x: Tensor) -> Tuple[Tensor, Tuple[int, int, int]]:
    x = _as_bits(x)
    if x.ndim != 3:
        raise ValueError("Expected [B, T, N] input")
    batch, seq_len, num_streams = x.shape
    stream_major = x.transpose(1, 2).contiguous().view(batch * num_streams, seq_len)
    return stream_major, (batch, seq_len, num_streams)


def _restore_streams(x: Tensor, shape: Tuple[int, int, int]) -> Tensor:
    batch, seq_len, num_streams = shape
    return x.view(batch, num_streams, seq_len).transpose(1, 2).contiguous()


@torch.no_grad()
def qkv1bit_forward_triton(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    K: int,
    return_aux: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("Triton backend requires CUDA tensors")

    q_stream, shape = _reshape_streams(query)
    k_stream, _ = _reshape_streams(key)
    v_stream, _ = _reshape_streams(value)
    batch_streams, seq_len = q_stream.shape
    K = int(max(1, min(K, seq_len)))

    eq = (q_stream[:, :, None] == k_stream[:, None, :]).to(torch.float32)
    run = torch.zeros_like(eq)

    block_size_t = triton.next_power_of_2(min(seq_len, 1024))
    _diag_scan_kernel[(batch_streams, seq_len)](
        eq,
        run,
        seq_len,
        eq.stride(0),
        eq.stride(1),
        eq.stride(2),
        run.stride(0),
        run.stride(1),
        run.stride(2),
        BLOCK_SIZE_T=block_size_t,
    )

    run = run.clamp_max(K).to(torch.long)

    i_idx = torch.arange(seq_len, device=run.device).view(1, seq_len, 1)
    j_idx = torch.arange(seq_len, device=run.device).view(1, 1, seq_len)
    valid = (j_idx < i_idx) & (run > 0)

    score = torch.where(valid, run * (seq_len + 1) + j_idx, torch.full_like(run, -1))
    best_score, best_j = score.max(dim=-1)
    best_j = torch.where(best_score >= 0, best_j, torch.full_like(best_j, -1))
    best_len = torch.where(best_score >= 0, best_score // (seq_len + 1), torch.zeros_like(best_score))

    gather_idx = (best_j + 1).clamp(0, seq_len - 1)
    output = torch.gather(v_stream, dim=1, index=gather_idx)
    output = torch.where(best_j >= 0, output, torch.zeros_like(output)).to(torch.float32)

    output = _restore_streams(output, shape)
    if not return_aux:
        return output

    aux = {
        "best_j": _restore_streams(best_j, shape).to(torch.long),
        "best_len": _restore_streams(best_len, shape).to(torch.long),
    }
    return output, aux
