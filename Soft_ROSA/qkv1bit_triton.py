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


@triton.jit
def _backward_qk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    y_ptr,
    go_ptr,
    out_ptr,
    T,
    K,
    stride_q_b,
    stride_q_t,
    stride_k_b,
    stride_k_t,
    stride_v_b,
    stride_v_t,
    stride_y_b,
    stride_y_t,
    stride_go_b,
    stride_go_t,
    stride_out_b,
    stride_out_t,
    FLIP_QUERY: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)

    offs_t = pid_blk * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    mask_t = offs_t < T

    q_ptr_b = q_ptr + pid_b * stride_q_b
    k_ptr_b = k_ptr + pid_b * stride_k_b
    v_ptr_b = v_ptr + pid_b * stride_v_b
    y_ptr_b = y_ptr + pid_b * stride_y_b
    go_ptr_b = go_ptr + pid_b * stride_go_b
    out_ptr_b = out_ptr + pid_b * stride_out_b

    accum = tl.zeros((BLOCK_SIZE_T,), dtype=tl.float32)

    for i in range(0, T):
        cur_best_len = tl.zeros((BLOCK_SIZE_T,), dtype=tl.int32)
        cur_best_j = tl.full((BLOCK_SIZE_T,), -1, dtype=tl.int32)

        max_check = min(K, i + 1)
        for j in range(0, T):
            valid_j = mask_t & (j < i)
            max_match = min(max_check, j + 1)
            length = tl.zeros((BLOCK_SIZE_T,), dtype=tl.int32)

            for off in range(0, K):
                can_continue = valid_j & (off < max_match) & (length == off)

                qi = i - off
                kj = j - off
                q_base = tl.load(q_ptr_b + qi * stride_q_t)
                k_base = tl.load(k_ptr_b + kj * stride_k_t)
                qv = tl.where(FLIP_QUERY & (offs_t == qi), 1 - q_base, q_base)
                kv = tl.where((not FLIP_QUERY) & (offs_t == kj), 1 - k_base, k_base)
                is_match = can_continue & (qv == kv)
                length += is_match.to(tl.int32)

            better = (length > cur_best_len) | ((length == cur_best_len) & (length > 0) & (j > cur_best_j))
            cur_best_len = tl.where(better, length, cur_best_len)
            cur_best_j = tl.where(better, j, cur_best_j)

        gather_idx = cur_best_j + 1
        off_y = tl.where(
            cur_best_j >= 0,
            tl.load(v_ptr_b + gather_idx * stride_v_t, mask=mask_t, other=0),
            0,
        )
        base_y = tl.load(y_ptr_b + i * stride_y_t)
        grad_y = tl.load(go_ptr_b + i * stride_go_t)
        accum += (off_y.to(tl.float32) - base_y.to(tl.float32)) * grad_y

    x_t = tl.load(
        (q_ptr_b if FLIP_QUERY else k_ptr_b) + offs_t * (stride_q_t if FLIP_QUERY else stride_k_t),
        mask=mask_t,
        other=0,
    )
    flip_dir = 1.0 - 2.0 * x_t.to(tl.float32)
    tl.store(out_ptr_b + offs_t * stride_out_t, accum * flip_dir, mask=mask_t)


@triton.jit
def _backward_v_kernel(
    best_j_ptr,
    go_ptr,
    out_ptr,
    T,
    stride_best_j_b,
    stride_best_j_t,
    stride_go_b,
    stride_go_t,
    stride_out_b,
    stride_out_t,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)

    offs_t = pid_blk * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    mask_t = offs_t < T

    best_j_ptr_b = best_j_ptr + pid_b * stride_best_j_b
    go_ptr_b = go_ptr + pid_b * stride_go_b
    out_ptr_b = out_ptr + pid_b * stride_out_b

    accum = tl.zeros((BLOCK_SIZE_T,), dtype=tl.float32)
    for i in range(0, T):
        best_j = tl.load(best_j_ptr_b + i * stride_best_j_t)
        grad_y = tl.load(go_ptr_b + i * stride_go_t)
        hit = mask_t & (best_j >= 0) & (best_j + 1 == offs_t)
        accum += tl.where(hit, grad_y, 0.0)

    tl.store(out_ptr_b + offs_t * stride_out_t, accum, mask=mask_t)


def _reshape_streams(x: Tensor) -> Tuple[Tensor, Tuple[int, int, int]]:
    x = _as_bits(x)
    return _reshape_generic_streams(x)


def _reshape_generic_streams(x: Tensor) -> Tuple[Tensor, Tuple[int, int, int]]:
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


@torch.no_grad()
def qkv1bit_backward_triton(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    best_j: Tensor,
    grad_output: Tensor,
    *,
    K: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("Triton backward requires CUDA tensors")

    q_stream, shape = _reshape_streams(query)
    k_stream, _ = _reshape_streams(key)
    v_stream, _ = _reshape_streams(value)
    y_stream, _ = _reshape_generic_streams(output.to(torch.uint8))
    best_j_stream, _ = _reshape_generic_streams(best_j.to(torch.int32))
    grad_stream, _ = _reshape_generic_streams(grad_output.to(torch.float32))

    batch_streams, seq_len = q_stream.shape
    K = int(max(1, min(K, seq_len)))
    block_size_t = min(128, triton.next_power_of_2(min(seq_len, 128)))
    grid = (batch_streams, triton.cdiv(seq_len, block_size_t))

    dq = torch.empty_like(grad_stream)
    dk = torch.empty_like(grad_stream)
    dv = torch.empty_like(grad_stream)

    _backward_qk_kernel[grid](
        q_stream,
        k_stream,
        v_stream,
        y_stream,
        grad_stream,
        dq,
        seq_len,
        K,
        q_stream.stride(0),
        q_stream.stride(1),
        k_stream.stride(0),
        k_stream.stride(1),
        v_stream.stride(0),
        v_stream.stride(1),
        y_stream.stride(0),
        y_stream.stride(1),
        grad_stream.stride(0),
        grad_stream.stride(1),
        dq.stride(0),
        dq.stride(1),
        FLIP_QUERY=True,
        BLOCK_SIZE_T=block_size_t,
    )
    _backward_qk_kernel[grid](
        q_stream,
        k_stream,
        v_stream,
        y_stream,
        grad_stream,
        dk,
        seq_len,
        K,
        q_stream.stride(0),
        q_stream.stride(1),
        k_stream.stride(0),
        k_stream.stride(1),
        v_stream.stride(0),
        v_stream.stride(1),
        y_stream.stride(0),
        y_stream.stride(1),
        grad_stream.stride(0),
        grad_stream.stride(1),
        dk.stride(0),
        dk.stride(1),
        FLIP_QUERY=False,
        BLOCK_SIZE_T=block_size_t,
    )
    _backward_v_kernel[grid](
        best_j_stream,
        grad_stream,
        dv,
        seq_len,
        best_j_stream.stride(0),
        best_j_stream.stride(1),
        grad_stream.stride(0),
        grad_stream.stride(1),
        dv.stride(0),
        dv.stride(1),
        BLOCK_SIZE_T=block_size_t,
    )

    return (
        _restore_streams(dq, shape).to(torch.float32),
        _restore_streams(dk, shape).to(torch.float32),
        _restore_streams(dv, shape).to(torch.float32),
    )
