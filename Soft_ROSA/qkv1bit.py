from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

Tensor = torch.Tensor


def _load_triton_backend():
    try:
        from .qkv1bit_triton import qkv1bit_forward_triton
    except ImportError:
        from qkv1bit_triton import qkv1bit_forward_triton
    return qkv1bit_forward_triton


def _load_cuda_backend():
    try:
        from .qkv1bit_cuda import qkv1bit_forward_cuda
    except ImportError:
        from qkv1bit_cuda import qkv1bit_forward_cuda
    return qkv1bit_forward_cuda


def _as_bits(x: Tensor) -> Tensor:
    if x.dtype == torch.bool:
        return x.to(torch.long)
    if x.is_floating_point():
        return (x > 0).to(torch.long)
    return x.to(torch.long)


@torch.no_grad()
def hard_qkv1bit_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    K: int,
    return_aux: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
    """Reference Hard ROSA forward for QKV-1bit.

    Inputs are shaped [B, T, N] where N is a flat collection of independent
    1-bit streams (for example H*C from a wind_rosa style interface).
    """

    q = _as_bits(query)
    k = _as_bits(key)
    v = _as_bits(value)

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("query, key, value must all have shape [B, T, N]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("query, key, value must have identical shape [B, T, N]")

    batch, seq_len, num_streams = q.shape
    K = int(max(1, min(K, seq_len)))

    best_len = torch.zeros(batch, seq_len, num_streams, dtype=torch.long, device=q.device)
    best_j = torch.full((batch, seq_len, num_streams), -1, dtype=torch.long, device=q.device)

    for d in range(1, seq_len):
        diag_len = seq_len - d
        prev = torch.zeros(batch, num_streams, dtype=torch.long, device=q.device)
        diag_eq = (q[:, d:, :] == k[:, :diag_len, :]).to(torch.long)

        for t in range(diag_len):
            cur = diag_eq[:, t, :] * (prev + 1)
            cur = cur.clamp_max(K)

            i = d + t
            j = t
            current_best_len = best_len[:, i, :]
            current_best_j = best_j[:, i, :]
            better = (cur > current_best_len) | (
                (cur == current_best_len) & (cur > 0) & (j > current_best_j)
            )

            best_len[:, i, :] = torch.where(better, cur, current_best_len)
            best_j[:, i, :] = torch.where(
                better,
                torch.full_like(current_best_j, j),
                current_best_j,
            )
            prev = cur

    gather_idx = (best_j + 1).clamp(0, seq_len - 1)
    output = torch.gather(v, dim=1, index=gather_idx)
    output = torch.where(best_j >= 0, output, torch.zeros_like(output))
    output = output.to(torch.float32)

    if not return_aux:
        return output
    return output, {"best_j": best_j, "best_len": best_len}


@torch.no_grad()
def qkv1bit_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    K: int,
    backend: str = "reference",
    return_aux: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
    backend = backend.lower()
    if backend in {"reference", "ref", "ref-all-channels"}:
        return hard_qkv1bit_forward(query, key, value, K=K, return_aux=return_aux)

    if backend == "triton":
        return _load_triton_backend()(query, key, value, K=K, return_aux=return_aux)

    if backend == "cuda":
        return _load_cuda_backend()(query, key, value, K=K, return_aux=return_aux)

    raise ValueError(f"Unsupported backend={backend!r}")


@torch.no_grad()
def finite_diff_bwd_channelwise(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dy: Tensor,
    *,
    K: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Naive finite differences: flip one stream bit at a time."""

    q = _as_bits(query).clone()
    k = _as_bits(key).clone()
    v = _as_bits(value).clone()
    dy = dy.to(torch.float32)

    if q.shape != dy.shape:
        raise ValueError("dy must have the same shape [B, T, N] as q/k/v")

    base = hard_qkv1bit_forward(q, k, v, K=K).to(torch.float32)
    grads = []

    for x in (q, k, v):
        dx = torch.empty_like(base)
        _, seq_len, num_streams = x.shape

        for t in range(seq_len):
            for n in range(num_streams):
                x[:, t, n] = 1 - x[:, t, n]
                off = hard_qkv1bit_forward(q, k, v, K=K).to(torch.float32)
                x[:, t, n] = 1 - x[:, t, n]

                change = ((off - base) * dy).sum(dim=(1, 2))
                flip_dir = 1.0 - 2.0 * x[:, t, n].to(torch.float32)
                dx[:, t, n] = change * flip_dir

        grads.append(dx)

    return tuple(grads)  # type: ignore[return-value]


@torch.no_grad()
def finite_diff_bwd_all_channels(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dy: Tensor,
    *,
    K: int,
    backend: str = "reference",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Specialized QKV-1bit finite differences: flip all streams at once."""

    q = _as_bits(query).clone()
    k = _as_bits(key).clone()
    v = _as_bits(value).clone()
    dy = dy.to(torch.float32)

    if q.shape != dy.shape:
        raise ValueError("dy must have the same shape [B, T, N] as q/k/v")

    base = qkv1bit_forward(q, k, v, K=K, backend=backend).to(torch.float32)
    grads = []

    for x in (q, k, v):
        dx = torch.empty_like(base)
        _, seq_len, _ = x.shape

        for t in range(seq_len):
            x[:, t, :] = 1 - x[:, t, :]
            off = qkv1bit_forward(q, k, v, K=K, backend=backend).to(torch.float32)
            x[:, t, :] = 1 - x[:, t, :]

            change_per_stream = ((off - base) * dy).sum(dim=1)
            flip_dir = 1.0 - 2.0 * x[:, t, :].to(torch.float32)
            dx[:, t, :] = change_per_stream * flip_dir

        grads.append(dx)

    return tuple(grads)  # type: ignore[return-value]


class _QKV1BitRosaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, K: int, backend: str):
        q_bits = _as_bits(query)
        k_bits = _as_bits(key)
        v_bits = _as_bits(value)
        out = qkv1bit_forward(q_bits, k_bits, v_bits, K=K, backend=backend)
        ctx.save_for_backward(q_bits, k_bits, v_bits)
        ctx.K = int(K)
        ctx.backend = backend
        return out.to(torch.float32)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        q_bits, k_bits, v_bits = ctx.saved_tensors
        dq, dk, dv = finite_diff_bwd_all_channels(
            q_bits,
            k_bits,
            v_bits,
            grad_output.to(torch.float32),
            K=ctx.K,
            backend=ctx.backend,
        )
        return dq, dk, dv, None, None


def qkv1bit_rosa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    K: int,
    backend: str = "triton",
) -> Tensor:
    """Autograd-compatible QKV-1bit ROSA operator."""

    return _QKV1BitRosaFunction.apply(query, key, value, int(K), backend)
