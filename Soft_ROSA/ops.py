from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

try:
    from .qkv1bit import qkv1bit_rosa
    from .soft_rosa import soft_rosa_forward
except ImportError:
    from qkv1bit import qkv1bit_rosa
    from soft_rosa import soft_rosa_forward

Tensor = torch.Tensor

__all__ = [
    "soft_rosa_ops",
    "soft_rosa_serial_ops",
    "soft_rosa_parallel_ops",
    "qkv1bit_rosa_ops",
]


def _reshape_bhtd(query: Tensor, key: Tensor, value: Tensor):
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Expected query/key/value with shape [B, H, T, D]")
    if query.shape[:3] != key.shape[:3] or query.shape[:3] != value.shape[:3]:
        raise ValueError("query/key/value must agree on [B, H, T]")

    batch, heads, seq_len, dim = query.shape
    value_dim = value.size(-1)
    q = query.reshape(batch * heads, seq_len, dim)
    k = key.reshape(batch * heads, seq_len, dim)
    v = value.reshape(batch * heads, seq_len, value_dim)
    return q, k, v, batch, heads, seq_len, value_dim


def _restore_bhtd(output: Tensor, batch: int, heads: int, seq_len: int, value_dim: int) -> Tensor:
    return output.reshape(batch, heads, seq_len, value_dim)


def _soft_rosa_ops_impl(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    alpha: float,
    gamma: float,
    similarity: str,
    max_lookback: Optional[int],
    scan_backend: str,
) -> Tensor:
    q, k, v, batch, heads, seq_len, value_dim = _reshape_bhtd(query, key, value)
    out = soft_rosa_forward(
        q,
        k,
        v,
        alpha=alpha,
        gamma=gamma,
        similarity=similarity,
        max_lookback=max_lookback,
        scan_backend=scan_backend,
    )
    return _restore_bhtd(out, batch, heads, seq_len, value_dim)


def soft_rosa_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    alpha: float = 10.0,
    gamma: float = 5.0,
    similarity: str = "cosine_margin",
    max_lookback: Optional[int] = None,
    scan_backend: str = "auto",
) -> Tensor:
    """rosa_soft-like operator wrapper for exact Soft ROSA."""

    return _soft_rosa_ops_impl(
        query,
        key,
        value,
        alpha=alpha,
        gamma=gamma,
        similarity=similarity,
        max_lookback=max_lookback,
        scan_backend=scan_backend,
    )


def soft_rosa_serial_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    alpha: float = 10.0,
    gamma: float = 5.0,
    similarity: str = "cosine_margin",
    max_lookback: Optional[int] = None,
) -> Tensor:
    return _soft_rosa_ops_impl(
        query,
        key,
        value,
        alpha=alpha,
        gamma=gamma,
        similarity=similarity,
        max_lookback=max_lookback,
        scan_backend="serial",
    )


def soft_rosa_parallel_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    alpha: float = 10.0,
    gamma: float = 5.0,
    similarity: str = "cosine_margin",
    max_lookback: Optional[int] = None,
) -> Tensor:
    return _soft_rosa_ops_impl(
        query,
        key,
        value,
        alpha=alpha,
        gamma=gamma,
        similarity=similarity,
        max_lookback=max_lookback,
        scan_backend="parallel",
    )


def qkv1bit_rosa_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    K: int,
    backend: str = "triton",
) -> Tensor:
    """Hard QKV-1bit operator with wind_rosa-style flattened stream layout."""

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Expected query/key/value with shape [B, T, H*C]")
    return qkv1bit_rosa(query, key, value, K=K, backend=backend)
