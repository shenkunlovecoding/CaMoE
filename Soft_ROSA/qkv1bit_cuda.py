from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.cpp_extension import load

Tensor = torch.Tensor


def _as_bits(x: Tensor) -> Tensor:
    if x.dtype == torch.bool:
        return x.to(torch.uint8)
    if x.is_floating_point():
        return (x > 0).to(torch.uint8)
    return x.to(torch.uint8)


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


def _ensure_loaded() -> None:
    if hasattr(torch.ops, "soft_rosa_qkv1bit") and hasattr(torch.ops.soft_rosa_qkv1bit, "forward"):
        return

    root = Path(__file__).resolve().parent
    sources = [
        str(root / "csrc" / "qkv1bit.cpp"),
        str(root / "csrc" / "qkv1bit.cu"),
    ]
    load(
        name="soft_rosa_qkv1bit_ext",
        sources=sources,
        is_python_module=False,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
    )


@torch.no_grad()
def qkv1bit_forward_cuda(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    K: int,
    return_aux: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("CUDA backend requires CUDA tensors")

    _ensure_loaded()

    q_stream, shape = _reshape_streams(query)
    k_stream, _ = _reshape_streams(key)
    v_stream, _ = _reshape_streams(value)

    output, best_j, best_len = torch.ops.soft_rosa_qkv1bit.forward(
        q_stream.contiguous(),
        k_stream.contiguous(),
        v_stream.contiguous(),
        int(K),
    )

    output = _restore_streams(output.to(torch.float32), shape)
    if not return_aux:
        return output

    aux = {
        "best_j": _restore_streams(best_j.to(torch.long), shape),
        "best_len": _restore_streams(best_len.to(torch.long), shape),
    }
    return output, aux
