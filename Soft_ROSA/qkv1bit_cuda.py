from __future__ import annotations

import hashlib
import os
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


def _ensure_loaded() -> None:
    if hasattr(torch.ops, "soft_rosa_qkv1bit") and hasattr(torch.ops.soft_rosa_qkv1bit, "forward"):
        return

    os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")

    root = Path(__file__).resolve().parent
    source_paths = [root / "csrc" / "qkv1bit.cpp", root / "csrc" / "qkv1bit.cu"]
    digest = hashlib.sha1()
    for path in source_paths:
        digest.update(path.read_bytes())
    ext_name = f"soft_rosa_qkv1bit_ext_{digest.hexdigest()[:10]}"
    load(
        name=ext_name,
        sources=[str(path) for path in source_paths],
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


@torch.no_grad()
def qkv1bit_backward_cuda(
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
        raise ValueError("CUDA backward requires CUDA tensors")

    _ensure_loaded()

    q_stream, shape = _reshape_streams(query)
    k_stream, _ = _reshape_streams(key)
    v_stream, _ = _reshape_streams(value)
    y_stream, _ = _reshape_generic_streams(output.to(torch.uint8))
    best_j_stream, _ = _reshape_generic_streams(best_j.to(torch.long))
    grad_stream, _ = _reshape_generic_streams(grad_output.to(torch.float32))

    dq, dk, dv = torch.ops.soft_rosa_qkv1bit.backward(
        q_stream.contiguous(),
        k_stream.contiguous(),
        v_stream.contiguous(),
        y_stream.contiguous(),
        best_j_stream.contiguous(),
        grad_stream.contiguous(),
        int(K),
    )
    return (
        _restore_streams(dq.to(torch.float32), shape),
        _restore_streams(dk.to(torch.float32), shape),
        _restore_streams(dv.to(torch.float32), shape),
    )
