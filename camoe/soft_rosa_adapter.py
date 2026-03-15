"""Adapter for the vendored Soft_ROSA experimental operators."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from .kernel_init import announce_kernel_init


ROOT_DIR = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load_soft_rosa_ops() -> tuple[Callable, Callable, Callable, Callable]:
    root_str = str(ROOT_DIR)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    sys.modules.pop("Soft_ROSA", None)

    try:
        from Soft_ROSA import qkv1bit_forward, qkv1bit_rosa_ops, soft_rosa_forward, soft_rosa_ops

        return soft_rosa_ops, qkv1bit_rosa_ops, qkv1bit_forward, soft_rosa_forward
    except ImportError as exc:
        raise RuntimeError(
            "Soft_ROSA backend requested, but the vendored Soft_ROSA package is not importable."
        ) from exc


def _reshape_btd_to_bhtd(x: torch.Tensor, bits_per_symbol: int) -> torch.Tensor:
    batch, steps, hidden = x.shape
    if hidden % bits_per_symbol != 0:
        raise ValueError(
            f"Hidden size {hidden} must be divisible by bits_per_symbol={bits_per_symbol}."
        )
    heads = hidden // bits_per_symbol
    return x.view(batch, steps, heads, bits_per_symbol).permute(0, 2, 1, 3).contiguous()


def _reshape_bhtd_to_btd(x: torch.Tensor) -> torch.Tensor:
    batch, heads, steps, bits = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(batch, steps, heads * bits)


def soft_rosa_exact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bits_per_symbol: int,
    truncation_length: int,
    scan_backend: str = "auto",
) -> torch.Tensor:
    soft_rosa_ops, _, _, _ = _load_soft_rosa_ops()
    normalized = scan_backend.lower()
    if normalized == "auto" and q.device.type == "cuda":
        normalized = "cuda"
    if normalized in {"cuda", "triton"} and q.device.type == "cuda":
        announce_kernel_init("soft_rosa_exact", normalized)
    q_bhtd = _reshape_btd_to_bhtd(q, bits_per_symbol)
    k_bhtd = _reshape_btd_to_bhtd(k, bits_per_symbol)
    v_bhtd = _reshape_btd_to_bhtd(v, bits_per_symbol)
    out = soft_rosa_ops(
        q_bhtd,
        k_bhtd,
        v_bhtd,
        max_lookback=truncation_length,
        scan_backend=scan_backend,
    )
    return _reshape_bhtd_to_btd(out).to(v.dtype)


def soft_rosa_exact_with_aux(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bits_per_symbol: int,
    truncation_length: int,
    scan_backend: str = "auto",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _, _, _, soft_rosa_forward = _load_soft_rosa_ops()
    normalized = scan_backend.lower()
    if normalized == "auto" and q.device.type == "cuda":
        normalized = "cuda"
    if normalized in {"cuda", "triton"} and q.device.type == "cuda":
        announce_kernel_init("soft_rosa_exact", normalized)
    q_bhtd = _reshape_btd_to_bhtd(q, bits_per_symbol)
    k_bhtd = _reshape_btd_to_bhtd(k, bits_per_symbol)
    v_bhtd = _reshape_btd_to_bhtd(v, bits_per_symbol)

    batch, heads, steps, bits = q_bhtd.shape
    q_flat = q_bhtd.reshape(batch * heads, steps, bits)
    k_flat = k_bhtd.reshape(batch * heads, steps, bits)
    v_flat = v_bhtd.reshape(batch * heads, steps, bits)
    out_flat, aux = soft_rosa_forward(
        q_flat,
        k_flat,
        v_flat,
        max_lookback=truncation_length,
        scan_backend=scan_backend,
        return_aux=True,
    )
    out = out_flat.reshape(batch, heads, steps, bits)

    best_j = aux["best_j"].reshape(batch, heads, steps)
    ell = aux["ell"].reshape(batch, heads, steps, steps)
    gather_idx = best_j.clamp_min(0).unsqueeze(-1)
    chosen_ell = torch.gather(ell, dim=-1, index=gather_idx).squeeze(-1)
    chosen_ell = torch.where(best_j >= 0, chosen_ell, torch.zeros_like(chosen_ell))

    return _reshape_bhtd_to_btd(out).to(v.dtype), {
        "best_j": best_j.permute(0, 2, 1).contiguous(),
        "chosen_ell": chosen_ell.permute(0, 2, 1).contiguous(),
    }


def soft_rosa_qkv1bit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    truncation_length: int,
    backend: str = "auto",
) -> torch.Tensor:
    return soft_rosa_qkv1bit_with_aux(
        q,
        k,
        v,
        truncation_length=truncation_length,
        backend=backend,
    )[0]


def soft_rosa_qkv1bit_with_aux(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    truncation_length: int,
    backend: str = "auto",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"Soft_ROSA qkv1bit expects matching q/k/v shapes, got {q.shape}, {k.shape}, {v.shape}")
    if q.ndim != 3:
        raise ValueError(f"Soft_ROSA qkv1bit expects [B, T, N], got {tuple(q.shape)}")

    _, qkv1bit_rosa_ops, qkv1bit_forward, _ = _load_soft_rosa_ops()
    normalized = backend.lower()
    if normalized == "auto":
        if q.device.type == "cuda":
            normalized = "cuda"
        else:
            normalized = "reference"
    if normalized in {"cuda", "triton"} and q.device.type == "cuda":
        announce_kernel_init("soft_rosa_qkv1bit", normalized)
    out = qkv1bit_rosa_ops(q, k, v, K=truncation_length, backend=normalized).to(v.dtype)
    with torch.no_grad():
        _, aux = qkv1bit_forward(q, k, v, K=truncation_length, backend=normalized, return_aux=True)
    return out, aux
