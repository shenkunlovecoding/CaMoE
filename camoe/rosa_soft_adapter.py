"""Adapter for the local/installed rosa_soft operators."""

from __future__ import annotations

import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
_SCAN_CPU_WARNING_EMITTED = False


@lru_cache(maxsize=1)
def _load_rosa_soft_ops() -> tuple[Callable, Callable, Callable]:
    build_root = ROOT_DIR / "rosa_soft" / "build"
    candidates = [ROOT_DIR / "rosa_soft"]
    candidates.extend(sorted(build_root.glob("lib.*"), reverse=True))

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        sys.modules.pop("rosa_soft", None)
        try:
            from rosa_soft import rosa_scan_ops, rosa_soft_ops, rosa_sufa_ops

            return rosa_soft_ops, rosa_sufa_ops, rosa_scan_ops
        except ImportError as exc:
            last_error = exc
            continue

    try:
        from rosa_soft import rosa_scan_ops, rosa_soft_ops, rosa_sufa_ops

        return rosa_soft_ops, rosa_sufa_ops, rosa_scan_ops
    except ImportError as exc:
        raise RuntimeError(
            "rosa_soft backend requested, but the rosa_soft package is not importable. "
            "Install it into the active environment or build the local rosa_soft package first."
        ) from (last_error or exc)


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


def rosa_soft(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bits_per_symbol: int,
    mode: str,
    truncation_length: int,
    schmitt_trigger: float = 0.0,
    suffix_factor: float = 0.5,
    quant_mode: str = "soft",
    quant_scale: float | None = None,
) -> torch.Tensor:
    """Run a rosa_soft-family operator on `[B, T, H*bits]` tensors."""

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"rosa_soft expects matching q/k/v shapes, got {q.shape}, {k.shape}, {v.shape}")

    rosa_soft_ops, rosa_sufa_ops, rosa_scan_ops = _load_rosa_soft_ops()
    backend = str(mode).lower()
    if backend not in {"soft", "sufa", "scan"}:
        raise ValueError(f"Unsupported rosa_soft mode: {mode!r}")

    if backend == "scan" and q.device.type != "cuda":
        global _SCAN_CPU_WARNING_EMITTED
        if not _SCAN_CPU_WARNING_EMITTED:
            warnings.warn(
                "rosa_backend='scan' currently requires CUDA/Triton on this platform. "
                "Falling back to 'sufa' for CPU execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            _SCAN_CPU_WARNING_EMITTED = True
        backend = "sufa"

    q_bhtd = _reshape_btd_to_bhtd(q, bits_per_symbol)
    k_bhtd = _reshape_btd_to_bhtd(k, bits_per_symbol)
    v_bhtd = _reshape_btd_to_bhtd(v, bits_per_symbol)

    if backend == "soft":
        out = rosa_soft_ops(
            q_bhtd,
            k_bhtd,
            v_bhtd,
            quant_mode=quant_mode,
            quant_scale=quant_scale,
            schmitt_trigger=schmitt_trigger,
        )
    elif backend == "sufa":
        sufa_kernel = os.getenv("CAMOE_ROSA_SUFA_KERNEL", "auto").strip().lower() or "auto"
        out = rosa_sufa_ops(
            q_bhtd,
            k_bhtd,
            v_bhtd,
            suffix_window=truncation_length,
            suffix_factor=suffix_factor,
            quant_mode=quant_mode,
            quant_scale=quant_scale,
            schmitt_trigger=schmitt_trigger,
            kernel=sufa_kernel,
        )
    else:
        out = rosa_scan_ops(
            q_bhtd,
            k_bhtd,
            v_bhtd,
            suffix_window=truncation_length,
            suffix_factor=suffix_factor,
            quant_mode=quant_mode,
            quant_scale=quant_scale,
            schmitt_trigger=schmitt_trigger,
        )

    return _reshape_bhtd_to_btd(out).to(v.dtype)
