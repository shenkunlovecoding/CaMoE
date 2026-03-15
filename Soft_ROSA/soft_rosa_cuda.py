from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

Tensor = torch.Tensor


def _ensure_loaded() -> None:
    if hasattr(torch.ops, "soft_rosa_scan") and hasattr(torch.ops.soft_rosa_scan, "forward"):
        return

    os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")

    root = Path(__file__).resolve().parent
    source_paths = [root / "csrc" / "soft_rosa_scan.cpp", root / "csrc" / "soft_rosa_scan.cu"]
    digest = hashlib.sha1()
    for path in source_paths:
        digest.update(path.read_bytes())
    ext_name = f"soft_rosa_scan_ext_{digest.hexdigest()[:10]}"
    load(
        name=ext_name,
        sources=[str(path) for path in source_paths],
        is_python_module=False,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
    )


class _DiagonalScanCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        if not x.is_cuda:
            raise ValueError("CUDA diagonal scan requires CUDA tensors")
        _ensure_loaded()
        x32 = x.contiguous().to(torch.float32)
        out = torch.ops.soft_rosa_scan.forward(x32)
        ctx.save_for_backward(x32, out)
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x32, out = ctx.saved_tensors
        grad_output = grad_output.contiguous().to(torch.float32)
        grad_input = torch.ops.soft_rosa_scan.backward(x32, out, grad_output)
        return grad_input.to(grad_output.dtype)


def diagonal_affine_scan_cuda(x: Tensor) -> Tensor:
    return _DiagonalScanCudaFunction.apply(x)
