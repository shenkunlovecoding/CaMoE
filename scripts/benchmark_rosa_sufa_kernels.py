from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
ROSA_SOFT_SOURCE_ROOT = ROOT_DIR / "rosa_soft"
if str(ROSA_SOFT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROSA_SOFT_SOURCE_ROOT))

from rosa_soft.rosa_sufa import rosa_sufa_ops


def _measure_ms(fn, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeat


def benchmark_kernel(
    *,
    kernel: str,
    batch: int,
    heads: int,
    steps: int,
    bits: int,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    q = torch.randn(batch, heads, steps, bits, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(batch, heads, steps, bits, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(batch, heads, steps, bits, device="cuda", dtype=torch.float32, requires_grad=True)

    def forward_only() -> None:
        out = rosa_sufa_ops(
            q,
            k,
            v,
            suffix_window=8,
            suffix_factor=0.5,
            quant_mode="soft",
            quant_scale=None,
            schmitt_trigger=0.0,
            kernel=kernel,
        )
        torch.cuda.synchronize()
        del out

    def step() -> None:
        out = rosa_sufa_ops(
            q,
            k,
            v,
            suffix_window=8,
            suffix_factor=0.5,
            quant_mode="soft",
            quant_scale=None,
            schmitt_trigger=0.0,
            kernel=kernel,
        )
        out.sum().backward()
        q.grad = None
        k.grad = None
        v.grad = None

    forward_ms = _measure_ms(forward_only, warmup=warmup, repeat=repeat)
    total_ms = _measure_ms(step, warmup=warmup, repeat=repeat)
    return forward_ms, total_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark experimental rosa_sufa_ops kernels.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    kernels = ("torch", "proxy_triton", "truncated_cuda")
    print(f"{'kernel':<16} {'forward_ms':>12} {'step_ms':>12}")
    print("-" * 42)
    for kernel in kernels:
        try:
            forward_ms, total_ms = benchmark_kernel(
                kernel=kernel,
                batch=args.batch,
                heads=args.heads,
                steps=args.steps,
                bits=args.bits,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            print(f"{kernel:<16} {forward_ms:>12.3f} {total_ms:>12.3f}")
        except Exception as exc:
            print(f"{kernel:<16} {'skip':>12} {'skip':>12}  # {exc}")


if __name__ == "__main__":
    main()
