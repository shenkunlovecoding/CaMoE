from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from camoe.expert_rosa import ROSAExpert


def _measure_ms(fn, *, warmup: int, repeat: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeat


def _benchmark_case(
    *,
    backend: str,
    batch: int,
    steps: int,
    heads: int,
    bits: int,
    truncation_length: int,
    warmup: int,
    repeat: int,
    device: str,
) -> tuple[float, float]:
    dim = heads * bits
    expert = ROSAExpert(
        dim=dim,
        slim_heads=heads,
        bits_per_symbol=bits,
        backend=backend,
        truncation_length=truncation_length,
        sequence_length=steps,
    ).to(device)
    expert.train()

    x = torch.randn(batch, steps, dim, device=device, dtype=torch.float32, requires_grad=True)

    def forward_only() -> None:
        out = expert(x)
        if device == "cuda":
            torch.cuda.synchronize()
        del out

    def step_once() -> None:
        out = expert(x)
        out.square().mean().backward()
        x.grad = None
        expert.zero_grad(set_to_none=True)

    forward_ms = _measure_ms(forward_only, warmup=warmup, repeat=repeat, device=device)
    step_ms = _measure_ms(step_once, warmup=warmup, repeat=repeat, device=device)
    return forward_ms, step_ms


def _default_backends(bits: int) -> list[str]:
    common = ["wind", "soft", "sufa", "scan", "soft_exact", "soft_exact_serial"]
    if torch.cuda.is_available():
        common.extend(["soft_exact_cuda", "soft_exact_triton"])
    if bits == 1:
        common.append("soft_qkv1bit")
        if torch.cuda.is_available():
            common.extend(["soft_qkv1bit_cuda", "soft_qkv1bit_triton"])
    return common


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ROSAExpert backends.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--truncation-length", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backends", nargs="*", default=None)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    backends = args.backends or _default_backends(args.bits)
    print(
        f"device={device} batch={args.batch} steps={args.steps} "
        f"heads={args.heads} bits={args.bits} trunc={args.truncation_length}"
    )
    print(f"{'backend':<24} {'forward_ms':>12} {'step_ms':>12}")
    print("-" * 52)
    for backend in backends:
        try:
            forward_ms, step_ms = _benchmark_case(
                backend=backend,
                batch=args.batch,
                steps=args.steps,
                heads=args.heads,
                bits=args.bits,
                truncation_length=args.truncation_length,
                warmup=args.warmup,
                repeat=args.repeat,
                device=device,
            )
            print(f"{backend:<24} {forward_ms:>12.3f} {step_ms:>12.3f}")
        except Exception as exc:
            print(f"{backend:<24} {'skip':>12} {'skip':>12}  # {exc}")


if __name__ == "__main__":
    main()
