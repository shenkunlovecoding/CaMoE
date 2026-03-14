from __future__ import annotations

import argparse
import importlib
import statistics
import sys
import time
from typing import Callable, Dict, Iterable, Optional

import torch

if __package__ is None or __package__ == "":
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import soft_rosa as soft_rosa_pkg
else:
    from . import (
        diagonal_affine_scan_with_backend,
        qkv1bit_rosa_ops,
        soft_rosa_ops,
        soft_rosa_parallel_ops,
        soft_rosa_serial_ops,
    )

if __package__ is None or __package__ == "":
    soft_rosa_ops = soft_rosa_pkg.soft_rosa_ops
    soft_rosa_parallel_ops = soft_rosa_pkg.soft_rosa_parallel_ops
    soft_rosa_serial_ops = soft_rosa_pkg.soft_rosa_serial_ops
    qkv1bit_rosa_ops = soft_rosa_pkg.qkv1bit_rosa_ops
    diagonal_affine_scan_with_backend = soft_rosa_pkg.diagonal_affine_scan_with_backend


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed_run(fn: Callable[[], None], *, repeat: int, device: str) -> float:
    times = []
    for _ in range(repeat):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return statistics.median(times)


def _load_rosa_soft_project():
    try:
        return importlib.import_module("rosa_soft.rosa_soft")
    except Exception:
        return None


def benchmark_scan(device: str, repeat: int) -> None:
    print("[1] Serial vs Parallel Diagonal Scan")
    if not device.startswith("cuda"):
        print("CUDA is required for the parallel Triton scan benchmark.")
        print()
        return

    cases = [
        (4, 64),
        (8, 128),
        (8, 256),
    ]

    for batch_streams, seq_len in cases:
        torch.manual_seed(0)
        x = torch.rand(batch_streams, seq_len, seq_len, device=device, dtype=torch.float32)
        i_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)
        j_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len)
        x = x * (j_idx < i_idx)

        y_serial = diagonal_affine_scan_with_backend(x, backend="serial")
        y_parallel = diagonal_affine_scan_with_backend(x, backend="parallel")
        max_diff = (y_serial - y_parallel).abs().max().item()

        serial_ms = timed_run(
            lambda: diagonal_affine_scan_with_backend(x, backend="serial"),
            repeat=repeat,
            device=device,
        )
        parallel_ms = timed_run(
            lambda: diagonal_affine_scan_with_backend(x, backend="parallel"),
            repeat=repeat,
            device=device,
        )

        print(
            f"B*H={batch_streams} T={seq_len} | "
            f"serial={serial_ms:.2f} ms | "
            f"parallel={parallel_ms:.2f} ms | "
            f"speedup={serial_ms / parallel_ms:.2f}x | "
            f"maxdiff={max_diff:.3e}"
        )
    print()


def benchmark_ops(device: str, repeat: int) -> None:
    print("[2] Cross-project Operator Benchmark")
    print("This compares operator-style APIs, not identical semantics.")
    print("`soft_rosa_*` uses exact soft DP, while `rosa_soft_ops` uses hard forward + soft backward proxy.")
    print()

    project = _load_rosa_soft_project()
    available: Dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "soft_rosa_serial_ops": lambda q, k, v: soft_rosa_serial_ops(q, k, v, alpha=10.0, gamma=5.0),
        "soft_rosa_parallel_ops": lambda q, k, v: soft_rosa_parallel_ops(q, k, v, alpha=10.0, gamma=5.0),
        "soft_rosa_ops": lambda q, k, v: soft_rosa_ops(q, k, v, alpha=10.0, gamma=5.0),
    }

    if project is not None:
        available["rosa_soft_ops"] = lambda q, k, v: project.rosa_soft_ops(q, k, v)
        available["rosa_sufa_ops"] = lambda q, k, v: project.rosa_sufa_ops(q, k, v)
        available["rosa_scan_ops"] = lambda q, k, v: project.rosa_scan_ops(q, k, v)

    cases = [
        (1, 2, 64, 8),
        (1, 4, 128, 8),
    ]

    for batch, heads, seq_len, dim in cases:
        print(f"B={batch} H={heads} T={seq_len} D={dim}")
        torch.manual_seed(0)
        q = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float32)

        for name, fn in available.items():
            qx = q.detach().clone().requires_grad_(True)
            kx = k.detach().clone().requires_grad_(True)
            vx = v.detach().clone().requires_grad_(True)

            def run():
                out = fn(qx, kx, vx)
                loss = out.square().mean()
                grads = torch.autograd.grad(loss, (qx, kx, vx), retain_graph=False, create_graph=False)
                return grads

            try:
                run()
                ms = timed_run(run, repeat=repeat, device=device)
                print(f"  op={name:<22} time={ms:8.2f} ms")
            except Exception as e:
                print(f"  op={name:<22} failed={type(e).__name__}: {e}")
        print()


def benchmark_qkv1bit_ops(device: str, repeat: int) -> None:
    print("[3] QKV-1bit Operator Benchmark")
    print("This benchmarks the autograd operator wrapper `qkv1bit_rosa_ops`.")
    print("All three backends share the same specialized all-channels-at-once backward logic.")
    print()

    backends = ["reference"]
    if device.startswith("cuda"):
        backends += ["triton", "cuda"]

    cases = [
        (1, 16, 8, 6),
        (1, 12, 16, 6),
        (1, 12, 32, 6),
    ]

    for batch, seq_len, num_streams, K in cases:
        print(f"B={batch} T={seq_len} N={num_streams} K={K}")
        torch.manual_seed(1234)
        q = torch.randint(0, 2, (batch, seq_len, num_streams), device=device, dtype=torch.float32).mul(2).sub(1)
        k = torch.randint(0, 2, (batch, seq_len, num_streams), device=device, dtype=torch.float32).mul(2).sub(1)
        v = torch.randint(0, 2, (batch, seq_len, num_streams), device=device, dtype=torch.float32).mul(2).sub(1)

        timings: Dict[str, float] = {}
        for backend in backends:
            qx = q.detach().clone().requires_grad_(True)
            kx = k.detach().clone().requires_grad_(True)
            vx = v.detach().clone().requires_grad_(True)

            def run():
                out = qkv1bit_rosa_ops(qx, kx, vx, K=K, backend=backend)
                loss = out.square().mean()
                torch.autograd.grad(loss, (qx, kx, vx), retain_graph=False, create_graph=False)

            run()
            timings[backend] = timed_run(run, repeat=repeat, device=device)

        ref_time = timings["reference"]
        for backend in backends:
            print(
                f"  backend={backend:<10} "
                f"time={timings[backend]:8.2f} ms "
                f"speedup={ref_time / timings[backend]:6.2f}x"
            )
        print()


def main() -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Soft ROSA operator benchmark")
    parser.add_argument("--device", default=default_device, help="cpu or cuda")
    parser.add_argument("--repeat", type=int, default=3, help="benchmark repeats")
    args = parser.parse_args()

    print("Soft ROSA benchmark suite")
    print(f"device={args.device}")
    print()

    benchmark_scan(args.device, args.repeat)
    benchmark_ops(args.device, args.repeat)
    benchmark_qkv1bit_ops(args.device, args.repeat)


if __name__ == "__main__":
    main()
