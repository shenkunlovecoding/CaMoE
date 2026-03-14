from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch

from qkv1bit import (
    finite_diff_bwd_all_channels,
    finite_diff_bwd_channelwise,
    hard_qkv1bit_forward,
    qkv1bit_forward,
    qkv1bit_rosa,
)


@dataclass
class BenchmarkRow:
    name: str
    time_ms: float
    speedup_vs_channelwise: float
    forward_ratio: float


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_bits(batch: int, seq_len: int, num_streams: int, device: str) -> Tuple[torch.Tensor, ...]:
    q = torch.randint(0, 2, (batch, seq_len, num_streams), device=device, dtype=torch.long)
    k = torch.randint(0, 2, (batch, seq_len, num_streams), device=device, dtype=torch.long)
    v = torch.randint(0, 2, (batch, seq_len, num_streams), device=device, dtype=torch.long)
    dy = torch.randn(batch, seq_len, num_streams, device=device, dtype=torch.float32)
    return q, k, v, dy


def bits_to_logits(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.float32).mul(2.0).sub(1.0)


def available_backends(device: str) -> List[str]:
    backends = ["reference"]
    if device.startswith("cuda"):
        backends += ["triton", "cuda"]
    return backends


def check_channel_independence(device: str) -> None:
    torch.manual_seed(0)
    q, k, v, _ = make_bits(batch=2, seq_len=10, num_streams=6, device=device)
    base = hard_qkv1bit_forward(q, k, v, K=5)
    t = 4

    q_all = q.clone()
    q_all[:, t, :] = 1 - q_all[:, t, :]
    off_all = hard_qkv1bit_forward(q_all, k, v, K=5)

    worst = 0.0
    for n in range(q.size(-1)):
        q_one = q.clone()
        q_one[:, t, n] = 1 - q_one[:, t, n]
        off_one = hard_qkv1bit_forward(q_one, k, v, K=5)
        diff = (off_all[:, :, n] - base[:, :, n]) - (off_one[:, :, n] - base[:, :, n])
        worst = max(worst, diff.abs().max().item())

    print("[1] Channel independence check")
    print(f"max channel-slice difference after simultaneous flip vs single flip: {worst:.6f}")
    print()


def _spec_cases(device: str) -> List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    cases = []

    # tie-break under K=1: multiple past matches with equal length, must choose the largest j
    q = torch.tensor([[[1], [0], [1], [1]]], device=device, dtype=torch.long)
    k = q.clone()
    v = torch.tensor([[[0], [1], [0], [1]]], device=device, dtype=torch.long)
    cases.append(("tie_break_k1", q, k, v, 1))

    # no-match must return zero and best_j = -1
    q = torch.tensor([[[0], [0], [0]]], device=device, dtype=torch.long)
    k = torch.tensor([[[1], [1], [1]]], device=device, dtype=torch.long)
    v = torch.tensor([[[1], [0], [1]]], device=device, dtype=torch.long)
    cases.append(("no_match", q, k, v, 3))

    # K truncation must clamp the reported best_len
    q = torch.tensor([[[1], [1], [1], [1], [1]]], device=device, dtype=torch.long)
    k = q.clone()
    v = q.clone()
    cases.append(("k_truncation", q, k, v, 2))

    # T = 1 edge case
    q = torch.tensor([[[1], [0]]], device=device, dtype=torch.long)[:, :1, :]
    k = q.clone()
    v = q.clone()
    cases.append(("t_eq_1", q, k, v, 1))

    # flat streams random parity
    torch.manual_seed(123)
    q = torch.randint(0, 2, (2, 7, 11), device=device, dtype=torch.long)
    k = torch.randint(0, 2, (2, 7, 11), device=device, dtype=torch.long)
    v = torch.randint(0, 2, (2, 7, 11), device=device, dtype=torch.long)
    cases.append(("flat_streams_random", q, k, v, 4))

    return cases


def check_forward_parity(device: str) -> None:
    print("[2] Forward parity and spec checks")
    for name, q, k, v, K in _spec_cases(device):
        ref_y, ref_aux = hard_qkv1bit_forward(q, k, v, K=K, return_aux=True)
        print(f"case={name}")
        for backend in available_backends(device):
            y, aux = qkv1bit_forward(q, k, v, K=K, backend=backend, return_aux=True)
            y_diff = (y - ref_y).abs().max().item()
            j_diff = (aux["best_j"] - ref_aux["best_j"]).abs().max().item()
            l_diff = (aux["best_len"] - ref_aux["best_len"]).abs().max().item()
            print(
                f"  backend={backend:<9} "
                f"max|y-ref|={y_diff:.6f} "
                f"max|j-ref|={j_diff:.6f} "
                f"max|len-ref|={l_diff:.6f}"
            )
    print()


def check_gradient_parity(device: str) -> None:
    print("[3] Gradient parity against channelwise reference")
    torch.manual_seed(1)
    q_bits, k_bits, v_bits, dy = make_bits(batch=2, seq_len=12, num_streams=8, device=device)
    dq_ref, dk_ref, dv_ref = finite_diff_bwd_channelwise(q_bits, k_bits, v_bits, dy, K=6)

    for backend in available_backends(device):
        q = bits_to_logits(q_bits).detach().clone().requires_grad_(True)
        k = bits_to_logits(k_bits).detach().clone().requires_grad_(True)
        v = bits_to_logits(v_bits).detach().clone().requires_grad_(True)

        out = qkv1bit_rosa(q, k, v, K=6, backend=backend)
        loss = (out * dy).sum()
        loss.backward()

        dq_diff = (q.grad - dq_ref).abs().max().item()
        dk_diff = (k.grad - dk_ref).abs().max().item()
        dv_diff = (v.grad - dv_ref).abs().max().item()
        print(
            f"backend={backend:<9} "
            f"max|dq-ref|={dq_diff:.6f} "
            f"max|dk-ref|={dk_diff:.6f} "
            f"max|dv-ref|={dv_diff:.6f}"
        )
    print()


def timed_run(fn, *, repeat: int, device: str) -> float:
    times = []
    for _ in range(repeat):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return statistics.median(times)


def _benchmark_backend(name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dy: torch.Tensor, K: int):
    if name == "ref-channelwise":
        return lambda: finite_diff_bwd_channelwise(q, k, v, dy, K=K)
    if name == "ref-all-channels":
        return lambda: finite_diff_bwd_all_channels(q, k, v, dy, K=K, backend="reference")
    if name == "triton":
        return lambda: finite_diff_bwd_all_channels(q, k, v, dy, K=K, backend="triton")
    if name == "cuda":
        return lambda: finite_diff_bwd_all_channels(q, k, v, dy, K=K, backend="cuda")
    raise ValueError(name)


def benchmark_suite(device: str, repeat: int) -> None:
    print("[4] Benchmark")
    print("This benchmark measures the specialized backward path.")
    print("Theoretical ratio is computed from forward-call counts only.")
    print()

    cases = [
        (1, 16, 8, 6),
        (1, 12, 16, 6),
        (1, 12, 32, 6),
    ]

    backend_names = ["ref-channelwise", "ref-all-channels"]
    if device.startswith("cuda"):
        backend_names += ["triton", "cuda"]

    for batch, seq_len, num_streams, K in cases:
        torch.manual_seed(1234)
        q, k, v, dy = make_bits(batch=batch, seq_len=seq_len, num_streams=num_streams, device=device)

        for name in backend_names:
            _benchmark_backend(name, q, k, v, dy, K)()

        timings: Dict[str, float] = {}
        for name in backend_names:
            timings[name] = timed_run(
                _benchmark_backend(name, q, k, v, dy, K),
                repeat=repeat,
                device=device,
            )

        channelwise_time = timings["ref-channelwise"]
        print(f"B={batch} T={seq_len} N={num_streams} K={K}")
        for name in backend_names:
            forward_ratio = 1.0 if name == "ref-channelwise" else (1 + 3 * seq_len * num_streams) / (1 + 3 * seq_len)
            speedup = channelwise_time / timings[name]
            row = BenchmarkRow(
                name=name,
                time_ms=timings[name],
                speedup_vs_channelwise=speedup,
                forward_ratio=forward_ratio,
            )
            print(
                f"  backend={row.name:<16} "
                f"time={row.time_ms:>8.2f} ms "
                f"speedup={row.speedup_vs_channelwise:>6.2f}x "
                f"theoretical={row.forward_ratio:>6.2f}x"
            )
        print()


def main() -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="QKV-1bit ROSA backend demo and benchmark")
    parser.add_argument("--device", default=default_device, help="cpu or cuda")
    parser.add_argument("--repeat", type=int, default=3, help="benchmark repeats")
    args = parser.parse_args()

    print("QKV-1bit ROSA operator demo")
    print(f"device={args.device}")
    print()

    check_channel_independence(args.device)
    check_forward_parity(args.device)
    check_gradient_parity(args.device)
    benchmark_suite(args.device, repeat=args.repeat)


if __name__ == "__main__":
    main()
