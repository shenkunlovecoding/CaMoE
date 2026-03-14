from __future__ import annotations

import os
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
ROSA_SOFT_SOURCE_ROOT = ROOT_DIR / "rosa_soft"
if str(ROSA_SOFT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROSA_SOFT_SOURCE_ROOT))

from camoe.expert_rosa import ROSAExpert
import rosa_soft.rosa_sufa as rosa_sufa_module
from rosa_soft.rosa_sufa import rosa_sufa_ops, suffix_attention_proxy
from rosa_soft.sufa_triton import is_triton_window_available
from rosa_soft.sufa_truncated_cuda import truncated_cuda_hard_forward


def _pack_bits_reference(x: torch.Tensor) -> torch.Tensor:
    bits = x.to(torch.int64)
    shifts = torch.arange(bits.size(-1), dtype=torch.int64)
    return (bits << shifts.view(1, 1, 1, -1)).sum(dim=-1)


def _decode_reference_symbols(
    packed_symbols: torch.Tensor,
    match_length: torch.Tensor,
    bits_per_symbol: int,
) -> torch.Tensor:
    shifts = torch.arange(bits_per_symbol, dtype=torch.int64)
    bits = ((packed_symbols.unsqueeze(-1) >> shifts.view(1, 1, 1, -1)) & 1).float()
    signed = bits.mul(2).sub(1)
    return signed * (match_length > 0).unsqueeze(-1).float()


def _truncated_reference(
    q_bits: torch.Tensor,
    k_bits: torch.Tensor,
    v_bits: torch.Tensor,
    *,
    suffix_window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, steps = q_bits.shape
    packed_y = torch.zeros_like(v_bits)
    match_length = torch.zeros((batch, heads, steps), dtype=torch.int64)
    endpos = torch.full((batch, heads, steps), -1, dtype=torch.int64)

    for batch_idx in range(batch):
        for head_idx in range(heads):
            q_head = q_bits[batch_idx, head_idx].tolist()
            k_head = k_bits[batch_idx, head_idx].tolist()
            v_head = v_bits[batch_idx, head_idx].tolist()
            for step_idx in range(steps):
                best_length = 0
                best_endpos = -1
                max_width = min(suffix_window, step_idx + 1)
                for width in range(max_width, 0, -1):
                    target = q_head[step_idx + 1 - width : step_idx + 1]
                    for start in range(step_idx - width, -1, -1):
                        if k_head[start : start + width] == target:
                            best_length = width
                            best_endpos = start + width - 1
                            break
                    if best_length > 0:
                        break
                if best_length > 0:
                    packed_y[batch_idx, head_idx, step_idx] = v_head[best_endpos + 1]
                    match_length[batch_idx, head_idx, step_idx] = best_length
                    endpos[batch_idx, head_idx, step_idx] = best_endpos

    return packed_y, match_length, endpos


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for SUFA GPU kernel tests.")
class RosaSufaKernelTests(unittest.TestCase):
    def test_proxy_triton_matches_torch_reference(self) -> None:
        if not is_triton_window_available():
            self.skipTest("Triton is not available in this environment.")

        torch.manual_seed(0)
        q_ref = torch.randn(2, 4, 16, 4, device="cuda", requires_grad=True)
        k_ref = torch.randn(2, 4, 16, 4, device="cuda", requires_grad=True)
        v_ref = torch.randn(2, 4, 16, 4, device="cuda", requires_grad=True)
        endpos = torch.randint(-1, 14, (2, 4, 16), device="cuda", dtype=torch.int64)
        grad_out = torch.randn(2, 4, 16, 4, device="cuda")

        out_ref = suffix_attention_proxy(
            q_ref,
            k_ref,
            v_ref,
            endpos=endpos,
            scale=None,
            suffix_window=4,
            suffix_factor=0.5,
            quant_mode="soft",
            quant_scale=None,
            kernel="torch",
        )
        grad_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), grad_outputs=grad_out)

        q_tri = q_ref.detach().clone().requires_grad_(True)
        k_tri = k_ref.detach().clone().requires_grad_(True)
        v_tri = v_ref.detach().clone().requires_grad_(True)
        out_tri = suffix_attention_proxy(
            q_tri,
            k_tri,
            v_tri,
            endpos=endpos,
            scale=None,
            suffix_window=4,
            suffix_factor=0.5,
            quant_mode="soft",
            quant_scale=None,
            kernel="proxy_triton",
        )
        grad_tri = torch.autograd.grad(out_tri, (q_tri, k_tri, v_tri), grad_outputs=grad_out)

        self.assertTrue(torch.allclose(out_ref, out_tri, atol=1e-5, rtol=1e-4))
        for ref_grad, tri_grad in zip(grad_ref, grad_tri):
            self.assertTrue(torch.allclose(ref_grad, tri_grad, atol=2e-5, rtol=2e-4))

    def test_auto_kernel_warns_once_and_falls_back_for_schmitt_trigger(self) -> None:
        torch.manual_seed(1)
        q = torch.randn(1, 2, 8, 4, device="cuda", requires_grad=True)
        k = torch.randn(1, 2, 8, 4, device="cuda", requires_grad=True)
        v = torch.randn(1, 2, 8, 4, device="cuda", requires_grad=True)

        with mock.patch.object(rosa_sufa_module, "_AUTO_SCHMITT_WARNING_EMITTED", False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out_auto_1 = rosa_sufa_ops(
                    q,
                    k,
                    v,
                    suffix_window=4,
                    suffix_factor=0.5,
                    quant_mode="soft",
                    quant_scale=None,
                    schmitt_trigger=0.1,
                    kernel="auto",
                )
                out_auto_2 = rosa_sufa_ops(
                    q,
                    k,
                    v,
                    suffix_window=4,
                    suffix_factor=0.5,
                    quant_mode="soft",
                    quant_scale=None,
                    schmitt_trigger=0.1,
                    kernel="auto",
                )

        out_torch = rosa_sufa_ops(
            q,
            k,
            v,
            suffix_window=4,
            suffix_factor=0.5,
            quant_mode="soft",
            quant_scale=None,
            schmitt_trigger=0.1,
            kernel="torch",
        )

        self.assertTrue(torch.allclose(out_auto_1, out_torch, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(out_auto_2, out_torch, atol=1e-5, rtol=1e-4))
        self.assertEqual(len(caught), 1)
        self.assertIn("falling back to kernel='torch'", str(caught[0].message))

    def test_truncated_cuda_matches_python_reference(self) -> None:
        q_bits = torch.tensor(
            [[[
                [1, 0, 1, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 1, 0],
                [1, 0, 0, 1],
            ]]],
            dtype=torch.float32,
            device="cuda",
        )
        k_bits = torch.tensor(
            [[[
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 1, 0],
                [1, 0, 0, 1],
            ]]],
            dtype=torch.float32,
            device="cuda",
        )
        v_bits = torch.tensor(
            [[[
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [0, 0, 1, 0],
            ]]],
            dtype=torch.float32,
            device="cuda",
        )

        x_hard, info = truncated_cuda_hard_forward(
            q_bits.mul(2).sub(1),
            k_bits.mul(2).sub(1),
            v_bits.mul(2).sub(1),
            suffix_window=4,
        )

        packed_q = _pack_bits_reference(q_bits.cpu())
        packed_k = _pack_bits_reference(k_bits.cpu())
        packed_v = _pack_bits_reference(v_bits.cpu())
        packed_y_ref, match_length_ref, endpos_ref = _truncated_reference(
            packed_q,
            packed_k,
            packed_v,
            suffix_window=4,
        )
        x_hard_ref = _decode_reference_symbols(packed_y_ref, match_length_ref, bits_per_symbol=4)

        self.assertTrue(torch.equal(info["length"].cpu(), match_length_ref))
        self.assertTrue(torch.equal(info["endpos"].cpu(), endpos_ref))
        self.assertTrue(torch.equal(x_hard.cpu(), x_hard_ref))

    def test_rosa_expert_sufa_with_truncated_cuda_env_smoke(self) -> None:
        expert = ROSAExpert(
            dim=8,
            slim_heads=2,
            bits_per_symbol=4,
            backend="sufa",
            truncation_length=4,
            sequence_length=8,
        ).cuda()
        x = torch.randn(2, 8, 8, device="cuda", requires_grad=True)

        with mock.patch.dict(os.environ, {"CAMOE_ROSA_SUFA_KERNEL": "truncated_cuda"}):
            out = expert(x)
            self.assertEqual(tuple(out.shape), (2, 8, 8))
            out.sum().backward()

        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_truncated_cuda_supports_multiple_specs_in_one_process(self) -> None:
        q1 = torch.randn(1, 2, 8, 4, device="cuda")
        k1 = torch.randn(1, 2, 8, 4, device="cuda")
        v1 = torch.randn(1, 2, 8, 4, device="cuda")
        out1, info1 = truncated_cuda_hard_forward(q1, k1, v1, suffix_window=4)

        q2 = torch.randn(1, 1, 6, 2, device="cuda")
        k2 = torch.randn(1, 1, 6, 2, device="cuda")
        v2 = torch.randn(1, 1, 6, 2, device="cuda")
        out2, info2 = truncated_cuda_hard_forward(q2, k2, v2, suffix_window=3)

        self.assertEqual(tuple(out1.shape), (1, 2, 8, 4))
        self.assertEqual(tuple(info1["length"].shape), (1, 2, 8))
        self.assertEqual(tuple(out2.shape), (1, 1, 6, 2))
        self.assertEqual(tuple(info2["length"].shape), (1, 1, 6))
