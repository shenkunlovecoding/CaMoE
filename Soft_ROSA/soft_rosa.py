from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False

Tensor = torch.Tensor


def affine_scan_serial(a: Tensor) -> Tensor:
    """Reference scan for c[t] = a[t] * c[t-1] + a[t].

    This is the serial form of the affine monoid scan described in the
    Soft ROSA note. It is intentionally simple and suitable for correctness
    experiments before replacing it with a Triton/CUDA scan kernel.
    """

    if a.ndim != 2:
        raise ValueError(f"Expected [B, L], got shape={tuple(a.shape)}")

    if a.size(1) == 0:
        return torch.zeros_like(a)

    states = [a[:, 0]]
    prev = states[0]
    for t in range(1, a.size(1)):
        prev = a[:, t] * prev + a[:, t]
        states.append(prev)
    return torch.stack(states, dim=1)


if _HAS_TRITON:
    @triton.jit
    def _diag_combine(g0, a0, g1, a1):
        g_ = g1 * g0
        a_ = g1 * a0 + a1
        return g_, a_


    @triton.jit
    def _diag_scan_kernel(
        x_ptr,
        y_ptr,
        T,
        stride_x_b,
        stride_x_r,
        stride_x_c,
        stride_y_b,
        stride_y_r,
        stride_y_c,
        BLOCK_SIZE_T: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        x_ptr_b = x_ptr + pid_b * stride_x_b
        y_ptr_b = y_ptr + pid_b * stride_y_b

        carry_g = 1.0
        carry_a = 0.0

        for block_start in range(0, T - pid_d, BLOCK_SIZE_T):
            off_c = block_start + tl.arange(0, BLOCK_SIZE_T)
            off_r = off_c + pid_d
            mask = off_r < T

            g = tl.load(
                x_ptr_b + off_r * stride_x_r + off_c * stride_x_c,
                mask=mask,
                other=1.0,
            )
            a = tl.where(mask, g, 0.0)

            _g, _a = _diag_combine(carry_g, carry_a, g, a)
            _g = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _g, g)
            _a = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _a, a)

            _g, _a = tl.associative_scan((_g, _a), axis=0, combine_fn=_diag_combine)
            tl.store(y_ptr_b + off_r * stride_y_r + off_c * stride_y_c, _a, mask=mask)

            _g, _a = tl.reduce((g, a), axis=0, combine_fn=_diag_combine)
            carry_g, carry_a = _diag_combine(carry_g, carry_a, _g, _a)


    @triton.jit
    def _diag_scan_bwd_kernel(
        x_ptr,
        y_ptr,
        g_ptr,
        o_ptr,
        T,
        stride_x_b,
        stride_x_r,
        stride_x_c,
        stride_y_b,
        stride_y_r,
        stride_y_c,
        stride_g_b,
        stride_g_r,
        stride_g_c,
        stride_o_b,
        stride_o_r,
        stride_o_c,
        BLOCK_SIZE_T: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        x_ptr_b = x_ptr + pid_b * stride_x_b
        y_ptr_b = y_ptr + pid_b * stride_y_b
        g_ptr_b = g_ptr + pid_b * stride_g_b
        o_ptr_b = o_ptr + pid_b * stride_o_b

        carry_a = 1.0
        carry_b = 0.0

        M = tl.cdiv(T - pid_d, BLOCK_SIZE_T)
        for i in range(M):
            block_start = (M - 1 - i) * BLOCK_SIZE_T
            off_c = block_start + BLOCK_SIZE_T - 1 - tl.arange(0, BLOCK_SIZE_T)
            off_r = off_c + pid_d

            mask = off_r < T

            a = tl.load(
                x_ptr_b + (off_r + 1) * stride_x_r + (off_c + 1) * stride_x_c,
                mask=mask & (off_r < T - 1),
                other=1.0,
            )
            b = tl.load(
                g_ptr_b + off_r * stride_g_r + off_c * stride_g_c,
                mask=mask,
                other=0.0,
            )

            _a, _b = _diag_combine(carry_a, carry_b, a, b)
            _a = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _a, a)
            _b = tl.where(tl.arange(0, BLOCK_SIZE_T) == 0, _b, b)

            _a, _b = tl.associative_scan((_a, _b), axis=0, combine_fn=_diag_combine)

            y = tl.load(
                y_ptr_b + (off_r - 1) * stride_y_r + (off_c - 1) * stride_y_c,
                mask=mask & (off_r > pid_d),
                other=0.0,
            )
            tl.store(o_ptr_b + off_r * stride_o_r + off_c * stride_o_c, (y + 1) * _b, mask=mask)

            _a, _b = tl.reduce((a, b), axis=0, combine_fn=_diag_combine)
            carry_a, carry_b = _diag_combine(carry_a, carry_b, _a, _b)


    class _DiagonalScanFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: Tensor) -> Tensor:
            x32 = x.contiguous().to(torch.float32)
            out = torch.zeros_like(x32)
            seq_len = x32.size(-1)
            block_size_t = triton.next_power_of_2(min(seq_len, 1024))
            _diag_scan_kernel[(x32.size(0), seq_len)](
                x32,
                out,
                seq_len,
                x32.stride(0),
                x32.stride(1),
                x32.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                BLOCK_SIZE_T=block_size_t,
            )
            ctx.save_for_backward(x32, out)
            return out

        @staticmethod
        def backward(ctx, grad_output: Tensor):
            x32, out = ctx.saved_tensors
            grad_output = grad_output.contiguous().to(torch.float32)
            grad_input = torch.zeros_like(x32)
            seq_len = x32.size(-1)
            block_size_t = triton.next_power_of_2(min(seq_len, 1024))
            _diag_scan_bwd_kernel[(x32.size(0), seq_len)](
                x32,
                out,
                grad_output,
                grad_input,
                seq_len,
                x32.stride(0),
                x32.stride(1),
                x32.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                grad_input.stride(0),
                grad_input.stride(1),
                grad_input.stride(2),
                BLOCK_SIZE_T=block_size_t,
            )
            return grad_input


def diagonal_affine_scan(x: Tensor) -> Tensor:
    """Compute c[i,j] = x[i,j] * (1 + c[i-1,j-1]) on lower diagonals."""
    return diagonal_affine_scan_with_backend(x, backend="auto")


def diagonal_affine_scan_with_backend(x: Tensor, *, backend: str = "auto") -> Tensor:
    """Compute c[i,j] = x[i,j] * (1 + c[i-1,j-1]) on lower diagonals."""

    if x.ndim != 3:
        raise ValueError(f"Expected [B, T, T], got shape={tuple(x.shape)}")
    if x.size(-1) != x.size(-2):
        raise ValueError("Expected a square matrix in the last two dimensions")

    backend = backend.lower()
    if backend not in {"auto", "serial", "parallel"}:
        raise ValueError(f"Unsupported backend={backend!r}")

    use_parallel = False
    if backend == "parallel":
        if not x.is_cuda or not _HAS_TRITON:
            raise ValueError("Parallel diagonal scan requires CUDA + Triton")
        use_parallel = True
    elif backend == "auto":
        use_parallel = x.is_cuda and _HAS_TRITON

    if use_parallel:
        return _DiagonalScanFunction.apply(x).to(x.dtype)

    batch, seq_len, _ = x.shape
    out = torch.zeros_like(x)
    for d in range(seq_len):
        idx = torch.arange(seq_len - d, device=x.device)
        diag = x[:, idx + d, idx]
        out[:, idx + d, idx] = affine_scan_serial(diag)
    return out


def _pairwise_similarity(query: Tensor, key: Tensor, similarity: str) -> Tensor:
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must have shape [B, T, D]")
    if query.shape[:2] != key.shape[:2]:
        raise ValueError("query and key must agree on batch and time dimensions")

    if similarity == "dot":
        return torch.einsum("btd,bsd->bts", query, key)

    qn = F.normalize(query, dim=-1, eps=1e-8)
    kn = F.normalize(key, dim=-1, eps=1e-8)
    cosine = torch.einsum("btd,bsd->bts", qn, kn)

    if similarity == "cosine":
        return cosine
    if similarity == "cosine_margin":
        return 2.0 * cosine - 1.0

    raise ValueError(
        f"Unsupported similarity={similarity!r}. "
        "Expected one of {'dot', 'cosine', 'cosine_margin'}."
    )


def soft_rosa_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    alpha: float = 10.0,
    gamma: float = 5.0,
    similarity: str = "cosine_margin",
    max_lookback: Optional[int] = None,
    scan_backend: str = "auto",
    return_aux: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
    """Pure PyTorch Soft ROSA reference implementation.

    Args:
        query: [B, T, D]
        key: [B, T, D]
        value: [B, T, Dv]
        alpha: Match temperature for mu(i, j).
        gamma: Selection sharpness for the softmax over suffix scores.
        similarity: Similarity mode used inside mu(i, j).
        max_lookback: Optional diagonal truncation window.
        scan_backend: One of {"auto", "serial", "parallel"}.
        return_aux: Return diagnostic tensors when True.

    Returns:
        y or (y, aux). `y` has shape [B, T, Dv].
    """

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, value must all have shape [B, T, D]")
    if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
        raise ValueError("query, key, value must agree on [B, T]")

    batch, seq_len, _ = query.shape
    _, _, value_dim = value.shape
    device = query.device
    dtype = query.dtype

    sim = _pairwise_similarity(query, key, similarity=similarity)
    mu = torch.sigmoid(alpha * sim)

    ell = diagonal_affine_scan_with_backend(mu, backend=scan_backend)
    if max_lookback is not None:
        i_idx_full = torch.arange(seq_len, device=device).view(1, seq_len, 1)
        j_idx_full = torch.arange(seq_len, device=device).view(1, 1, seq_len)
        lookback_mask = (j_idx_full < i_idx_full) & ((i_idx_full - j_idx_full) <= max_lookback)
        ell = torch.where(lookback_mask, ell, torch.zeros_like(ell))

    if seq_len <= 1:
        y = torch.zeros(batch, seq_len, value_dim, device=device, dtype=value.dtype)
        aux = {
            "sim": sim,
            "mu": mu,
            "ell": ell,
            "weights": torch.zeros(batch, seq_len, 0, device=device, dtype=dtype),
            "best_j": torch.full((batch, seq_len), -1, device=device, dtype=torch.long),
        }
        return (y, aux) if return_aux else y

    scores = gamma * ell[:, :, : seq_len - 1]

    i_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    j_idx = torch.arange(seq_len - 1, device=device).view(1, 1, seq_len - 1)
    valid = j_idx < i_idx
    if max_lookback is not None:
        valid = valid & ((i_idx - j_idx) <= max_lookback)

    # Hard ROSA breaks equal-length ties by the largest historical endpoint j.
    # A tiny recency bias lets the soft argmax inherit the same preference.
    scores = scores + 1e-4 * j_idx.to(dtype)

    masked_scores = scores.masked_fill(~valid, -1e9)
    weights = torch.softmax(masked_scores, dim=-1)
    weights = weights * valid
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    shifted_value = value[:, 1:, :]
    y = torch.einsum("bts,bsd->btd", weights, shifted_value)
    best_j = weights.argmax(dim=-1)
    best_j = torch.where(
        valid.any(dim=-1),
        best_j,
        torch.full_like(best_j, -1),
    )

    if not return_aux:
        return y

    aux = {
        "sim": sim,
        "mu": mu,
        "ell": ell,
        "weights": weights,
        "best_j": best_j,
    }
    return y, aux


def hard_rosa_reference(
    query_symbols: Tensor,
    key_symbols: Tensor,
    value: Tensor,
    *,
    max_match: Optional[int] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Reference Hard ROSA for causal next-value retrieval.

    The semantics follow the note in the user prompt:
    for each query position i, choose a past key endpoint j < i that maximizes
    the suffix match length between q[:i] and k[:j], then return value[j + 1].

    Args:
        query_symbols: [B, T] integer tensor.
        key_symbols: [B, T] integer tensor.
        value: [B, T, Dv]
        max_match: Optional maximum suffix length K.
    """

    if query_symbols.ndim != 2 or key_symbols.ndim != 2:
        raise ValueError("query_symbols and key_symbols must have shape [B, T]")
    if value.ndim != 3 or value.shape[:2] != query_symbols.shape:
        raise ValueError("value must have shape [B, T, Dv] matching query_symbols")

    batch, seq_len = query_symbols.shape
    _, _, value_dim = value.shape

    output = torch.zeros(batch, seq_len, value_dim, device=value.device, dtype=value.dtype)
    best_j = torch.full((batch, seq_len), -1, device=query_symbols.device, dtype=torch.long)
    best_len = torch.zeros(batch, seq_len, device=query_symbols.device, dtype=torch.long)

    q_cpu = query_symbols.detach().cpu()
    k_cpu = key_symbols.detach().cpu()

    for b in range(batch):
        q_list = q_cpu[b].tolist()
        k_list = k_cpu[b].tolist()
        for i in range(seq_len):
            chosen_j = -1
            chosen_len = 0
            for j in range(min(i, seq_len - 1)):
                length = 0
                while i - length >= 0 and j - length >= 0:
                    if max_match is not None and length >= max_match:
                        break
                    if q_list[i - length] != k_list[j - length]:
                        break
                    length += 1

                if length > chosen_len or (length > 0 and length == chosen_len and j > chosen_j):
                    chosen_len = length
                    chosen_j = j

            if chosen_j >= 0 and chosen_j + 1 < seq_len:
                output[b, i] = value[b, chosen_j + 1]
                best_j[b, i] = chosen_j
                best_len[b, i] = chosen_len

    return output, {"best_j": best_j, "best_len": best_len}


def symbols_to_embeddings(symbols: Tensor, vocab_size: int) -> Tensor:
    """Convert integer symbols [B, T] into one-hot embeddings [B, T, V]."""

    if symbols.ndim != 2:
        raise ValueError("symbols must have shape [B, T]")
    return F.one_hot(symbols.to(torch.long), num_classes=vocab_size).to(torch.float32)
