from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from .soft_rosa_cuda import diagonal_affine_scan_cuda
except ImportError:
    from soft_rosa_cuda import diagonal_affine_scan_cuda

try:
    from .soft_rosa_triton import diagonal_affine_scan_triton, triton_is_available
except ImportError:
    from soft_rosa_triton import diagonal_affine_scan_triton, triton_is_available

Tensor = torch.Tensor
_HAS_TRITON = triton_is_available()


def affine_scan_serial(a: Tensor) -> Tensor:
    """Reference scan for c[t] = a[t] * c[t-1] + a[t]."""

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
    if backend not in {"auto", "serial", "parallel", "cuda", "triton"}:
        raise ValueError(f"Unsupported backend={backend!r}")

    use_cuda = False
    use_triton = False
    if backend in {"parallel", "cuda"}:
        if not x.is_cuda:
            raise ValueError("CUDA diagonal scan requires CUDA tensors")
        use_cuda = True
    elif backend == "triton":
        if not x.is_cuda or not _HAS_TRITON:
            raise ValueError("Triton diagonal scan requires CUDA + Triton")
        use_triton = True
    elif backend == "auto":
        use_cuda = x.is_cuda
        use_triton = x.is_cuda and _HAS_TRITON

    if use_cuda:
        try:
            return diagonal_affine_scan_cuda(x)
        except Exception:
            if backend == "cuda":
                raise
            if _HAS_TRITON and x.is_cuda:
                return diagonal_affine_scan_triton(x)
    if use_triton:
        return diagonal_affine_scan_triton(x)

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


def _build_lookback_mask(seq_len: int, device: torch.device, max_lookback: Optional[int]) -> Tensor:
    i_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    j_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len)
    valid = j_idx < i_idx
    if max_lookback is not None:
        valid = valid & ((i_idx - j_idx) <= max_lookback)
    return valid


def _causal_soft_selection(
    ell: Tensor,
    value: Tensor,
    *,
    gamma: float,
    max_lookback: Optional[int],
    return_weights: bool,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    batch, seq_len, _ = ell.shape
    device = ell.device
    value_dim = value.size(-1)

    if seq_len <= 1:
        empty_weights = torch.zeros(batch, seq_len, 0, device=device, dtype=ell.dtype) if return_weights else None
        best_j = torch.full((batch, seq_len), -1, device=device, dtype=torch.long)
        y = torch.zeros(batch, seq_len, value_dim, device=device, dtype=value.dtype)
        return y, best_j, empty_weights

    valid = _build_lookback_mask(seq_len, device, max_lookback)[:, :, : seq_len - 1]
    j_idx = torch.arange(seq_len - 1, device=device).view(1, 1, seq_len - 1)

    scores = ell[:, :, : seq_len - 1].to(torch.float32) * gamma
    scores = scores + 1e-4 * j_idx.to(scores.dtype)
    scores = scores.masked_fill(~valid, float("-inf"))

    best_j = scores.argmax(dim=-1)
    best_j = torch.where(valid.any(dim=-1), best_j, torch.full_like(best_j, -1))

    score_max = scores.max(dim=-1, keepdim=True).values
    score_max = torch.where(torch.isfinite(score_max), score_max, torch.zeros_like(score_max))
    numerators = torch.exp(scores - score_max) * valid

    denom = numerators.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    y = torch.einsum("bts,bsd->btd", numerators, value[:, 1:, :])
    y = y / denom.to(y.dtype)

    if not return_weights:
        return y, best_j, None

    return y, best_j, (numerators / denom).to(ell.dtype)


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
    """Pure PyTorch Soft ROSA reference implementation."""

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, value must all have shape [B, T, D]")
    if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
        raise ValueError("query, key, value must agree on [B, T]")

    sim = _pairwise_similarity(query, key, similarity=similarity)
    mu = torch.sigmoid(alpha * sim)
    ell = diagonal_affine_scan_with_backend(mu, backend=scan_backend)

    if max_lookback is not None:
        lookback_mask = _build_lookback_mask(ell.size(1), ell.device, max_lookback)
        ell = torch.where(lookback_mask, ell, torch.zeros_like(ell))

    y, best_j, weights = _causal_soft_selection(
        ell,
        value,
        gamma=gamma,
        max_lookback=max_lookback,
        return_weights=return_aux,
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
    """Reference Hard ROSA for causal next-value retrieval."""

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


try:
    from .ops import qkv1bit_rosa_ops, soft_rosa_ops, soft_rosa_parallel_ops, soft_rosa_serial_ops
except ImportError:
    try:
        from ops import qkv1bit_rosa_ops, soft_rosa_ops, soft_rosa_parallel_ops, soft_rosa_serial_ops
    except ImportError:
        pass
