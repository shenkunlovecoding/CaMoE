from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from torch import Tensor
from typing import *

from .rosa_sam import RosaContext
from .rosa_soft import RosaSoftWork
from .sufa_triton import (
    apply_suffix_window_triton,
    is_triton_window_available,
    triton_window_import_error,
)
from .sufa_truncated_cuda import truncated_cuda_hard_forward, validate_truncated_cuda_request
from .utils import decay_qk, gather_x, repeat_kv, unfold_qk


__all__ = [
    "rosa_sufa_ops",
    "suffix_attention_proxy",
]


_AUTO_PROXY_TRITON_DISABLED = False
_AUTO_PROXY_TRITON_WARNING_EMITTED = False
_AUTO_SCHMITT_WARNING_EMITTED = False


def _warn_auto_proxy_triton_once(message: str) -> None:
    global _AUTO_PROXY_TRITON_WARNING_EMITTED
    if _AUTO_PROXY_TRITON_WARNING_EMITTED:
        return
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    _AUTO_PROXY_TRITON_WARNING_EMITTED = True


def _warn_auto_schmitt_once(message: str) -> None:
    global _AUTO_SCHMITT_WARNING_EMITTED
    if _AUTO_SCHMITT_WARNING_EMITTED:
        return
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    _AUTO_SCHMITT_WARNING_EMITTED = True


def _proxy_triton_ready(query: Tensor, schmitt_trigger: float) -> tuple[bool, str | None]:
    if query.device.type != "cuda":
        return False, None
    if schmitt_trigger != 0.0:
        return False, "proxy_triton currently requires schmitt_trigger == 0.0."
    if _AUTO_PROXY_TRITON_DISABLED:
        return False, "proxy_triton was disabled after a previous runtime failure."
    if not is_triton_window_available():
        exc = triton_window_import_error()
        if exc is None:
            return False, "proxy_triton is unavailable in this environment."
        return False, f"proxy_triton is unavailable: {exc}"
    return True, None


def _resolve_sufa_kernel(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    kernel: str,
    suffix_window: int,
    schmitt_trigger: float,
) -> str:
    normalized = str(kernel).lower()
    if normalized not in {"auto", "torch", "proxy_triton", "truncated_cuda"}:
        raise ValueError(
            "Unsupported SUFA kernel. Expected one of "
            "{'auto', 'torch', 'proxy_triton', 'truncated_cuda'}, "
            f"got {kernel!r}."
        )

    if normalized == "auto":
        ready, reason = _proxy_triton_ready(query, schmitt_trigger)
        if ready:
            return "proxy_triton"
        if query.device.type == "cuda":
            if schmitt_trigger != 0.0:
                _warn_auto_schmitt_once(
                    "rosa_sufa_ops(kernel='auto') is falling back to kernel='torch' because "
                    "proxy_triton requires schmitt_trigger == 0.0."
                )
            elif reason is not None:
                _warn_auto_proxy_triton_once(
                    "rosa_sufa_ops(kernel='auto') is falling back to kernel='torch': "
                    f"{reason}"
                )
        return "torch"

    if normalized == "proxy_triton":
        ready, reason = _proxy_triton_ready(query, schmitt_trigger)
        if not ready:
            raise RuntimeError(reason or "proxy_triton requires CUDA tensors.")
        return normalized

    if normalized == "truncated_cuda":
        validate_truncated_cuda_request(
            query,
            key,
            value,
            suffix_window=suffix_window,
            schmitt_trigger=schmitt_trigger,
        )
        return normalized

    return normalized


class _ImmediateSufaWork:
    def __init__(self, query: Tensor, key: Tensor, value: Tensor, params: "RosaSufaParams") -> None:
        self._query = query
        self._key = key
        self._value = value
        self._params = params

    def wait(self) -> Tensor:
        if self._params is None:
            raise RuntimeError("wait() called twice")
        query = self._query
        key = self._key
        value = self._value
        params = self._params
        self._query = None
        self._key = None
        self._value = None
        self._params = None
        return RosaSufaFunction.apply(query, key, value, params)


def rosa_sufa_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    suffix_window: int = 8,
    suffix_factor: Optional[float] = 0.5,
    quant_mode: str = "soft",
    quant_scale: Optional[float] = None,
    schmitt_trigger: float = 0.0,
    kernel: str = "auto",
    async_op: bool = False,
) -> Union[Tensor, RosaSoftWork]:
    """ROSA Suffix Attention operator (SUFA, Production Default)."""

    resolved_kernel = _resolve_sufa_kernel(
        query,
        key,
        value,
        kernel=kernel,
        suffix_window=suffix_window,
        schmitt_trigger=schmitt_trigger,
    )
    proxy_kernel = "proxy_triton" if resolved_kernel in {"proxy_triton", "truncated_cuda"} else "torch"
    requested_kernel = str(kernel).lower()
    proxy_allow_fallback = requested_kernel == "auto" or resolved_kernel == "truncated_cuda"

    params = RosaSufaParams(
        scale=scale,
        suffix_window=suffix_window,
        suffix_factor=suffix_factor,
        quant_mode=quant_mode,
        quant_scale=quant_scale,
        schmitt_trigger=schmitt_trigger,
        requested_kernel=requested_kernel,
        resolved_kernel=resolved_kernel,
        proxy_kernel=proxy_kernel,
        proxy_allow_fallback=proxy_allow_fallback,
    )

    if resolved_kernel == "truncated_cuda":
        x_hard, info = truncated_cuda_hard_forward(
            query,
            key,
            value,
            suffix_window=suffix_window,
            schmitt_trigger=schmitt_trigger,
        )
        params.info["x_hard"] = x_hard
        params.info.update(info)
        if async_op:
            return _ImmediateSufaWork(query, key, value, params)
        return RosaSufaFunction.apply(query, key, value, params)

    work = RosaSoftWork()
    work._future = RosaContext().update(
        query=query,
        key=key,
        value=value,
        schmitt_trigger=schmitt_trigger,
        async_op=True,
    )
    work._params = params
    work._function_apply = RosaSufaFunction.apply
    work._query_key_value = (query, key, value)

    if async_op:
        return work
    return work.wait()


class RosaSufaParams:
    def __init__(
        self,
        scale: Optional[float],
        suffix_window: int,
        suffix_factor: Optional[float],
        quant_mode: str,
        quant_scale: Optional[float],
        schmitt_trigger: float,
        requested_kernel: str,
        resolved_kernel: str,
        proxy_kernel: str,
        proxy_allow_fallback: bool,
    ):
        self.scale = scale
        self.suffix_window = suffix_window
        self.suffix_factor = suffix_factor
        self.quant_mode = quant_mode
        self.quant_scale = quant_scale
        self.schmitt_trigger = schmitt_trigger
        self.requested_kernel = requested_kernel
        self.resolved_kernel = resolved_kernel
        self.proxy_kernel = proxy_kernel
        self.proxy_allow_fallback = proxy_allow_fallback

        self.info: Dict[str, Tensor] = {}
        self._ctx = None


class RosaSufaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, params: RosaSufaParams):
        if "x_hard" in params.info:
            x_hard = params.info.pop("x_hard")
        else:
            x_hard, info = RosaContext().update(
                query=query,
                key=key,
                value=value,
                schmitt_trigger=params.schmitt_trigger,
            )
            params.info.update(info)

        ctx.save_for_backward(query.detach(), key.detach(), value.detach())
        ctx.saved_params = params

        return x_hard

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        query, key, value = cast(Tuple[Tensor, ...], ctx.saved_tensors)
        params: RosaSufaParams = ctx.saved_params

        params.info.pop("length")
        endpos = params.info.pop("endpos")

        with torch.enable_grad():
            query.requires_grad_(True)
            key.requires_grad_(True)
            value.requires_grad_(True)

            x_soft = suffix_attention_proxy(
                query,
                key,
                value,
                endpos=endpos,
                scale=params.scale,
                suffix_window=params.suffix_window,
                suffix_factor=params.suffix_factor,
                quant_mode=params.quant_mode,
                quant_scale=params.quant_scale,
                kernel=params.proxy_kernel,
                allow_fallback=params.proxy_allow_fallback,
            )

            grad_query, grad_key, grad_value = torch.autograd.grad(
                outputs=x_soft,
                inputs=(query, key, value),
                grad_outputs=grad_output,
                retain_graph=False,
                only_inputs=True,
            )
        return grad_query, grad_key, grad_value, None


def _suffix_attention_proxy_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    endpos: Tensor,
    scale: Optional[float],
    suffix_window: int,
    suffix_factor: Optional[float],
    quant_mode: str,
    quant_scale: Optional[float],
) -> Tensor:
    bsz, num_heads, seq_len, num_q_bits = query.size()
    bsz, num_kv_heads, seq_len, num_k_bits = key.size()
    bsz, num_kv_heads, seq_len, _num_v_bits = value.size()

    if num_q_bits != num_k_bits:
        raise ValueError("query and key must have the same number of bits")
    if suffix_window <= 0:
        raise ValueError(f"suffix_window must be positive, got {suffix_window}.")

    if quant_scale is not None:
        if quant_scale < 1.0:
            raise ValueError(f"quant_scale must be >= 1.0, got {quant_scale}.")
        query = query * quant_scale
        key = key * quant_scale
        value = value * quant_scale
    else:
        quant_scale = 1.0

    if quant_mode == "tanh":
        xq = torch.tanh(query)
        xk = torch.tanh(key)
        xv = torch.tanh(value)
    elif quant_mode == "soft":
        xq = F.softsign(query)
        xk = F.softsign(key)
        xv = F.softsign(value)
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}, expected one of 'tanh' or 'soft'")

    n_rep = num_heads // num_kv_heads
    xk = repeat_kv(xk, n_rep)
    xv = repeat_kv(xv, n_rep)

    xq = unfold_qk(xq, win_size=suffix_window, offset=0)
    xk = unfold_qk(xk, win_size=suffix_window, offset=1)
    xq, xk = decay_qk(xq, xk, decay_factor=suffix_factor)

    xq = xq.reshape(bsz, num_heads, seq_len, num_q_bits * suffix_window)
    xk = xk.reshape(bsz, num_heads, seq_len, num_k_bits * suffix_window)

    attn_scale = (1.0 / num_q_bits) if scale is None else float(scale)
    xo = F.scaled_dot_product_attention(xq, xk, xv, scale=attn_scale, is_causal=True, attn_mask=None)

    pk = gather_x(xk, endpos)
    pv = gather_x(xv, endpos)

    gg = torch.sum(xq * pk, dim=-1, keepdim=True)
    gg = torch.sigmoid(gg * attn_scale)

    return xo * (1 - gg) + pv * gg


def _suffix_attention_proxy_proxy_triton(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    endpos: Tensor,
    scale: Optional[float],
    suffix_window: int,
    suffix_factor: Optional[float],
    quant_mode: str,
    quant_scale: Optional[float],
) -> Tensor:
    bsz, num_heads, seq_len, num_q_bits = query.size()
    bsz, num_kv_heads, seq_len, num_k_bits = key.size()
    bsz, num_kv_heads, seq_len, _num_v_bits = value.size()

    if num_q_bits != num_k_bits:
        raise ValueError("query and key must have the same number of bits")
    if suffix_window <= 0:
        raise ValueError(f"suffix_window must be positive, got {suffix_window}.")

    if quant_scale is not None:
        if quant_scale < 1.0:
            raise ValueError(f"quant_scale must be >= 1.0, got {quant_scale}.")
        query = query * quant_scale
        key = key * quant_scale
        value = value * quant_scale

    if quant_mode == "tanh":
        xq = torch.tanh(query)
        xk = torch.tanh(key)
        xv = torch.tanh(value)
    elif quant_mode == "soft":
        xq = F.softsign(query)
        xk = F.softsign(key)
        xv = F.softsign(value)
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}, expected one of 'tanh' or 'soft'")

    n_rep = num_heads // num_kv_heads
    xk = repeat_kv(xk, n_rep)
    xv = repeat_kv(xv, n_rep)

    xq = apply_suffix_window_triton(
        xq,
        win_size=suffix_window,
        offset=0,
        decay_factor=suffix_factor,
    )
    xk = apply_suffix_window_triton(
        xk,
        win_size=suffix_window,
        offset=1,
        decay_factor=suffix_factor,
    )

    xq = xq.reshape(bsz, num_heads, seq_len, num_q_bits * suffix_window)
    xk = xk.reshape(bsz, num_heads, seq_len, num_k_bits * suffix_window)

    attn_scale = (1.0 / num_q_bits) if scale is None else float(scale)
    xo = F.scaled_dot_product_attention(xq, xk, xv, scale=attn_scale, is_causal=True, attn_mask=None)

    pk = gather_x(xk, endpos)
    pv = gather_x(xv, endpos)

    gg = torch.sum(xq * pk, dim=-1, keepdim=True)
    gg = torch.sigmoid(gg * attn_scale)

    return xo * (1 - gg) + pv * gg


def suffix_attention_proxy(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    endpos: Tensor,
    scale: Optional[float],
    suffix_window: int,
    suffix_factor: Optional[float],
    quant_mode: str,
    quant_scale: Optional[float],
    kernel: str = "torch",
    allow_fallback: bool = False,
) -> Tensor:
    normalized = str(kernel).lower()
    if normalized == "proxy_triton":
        try:
            return _suffix_attention_proxy_proxy_triton(
                query,
                key,
                value,
                endpos=endpos,
                scale=scale,
                suffix_window=suffix_window,
                suffix_factor=suffix_factor,
                quant_mode=quant_mode,
                quant_scale=quant_scale,
            )
        except Exception as exc:
            if not allow_fallback:
                raise

            global _AUTO_PROXY_TRITON_DISABLED
            _AUTO_PROXY_TRITON_DISABLED = True
            _warn_auto_proxy_triton_once(
                "proxy_triton failed at runtime and has been disabled for the rest of this process. "
                f"Falling back to kernel='torch'. Original error: {exc}"
            )

    return _suffix_attention_proxy_reference(
        query,
        key,
        value,
        endpos=endpos,
        scale=scale,
        suffix_window=suffix_window,
        suffix_factor=suffix_factor,
        quant_mode=quant_mode,
        quant_scale=quant_scale,
    )
