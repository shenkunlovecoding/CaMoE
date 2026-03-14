from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.cpp_extension import load


ROOT_DIR = Path(__file__).resolve().parents[2]
CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_EXTENSION_TAG = "mspec_v3"


@dataclass(frozen=True)
class TruncatedCudaSpec:
    sequence_length: int
    bits_per_symbol: int
    suffix_window: int


_LOADED_NAMESPACES: dict[TruncatedCudaSpec, str] = {}
_SCRATCH_BY_SPEC_DEVICE: dict[tuple[TruncatedCudaSpec, torch.device], torch.Tensor] = {}


@dataclass(frozen=True)
class _SpecOpNames:
    forward: str
    backward: str
    scratch: str


def validate_truncated_cuda_request(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    suffix_window: int,
    schmitt_trigger: float,
) -> int:
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(f"truncated_cuda expects matching q/k/v shapes, got {query.shape}, {key.shape}, {value.shape}")
    if query.dim() != 4:
        raise ValueError(f"truncated_cuda expects [B, H, T, D] tensors, got {tuple(query.shape)}")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("truncated_cuda requires CUDA tensors.")
    if schmitt_trigger != 0.0:
        raise ValueError("truncated_cuda requires schmitt_trigger == 0.0.")
    if suffix_window <= 0:
        raise ValueError(f"suffix_window must be positive, got {suffix_window}.")

    bits_per_symbol = int(query.size(-1))
    if bits_per_symbol <= 0:
        raise ValueError("bits_per_symbol must be positive.")
    if bits_per_symbol > 8:
        raise ValueError(
            f"truncated_cuda currently supports bits_per_symbol <= 8, got {bits_per_symbol}."
        )
    if bits_per_symbol * int(suffix_window) > 64:
        raise ValueError(
            "truncated_cuda requires suffix_window * bits_per_symbol <= 64, "
            f"got {suffix_window} * {bits_per_symbol}."
        )
    return bits_per_symbol


def _pack_bits(x: torch.Tensor, bits_per_symbol: int) -> torch.Tensor:
    shifts = torch.arange(bits_per_symbol, device=x.device, dtype=torch.int64)
    bits = (x > 0).to(torch.int64)
    return (bits << shifts.view(1, 1, 1, -1)).sum(dim=-1).to(torch.uint8)


def _decode_signed_symbols(
    packed_symbols: torch.Tensor,
    match_length: torch.Tensor,
    bits_per_symbol: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    shifts = torch.arange(bits_per_symbol, device=packed_symbols.device, dtype=torch.int64)
    bits = ((packed_symbols.unsqueeze(-1) >> shifts.view(1, 1, 1, -1)) & 1).to(dtype)
    signed = bits.mul(2).sub(1)
    matched = (match_length > 0).unsqueeze(-1).to(dtype)
    return signed * matched


def _spec_namespace(spec: TruncatedCudaSpec) -> str:
    return f"rosa_sufa_truncated_t{spec.sequence_length}_c{spec.bits_per_symbol}_k{spec.suffix_window}"


def _spec_op_names(spec: TruncatedCudaSpec) -> _SpecOpNames:
    suffix = f"t{spec.sequence_length}_c{spec.bits_per_symbol}_k{spec.suffix_window}"
    return _SpecOpNames(
        forward=f"forward_with_metadata_{suffix}",
        backward=f"backward_{suffix}",
        scratch=f"scratch_size_{suffix}",
    )


def _ops_for_spec(spec: TruncatedCudaSpec) -> _SpecOpNames:
    if spec not in _LOADED_NAMESPACES:
        raise KeyError(f"truncated_cuda spec is not loaded: {spec}")
    return _spec_op_names(spec)


def _scratch_for_spec(spec: TruncatedCudaSpec, device: torch.device, scratch_bytes: int) -> torch.Tensor:
    cache_key = (spec, device)
    scratch = _SCRATCH_BY_SPEC_DEVICE.get(cache_key)
    if scratch is None or scratch.numel() < scratch_bytes:
        scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=device)
        _SCRATCH_BY_SPEC_DEVICE[cache_key] = scratch
    return scratch


def load_truncated_cuda(spec: TruncatedCudaSpec) -> str:
    namespace = _LOADED_NAMESPACES.get(spec)
    if namespace is not None:
        return namespace

    namespace = _spec_namespace(spec)
    extension_name = f"{namespace}_{_EXTENSION_TAG}"
    op_names = _spec_op_names(spec)
    forward_define = f"-DROSA_SUFA_TRUNCATED_FORWARD_OP={op_names.forward}"
    backward_define = f"-DROSA_SUFA_TRUNCATED_BACKWARD_OP={op_names.backward}"
    scratch_define = f"-DROSA_SUFA_TRUNCATED_SCRATCH_OP={op_names.scratch}"

    load(
        name=extension_name,
        sources=[
            str(CSRC_DIR / "wind_sufa_truncated.cu"),
            str(CSRC_DIR / "wind_sufa_truncated.cpp"),
        ],
        is_python_module=False,
        verbose=False,
        extra_cflags=[forward_define, backward_define, scratch_define],
        extra_cuda_cflags=[
            f"-D_T_={spec.sequence_length}",
            f"-D_C_={spec.bits_per_symbol}",
            f"-D_K_={spec.suffix_window}",
            forward_define,
            backward_define,
            scratch_define,
        ],
    )
    _LOADED_NAMESPACES[spec] = namespace
    return namespace


def truncated_cuda_symbolic_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    suffix_window: int,
    schmitt_trigger: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bits_per_symbol = validate_truncated_cuda_request(
        query,
        key,
        value,
        suffix_window=suffix_window,
        schmitt_trigger=schmitt_trigger,
    )
    spec = TruncatedCudaSpec(
        sequence_length=int(query.size(-2)),
        bits_per_symbol=bits_per_symbol,
        suffix_window=int(suffix_window),
    )
    load_truncated_cuda(spec)
    op_names = _ops_for_spec(spec)
    ops = torch.ops.rosa_sufa_truncated

    packed_q = _pack_bits(query.float(), bits_per_symbol).contiguous()
    packed_k = _pack_bits(key.float(), bits_per_symbol).contiguous()
    packed_v = _pack_bits(value.float(), bits_per_symbol).contiguous()

    batch, heads, steps = packed_q.shape
    scratch_bytes = int(getattr(ops, op_names.scratch)(batch, heads))
    scratch = _scratch_for_spec(spec, query.device, scratch_bytes)

    packed_y = torch.empty_like(packed_v)
    match_length = torch.empty((batch, heads, steps), dtype=torch.int32, device=query.device)
    endpos = torch.empty((batch, heads, steps), dtype=torch.int32, device=query.device)
    getattr(ops, op_names.forward)(
        packed_q,
        packed_k,
        packed_v,
        packed_y,
        match_length,
        endpos,
        scratch,
    )
    return packed_y, match_length, endpos


def truncated_cuda_hard_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    suffix_window: int,
    schmitt_trigger: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    packed_y, match_length, endpos = truncated_cuda_symbolic_forward(
        query,
        key,
        value,
        suffix_window=suffix_window,
        schmitt_trigger=schmitt_trigger,
    )
    x_hard = _decode_signed_symbols(
        packed_y,
        match_length,
        int(query.size(-1)),
        value.dtype,
    )
    info = {
        "length": match_length.to(torch.int64),
        "endpos": endpos.to(torch.int64),
    }
    return x_hard, info
