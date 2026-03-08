"""Minimal configuration for CaMoE v22."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

VERSION = "v22"


@dataclass
class CaMoEConfig:
    # Model
    vocab_size: int = 65536
    dim: int = 1024
    n_layers: int = 12
    n_heads: int = 16

    # Experts
    n_experts: int = 6
    ffn_expand: int = 4
    top_k: int = 2
    tie_weights: bool = True

    # Auction
    auction_noise_std: float = 0.01

    # Capital
    expert_capital_init: float = 1.0
    critic_capital_init: float = 0.5
    ema_decay: float = 0.95
    capital_floor: float = 0.01
    depreciation: float = 0.001

    # Critic
    critic_hidden_dim: int | None = None
    critic_update_interval: int = 8
    critic_lr: float = 3e-4

    # Schedules
    prewarm_steps: int = 2000
    market_warmup_steps: int = 3000
    critic_warmup_steps: int = 5000

    # Training
    lr: float = 3e-4
    batch_size: int = 32
    seq_len: int = 512
    total_steps: int = 50000
    grad_clip: float = 1.0
    ignore_index: int = -100
    version: str = VERSION

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by n_heads={self.n_heads}.")
        if self.top_k >= self.n_experts:
            raise ValueError("top_k must be smaller than n_experts for Vickrey pricing.")

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    def copy(self) -> "CaMoEConfig":
        return replace(self)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: "CaMoEConfig | Mapping[str, Any]") -> "CaMoEConfig":
        if isinstance(value, cls):
            return value.copy()
        valid = {field.name for field in fields(cls)}
        payload = {key: item for key, item in dict(value).items() if key in valid}
        return cls(**payload)


CONFIG_0_1B = CaMoEConfig(
    dim=512,
    n_layers=8,
    n_heads=8,
    n_experts=4,
    top_k=2,
    total_steps=15000,
)

CONFIG_0_4B = CaMoEConfig(
    dim=1024,
    n_layers=12,
    n_heads=16,
    n_experts=6,
    top_k=2,
    total_steps=50000,
)

CONFIG_MINIPILE = CONFIG_0_4B


def get_config(scale: str = "0.4b") -> CaMoEConfig:
    if scale == "0.1b":
        return CONFIG_0_1B.copy()
    if scale == "0.4b":
        return CONFIG_0_4B.copy()
    raise ValueError(f"Unknown scale: {scale}")
