"""Minimal configuration for prediction-market CaMoE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

VERSION = "v23.0"


@dataclass
class CaMoEConfig:
    # Model
    vocab_size: int = 65536
    dim: int = 1024
    n_layers: int = 12
    n_heads: int = 16

    # Experts
    # Number of classic RWKV FFN experts in the FFN market.
    n_experts: int = 6
    n_deepembed_experts: int = 0
    n_slim_deepembed_experts: int = 0
    ffn_expand: int = 4
    tie_weights: bool = True
    # Slim Wind ROSA sequence branch
    n_rosa_experts: int = 1
    rosa_backend: str = "wind"
    # Bits per symbolic channel. `4` yields an explicit 4-bit ROSA.
    rosa_bits: int = 8
    # Symbolic channel count after the slim projection. This can be much smaller than dim.
    slim_rosa_heads: int | None = None
    # Wind ROSA truncation length K.
    rosa_truncation_length: int = 8

    # Expertized DeepEmbed
    deepembed_mode: str = "1x"
    deepembed_expand: int = 4
    slim_deepembed_rank: int = 32

    # Fractal placeholder
    enable_fractal_placeholders: bool = False
    fractal_max_depth: int = 0
    fractal_expansion_cost: float = 1.0

    # Auction
    auction_noise_std: float = 0.01
    market_alpha_start: float = 0.0
    market_alpha_end: float = 1.0
    routing_noise_std: float = 0.01
    exploration_epsilon: float = 0.02
    bet_fraction: float = 0.05
    price_lr: float = 0.02
    price_temperature: float = 1.0
    liquidity_floor: float = 0.02
    routing_ste: bool = True
    ste_temperature_start: float = 2.0
    ste_temperature_mid: float = 1.0
    ste_temperature_end: float = 0.3
    ste_midpoint_steps: int = 1500
    ste_anneal_steps: int = 4000
    market_ramp_steps: int = 1000

    # Runtime
    enable_compile: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"
    enable_gradient_checkpointing: bool = True

    # Capital
    expert_capital_init: float = 1.0
    critic_capital_init: float = 0.5
    ema_decay: float = 0.95
    capital_floor: float = 0.01
    capital_ceiling: float = 10.0
    depreciation: float = 0.001

    # Critic
    critic_hidden_dim: int | None = None
    reward_hidden_dim: int | None = None
    critic_update_interval: int = 8
    critic_lr: float = 3e-4
    critic_profit_clip: float = 1.0
    reward_scale: float = 5.0
    reward_eps: float = 1e-8
    routing_entropy_reg: float = 0.0
    critic_shadow_prewarm: bool = True
    critic_shadow_market: bool = True

    # Schedules
    uniform_warmup_steps: int = 2000
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
        if self.n_experts < 0 or self.n_deepembed_experts < 0 or self.n_slim_deepembed_experts < 0:
            raise ValueError("Expert counts must be non-negative.")
        if self.total_ffn_experts < 2:
            raise ValueError("prediction-market routing requires at least 2 FFN-market experts.")
        if self.n_rosa_experts not in (0, 1):
            raise ValueError("The current implementation supports either 0 or 1 sequence ROSA expert per layer.")
        if self.rosa_backend not in {
            "wind",
            "soft",
            "sufa",
            "scan",
            "soft_exact",
            "soft_exact_serial",
            "soft_exact_cuda",
            "soft_exact_triton",
            "soft_qkv1bit",
            "soft_qkv1bit_triton",
            "soft_qkv1bit_cuda",
        }:
            raise ValueError(
                "rosa_backend must be one of {'wind', 'soft', 'sufa', 'scan', "
                "'soft_exact', 'soft_exact_serial', 'soft_exact_cuda', "
                "'soft_exact_triton', 'soft_qkv1bit', 'soft_qkv1bit_triton', "
                "'soft_qkv1bit_cuda'}."
            )
        if self.slim_rosa_heads is not None and self.slim_rosa_heads <= 0:
            raise ValueError("slim_rosa_heads must be positive when provided.")
        if self.rosa_bits <= 0:
            raise ValueError("rosa_bits must be positive.")
        if self.rosa_truncation_length <= 0:
            raise ValueError("rosa_truncation_length must be positive.")
        if self.deepembed_mode not in {"1x", "4x"}:
            raise ValueError("deepembed_mode must be '1x' or '4x'.")
        if self.deepembed_expand <= 0:
            raise ValueError("deepembed_expand must be positive.")
        if self.slim_deepembed_rank <= 0:
            raise ValueError("slim_deepembed_rank must be positive.")
        if self.fractal_max_depth < 0:
            raise ValueError("fractal_max_depth must be non-negative.")
        if self.fractal_expansion_cost < 0:
            raise ValueError("fractal_expansion_cost must be non-negative.")
        if self.capital_ceiling <= self.capital_floor:
            raise ValueError("capital_ceiling must be greater than capital_floor.")
        if self.routing_noise_std < 0:
            raise ValueError("routing_noise_std must be non-negative.")
        if not 0 <= self.exploration_epsilon <= 1:
            raise ValueError("exploration_epsilon must be in [0, 1].")
        if self.bet_fraction < 0:
            raise ValueError("bet_fraction must be non-negative.")
        if self.price_lr < 0:
            raise ValueError("price_lr must be non-negative.")
        if self.price_temperature <= 0:
            raise ValueError("price_temperature must be positive.")
        if not 0 <= self.liquidity_floor < 1:
            raise ValueError("liquidity_floor must be in [0, 1).")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive.")
        if self.reward_eps <= 0:
            raise ValueError("reward_eps must be positive.")
        if self.reward_hidden_dim is not None and self.reward_hidden_dim <= 0:
            raise ValueError("reward_hidden_dim must be positive when provided.")
        if self.uniform_warmup_steps < 0:
            raise ValueError("uniform_warmup_steps must be non-negative.")
        if self.critic_profit_clip <= 0:
            raise ValueError("critic_profit_clip must be positive.")
        if self.market_alpha_start < 0 or self.market_alpha_end < 0:
            raise ValueError("market_alpha_start/end must be non-negative.")
        if self.market_alpha_end < self.market_alpha_start:
            raise ValueError("market_alpha_end must be >= market_alpha_start.")
        if self.routing_entropy_reg < 0:
            raise ValueError("routing_entropy_reg must be non-negative.")
        if self.ste_temperature_start <= 0 or self.ste_temperature_mid <= 0 or self.ste_temperature_end <= 0:
            raise ValueError("STE temperatures must be positive.")
        if self.ste_midpoint_steps < 0 or self.ste_anneal_steps < 0:
            raise ValueError("ste_midpoint_steps / ste_anneal_steps must be non-negative.")
        if self.ste_anneal_steps > 0 and self.ste_midpoint_steps > self.ste_anneal_steps:
            raise ValueError("ste_midpoint_steps must be <= ste_anneal_steps.")
        if self.market_ramp_steps < 0:
            raise ValueError("market_ramp_steps must be non-negative.")

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def effective_slim_rosa_heads(self) -> int:
        return self.slim_rosa_heads or self.n_heads

    @property
    def total_ffn_experts(self) -> int:
        return self.n_experts + self.n_deepembed_experts + self.n_slim_deepembed_experts

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
    total_steps=15000,
)

CONFIG_0_4B = CaMoEConfig(
    dim=1024,
    n_layers=12,
    n_heads=16,
    n_experts=6,
    total_steps=50000,
)


def get_config(scale: str = "0.4b") -> CaMoEConfig:
    if scale == "0.1b":
        return CONFIG_0_1B.copy()
    if scale == "0.4b":
        return CONFIG_0_4B.copy()
    raise ValueError(f"Unknown scale: {scale}")
