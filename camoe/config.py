"""
CaMoE v21 配置文件
7 阶段训练调度 + 经济系统增强版
"""

from .config_pilot import CONFIG_PILOT

# ==========================================
# 版本控制
# ==========================================
VERSION = "v21"
SCALE = "0.4b"
VARIANT = "6R2T-Top2"

# ==========================================
# 自动生成标识符
# ==========================================
RUN_ID = f"FineWeb70-Cosmo30-{SCALE}-{VARIANT}-{VERSION}"

# ==========================================
# 通用分组名（用于分阶段训练）
# ==========================================
PARAM_GROUPS = [
    "rwkv_backbone",
    "router_conf",
    "rwkv_experts",
    "trans_experts",
    "bridge",
    "critic",
    "emb_head",
]

# ==========================================
# v21 七阶段调度（SFT/RLHF 占位）
# ==========================================
PHASE_SCHEDULE_V21 = [
    {
        "name": "prewarm",
        "steps": 2000,
        "data_profile": "fineweb_cosmo_70_30",
        "train_groups": ["router_conf", "bridge"],
        "lr_mult": {
            "rwkv_backbone": 0.0,
            "router_conf": 1.0,
            "rwkv_experts": 0.0,
            "trans_experts": 0.0,
            "bridge": 1.0,
            "critic": 0.0,
            "emb_head": 0.0,
        },
        "market_update": False,
        "use_market": False,
        "route_grad": True,
    },
    {
        "name": "warm",
        "steps": 3000,
        "data_profile": "fineweb_cosmo_70_30",
        "train_groups": ["all"],
        "lr_mult": {
            "rwkv_backbone": 0.35,
            "router_conf": 0.35,
            "rwkv_experts": 0.35,
            "trans_experts": 0.35,
            "bridge": 0.35,
            "critic": 0.35,
            "emb_head": 0.35,
        },
        "market_update": False,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "criticwarm",
        "steps": 4000,
        "data_profile": "fineweb_cosmo_70_30",
        "train_groups": ["critic"],
        "lr_mult": {
            "rwkv_backbone": 0.0,
            "router_conf": 0.0,
            "rwkv_experts": 0.0,
            "trans_experts": 0.0,
            "bridge": 0.0,
            "critic": 2.0,
            "emb_head": 0.0,
        },
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "prenormal",
        "steps": 3000,
        "data_profile": "fineweb_cosmo_70_30",
        "train_groups": ["all"],
        "lr_mult": {
            "rwkv_backbone": 0.35,
            "router_conf": 0.7,
            "rwkv_experts": 0.6,
            "trans_experts": 0.8,
            "bridge": 0.8,
            "critic": 1.6,
            "emb_head": 0.5,
        },
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "normal",
        "steps": 40000,
        "data_profile": "fineweb_cosmo_70_30",
        "train_groups": ["all"],
        "lr_mult": {
            "rwkv_backbone": 0.7,
            "router_conf": 1.0,
            "rwkv_experts": 1.0,
            "trans_experts": 1.0,
            "bridge": 1.0,
            "critic": 1.0,
            "emb_head": 0.8,
        },
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "sft",
        "steps": 0,
        "data_profile": "ultrachat_100",
        "train_groups": ["all"],
        "lr_mult": {
            "rwkv_backbone": 0.5,
            "router_conf": 0.6,
            "rwkv_experts": 0.8,
            "trans_experts": 0.8,
            "bridge": 0.6,
            "critic": 0.5,
            "emb_head": 0.8,
        },
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "rlhf",
        "steps": 0,
        "data_profile": "rlhf_placeholder",
        "train_groups": ["all"],
        "lr_mult": {
            "rwkv_backbone": 0.3,
            "router_conf": 0.4,
            "rwkv_experts": 0.5,
            "trans_experts": 0.5,
            "bridge": 0.4,
            "critic": 0.6,
            "emb_head": 0.6,
        },
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
]


def _total_steps(schedule):
    return int(sum(max(0, int(p.get("steps", 0))) for p in schedule))


# ==========================================
# 0.4B 主配置（v21）
# ==========================================
CONFIG_04B = {
    # ===== 元信息 =====
    "version": VERSION,
    "scale": SCALE,
    "variant": VARIANT,
    "project": f"CaMoE-{VERSION}",
    "run_name": RUN_ID,
    # ===== 模型结构 =====
    "n_embd": 1024,
    "n_layer": 16,
    "head_size": 64,
    "vocab_size": 65536,
    "tied_embeddings": True,
    "use_deep_embed_attention": False,
    "use_shared_deep_embed": True,
    "dea_q_dim": 256,
    "dea_kv_dim": 32,
    "dea_score_scale": 1024.0,
    "dea_cap_scale": 64.0,
    # ===== 专家配置 =====
    "num_rwkv_experts": 6,
    "num_trans_experts": 2,
    "top_k": 2,
    "prefix_len": 48,
    "low_rank_dim": 64,
    # ===== 优化器与阶段 =====
    "base_lr": 1e-4,
    "phase_schedule": PHASE_SCHEDULE_V21,
    "sft_steps": 0,
    "rlhf_steps": 0,
    "total_steps": _total_steps(PHASE_SCHEDULE_V21),
    "param_groups": PARAM_GROUPS,
    # ===== Market 参数（兼容旧键） =====
    "total_capital": 10000.0,
    "min_capital_share": 0.05,
    "tax_threshold": 2.0,
    "tax_rate": 0.1,
    # ===== 经济系统（v20） =====
    "economy": {
        # 中央银行 / QE
        "qe_low_ratio": 0.85,
        "qe_high_ratio": 1.20,
        "qe_inject_ratio": 0.20,
        "qe_drain_ratio": 0.10,
        "qe_floor_alloc_ratio": 0.70,
        # 基础算力保障
        "base_compute_floor_ratio": 0.06,
        # 破产保护 + 债务重组
        "critic_bankrupt_threshold_ratio": 0.20,
        "bailout_base": 1000.0,
        "bailout_decay": 0.65,
        "bailout_min": 200.0,
        "repay_ratio": 0.25,
        "restructure_alpha": 0.12,
        "donor_topk": 2,
        # 闲置税 + 折旧
        "idle_threshold": 0.01,
        "idle_tax_rate": 0.02,
        "depreciation_rate": 0.001,
        # 分红
        "base_commission": 0.8,
        "dividend_scale": 0.6,
        "dividend_std_factor": 0.5,
        # 风投注资（VC Injection）
        "vc_affinity_threshold": 0.15,
        "vc_low_cap_ratio": 0.85,
        "vc_selected_threshold": 0.01,
        "vc_inject_ratio": 0.01,
        "vc_max_inject_ratio": 0.12,
        # Market alpha EMA（非梯度）
        "market_alpha_ema": 0.98,
        "market_alpha_step": 0.2,
        "market_alpha_min": 0.0,
        "market_alpha_max": 1.0,
        # CriticWarm 奖励增强（中等）
        "criticwarm_reward_scale": 2.0,
        "criticwarm_penalty_scale": 0.4,
        "critic_bonus_scale": 0.2,
        "critic_bonus_clip": (-0.1, 0.3),
        "critic_bonus_ema_momentum": 0.95,
    },
    # ===== 训练参数 =====
    "micro_batch_size": 6,
    "ctx_len": 1024,
    "grad_accum": 8,
    "use_gradient_checkpoint": True,
    "checkpoint_att_stage": True,
    "checkpoint_expert_stage": True,
    "route_no_grad": True,
    "router_noise_std": 0.02,
    "market_alpha_init": 0.05,
    "aux_loss_coeff": 0.01,
    "lazy_prefix_union": True,
    # ===== 日志与评估 =====
    "log_interval": 10,
    "eval_interval": 1000,
    "eval_iters": 50,
    # ===== 数据 =====
    "data_path": "./data/camoe_mix_v20_fineweb70_cosmo30",
    "data_roots": {
        "fineweb_edu": "./data/fineweb_edu_sample10bt_rwkv_processed",
        "ultrachat_200k": "./data/ultrachat_200k_rwkv_processed",
        "cosmopedia_100k": "./data/cosmopedia_100k_rwkv_processed",
    },
    "data_profiles": {
        "fineweb_cosmo_70_30": {
            "mix": {
                "fineweb_edu": 0.70,
                "cosmopedia_100k": 0.30,
            }
        },
        "ultrachat_100": {
            "mix": {
                "ultrachat_200k": 1.0,
            }
        },
        "rlhf_placeholder": {
            "mix": {},
        },
    },
    # 兼容旧代码路径（默认等于主 profile）
    "mix": {
        "fineweb_edu": 0.70,
        "cosmopedia_100k": 0.30,
    },
    # ===== 其他 =====
    "weights_path": "model/rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    "vocab_file": "tokenizer/rwkv_vocab_v20230424.txt",
    "save_dir": f"checkpoints/{VERSION}_{SCALE}",
    # ===== 旧字段兼容（用于旧逻辑 fallback） =====
    "prewarm_steps": 2000,
    "warmup_steps": 5000,
    "lr_prewarm": 1e-4,
    "lr_warmup": 3.5e-5,
    "lr_normal": 1e-4,
}


# ==========================================
# 0.4B Toy 配置（快速验证）
# ==========================================
CONFIG_04B_TOY_PHASES = [
    {
        "name": "prewarm",
        "steps": 100,
        "data_profile": "fineweb_cosmo_70_30",
        "train_groups": ["router_conf", "bridge"],
        "lr_mult": {
            "rwkv_backbone": 0.0,
            "router_conf": 1.0,
            "rwkv_experts": 0.0,
            "trans_experts": 0.0,
            "bridge": 1.0,
            "critic": 0.0,
            "emb_head": 0.0,
        },
        "market_update": False,
        "use_market": False,
        "route_grad": True,
    },
    {
        "name": "warm",
        "steps": 150,
        "data_profile": "fineweb_cosmo_70_30",
        "train_groups": ["all"],
        "lr_mult": {g: 0.35 for g in PARAM_GROUPS},
        "market_update": False,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "criticwarm",
        "steps": 200,
        "data_profile": "fineweb_cosmo_70_30",
        "train_groups": ["critic"],
        "lr_mult": {
            "rwkv_backbone": 0.0,
            "router_conf": 0.0,
            "rwkv_experts": 0.0,
            "trans_experts": 0.0,
            "bridge": 0.0,
            "critic": 2.0,
            "emb_head": 0.0,
        },
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "prenormal",
        "steps": 100,
        "data_profile": "fineweb_cosmo_70_30",
        "train_groups": ["all"],
        "lr_mult": {
            "rwkv_backbone": 0.35,
            "router_conf": 0.7,
            "rwkv_experts": 0.6,
            "trans_experts": 0.8,
            "bridge": 0.8,
            "critic": 1.6,
            "emb_head": 0.5,
        },
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "normal",
        "steps": 450,
        "data_profile": "fineweb_cosmo_70_30",
        "train_groups": ["all"],
        "lr_mult": {
            "rwkv_backbone": 0.7,
            "router_conf": 1.0,
            "rwkv_experts": 1.0,
            "trans_experts": 1.0,
            "bridge": 1.0,
            "critic": 1.0,
            "emb_head": 0.8,
        },
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
    {"name": "sft", "steps": 0, "data_profile": "ultrachat_100", "train_groups": ["all"], "lr_mult": {}, "market_update": True, "use_market": True, "route_grad": True},
    {"name": "rlhf", "steps": 0, "data_profile": "rlhf_placeholder", "train_groups": ["all"], "lr_mult": {}, "market_update": True, "use_market": True, "route_grad": True},
]

CONFIG_04B_TOY = {
    **CONFIG_04B,
    "scale": "0.4b_toy",
    "run_name": f"Toy-0.4b-{VARIANT}-{VERSION}",
    "micro_batch_size": 1,
    "ctx_len": 512,
    "grad_accum": 24,
    "phase_schedule": CONFIG_04B_TOY_PHASES,
    "total_steps": _total_steps(CONFIG_04B_TOY_PHASES),
    "log_interval": 10,
    "eval_interval": 200,
    "eval_iters": 20,
    "base_lr": 8e-5,
    # Diagnostic/runtime switches
    "train_use_amp": True,
    "amp_dtype": "bfloat16",
    "cuda_use_fast_math": True,
    "cuda_force_fp32_kernel": False,
    "use_gradient_checkpoint": False,
    "checkpoint_att_stage": False,
    "checkpoint_expert_stage": False,
    "route_no_grad": True,
    "router_noise_std": 0.02,
    "market_alpha_init": 0.05,
    "aux_loss_coeff": 0.01,
    "lazy_prefix_union": True,
    "save_dir": f"checkpoints/{VERSION}_0.4b_toy",
}


# ==========================================
# 0.1B 配置（保持 pilot）
# ==========================================
CONFIG_01B = {
    **CONFIG_PILOT,
    "version": VERSION,
    "project": f"CaMoE-{VERSION}",
    "run_name": f"Pilot-0.1b-{VERSION}",
    "param_groups": PARAM_GROUPS,
    "router_noise_std": 0.02,
    "market_alpha_init": 0.05,
    "aux_loss_coeff": 0.01,
    "save_dir": f"checkpoints/{VERSION}_0.1b",
}


def get_config(scale: str = "0.4b"):
    if scale == "0.4b":
        return CONFIG_04B
    if scale == "0.4b_toy":
        return CONFIG_04B_TOY
    if scale == "0.1b":
        return CONFIG_01B
    if scale == "pilot":
        return CONFIG_PILOT
    raise ValueError(f"Unknown scale: {scale}")


# 向后兼容
CONFIG_MINIPILE = CONFIG_04B
