# config_pilot.py

VERSION = "v21"
SCALE = "0.1b"
VARIANT = "3R1T-Top2"

PARAM_GROUPS_PILOT = [
    "rwkv_backbone",
    "router_conf",      # ← 新增：confidence 网络独立分组
    "rwkv_experts",
    "trans_experts",
    "bridge",
    "critic",
    "emb_head",
]

# 每个 phase 的完整 lr_mult 模板（方便复用）
def _lr(backbone=0.0, router=0.0, rwkv_exp=0.0, trans_exp=0.0, 
        bridge=0.0, critic=0.0, emb=0.0):
    return {
        "rwkv_backbone": backbone,
        "router_conf":   router,
        "rwkv_experts":  rwkv_exp,
        "trans_experts": trans_exp,
        "bridge":        bridge,
        "critic":        critic,
        "emb_head":      emb,
    }

PILOT_PHASES = [
    {
        "name": "prewarm",
        "steps": 1000,
        "data_profile": "default",
        "train_groups": ["router_conf", "bridge"],
        "lr_mult": _lr(router=1.0, bridge=1.0),
        "market_update": False,
        "use_market": False,     # 纯梯度路由，让 gate 先学个大概
        "route_grad": True,
    },
    {
        "name": "warm",
        "steps": 2000,
        "data_profile": "default",
        "train_groups": ["all"],
        "lr_mult": _lr(0.35, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35),
        #                     ^^^ router 稍高，让它跟上其他模块
        "market_update": False,
        "use_market": True,      # 开始引入市场 bias，但不更新资本
        "route_grad": True,
    },
    {
        "name": "criticwarm",
        "steps": 2000,
        "data_profile": "default",
        "train_groups": ["critic"],
        "lr_mult": _lr(critic=2.0),
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "prenormal",
        "steps": 2000,
        "data_profile": "default",
        "train_groups": ["all"],
        "lr_mult": _lr(0.35, 0.7, 0.6, 0.8, 0.8, 1.6, 0.5),
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "normal",
        "steps": 23000,
        "data_profile": "default",
        "train_groups": ["all"],
        "lr_mult": _lr(0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8),
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "sft",
        "steps": 0,
        "data_profile": "default",
        "train_groups": ["all"],
        "lr_mult": _lr(),
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
    {
        "name": "rlhf",
        "steps": 0,
        "data_profile": "default",
        "train_groups": ["all"],
        "lr_mult": _lr(),
        "market_update": True,
        "use_market": True,
        "route_grad": True,
    },
]


CONFIG_PILOT = {
    # ===== 元信息 =====
    "version": VERSION,
    "scale": SCALE,
    "variant": VARIANT,
    "project": "CaMoE-Pilot",
    "run_name": f"TS-Pilot-{SCALE}-{VERSION}",

    # ===== 模型结构 (0.1B) =====
    "n_embd": 768,
    "n_layer": 12,
    "head_size": 64,
    "vocab_size": 65536,
    "tied_embeddings": True,
    "use_deep_embed_attention": False,
    "use_shared_deep_embed": True,

    # ===== 专家配置 =====
    "num_rwkv_experts": 3,
    "num_trans_experts": 1,
    "top_k": 2,
    "prefix_len": 32,
    "low_rank_dim": 32,

    # ===== 优化器与阶段 =====
    "base_lr": 8e-5,
    "param_groups": PARAM_GROUPS_PILOT,
    "phase_schedule": PILOT_PHASES,
    "total_steps": 30000,

    # ===== 训练参数 =====
    "micro_batch_size": 4,
    "ctx_len": 512,              # 256 → 512，TinyStories 够短，但多点上下文有好处
    "grad_accum": 12,
    "use_gradient_checkpoint": False,
    "checkpoint_att_stage": False,
    "checkpoint_expert_stage": False,

    # ===== v21 路由核心参数 =====
    "route_no_grad": False,      # 关键：v21 需要梯度流通
    "router_noise_std": 0.03,    # 训练噪声（0.1B 模型稍大一点，帮助探索）
    "market_alpha_init": 0.05,   # buffer 初始值（直接是 alpha，不是 logit）
    "aux_loss_coeff": 0.05,      # Load Balance Loss（从 0.01 加到 0.05，修复死层）
    "lazy_prefix_union": True,

    # ===== 市场/经济系统 =====
    "total_capital": 1000.0,
    "min_capital_share": 0.05,
    "tax_threshold": 1.5,
    "tax_rate": 0.2,
    "economy": {
        # QE/QT（保守设置，0.1B 不需要太激进）
        "qe_low_ratio": 0.85,
        "qe_high_ratio": 1.20,
        "qe_inject_ratio": 0.15,
        "qe_drain_ratio": 0.08,
        "qe_floor_alloc_ratio": 0.70,
        # 基础算力保障
        "base_compute_floor_ratio": 0.08,  # 比 0.4B 稍高，0.1B 专家少不能太偏
        # 破产保护
        "critic_bankrupt_threshold_ratio": 0.20,
        "bailout_base": 500.0,     # 总资本只有 1000，bailout 相应缩小
        "bailout_decay": 0.65,
        "bailout_min": 100.0,
        "repay_ratio": 0.25,
        "restructure_alpha": 0.12,
        "donor_topk": 2,
        # 闲置税 + 折旧
        "idle_threshold": 0.01,
        "idle_tax_rate": 0.02,
        "depreciation_rate": 0.001,
        # 分红
        "base_commission": 0.8,
        "dividend_scale": 0.4,
        "dividend_std_factor": 0.5,
        # 风投
        "vc_affinity_threshold": 0.15,
        "vc_low_cap_ratio": 0.85,
        "vc_selected_threshold": 0.01,
        "vc_inject_ratio": 0.01,
        "vc_max_inject_ratio": 0.12,
        # CriticWarm
        "criticwarm_reward_scale": 2.0,
        "criticwarm_penalty_scale": 0.4,
        "critic_bonus_scale": 0.2,
        "critic_bonus_clip": (-0.1, 0.3),
        "critic_bonus_ema_momentum": 0.95,
    },

    # ===== 稳定性 =====
    "train_use_amp": True,       # 开启 AMP（BF16 对 0.1B 没问题）
    "amp_dtype": "bfloat16",
    "cuda_use_fast_math": True,  # 0.1B 足够稳定
    "cuda_force_fp32_kernel": False,
    "stabilize_logits": False,   # 不需要了，v21 logits 更健康
    "nan_debug": False,

    # ===== 日志与评估 =====
    "log_interval": 10,
    "eval_interval": 500,        # 每 500 步评测，30k 步 = 60 次评测点
    "eval_iters": 20,            # 每次 20 batch

    # ===== 数据与权重 =====
    "data_path": "./data/camoe_toy_mix",
    "weights_path": "model/rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    "vocab_file": "tokenizer/rwkv_vocab_v20230424.txt",
    "save_dir": f"checkpoints/{VERSION}_{SCALE}_pilot",
}