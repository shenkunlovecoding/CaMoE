"""
CaMoE v19 配置文件
使用 version 字段自动生成路径
"""
from .config_pilot import CONFIG_PILOT
# ==========================================
# 版本控制
# ==========================================
VERSION = "v19"
SCALE = "0.4b"  # "0.1b" or "0.4b"
VARIANT = "6R2T-Top2-DEA"  # 架构变体：含 DeepEmbed + DeepEmbedAttention

# ==========================================
# 自动生成的标识符
# ==========================================
RUN_ID = f"FineWebEdu-UltraChat-Cosmopedia-{SCALE}-{VARIANT}-{VERSION}"

# ==========================================
# 0.4B 规模配置 (v19 主力)
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
    "n_layer": 16,  # 比 0.1b 的 12 层多
    "head_size": 64,
    "vocab_size": 65536,  
    "tied_embeddings": True,  # 共享 Input/Output Embedding
    "use_deep_embed_attention": True,  # v19: DeepEmbed + DEA 分支
    "use_shared_deep_embed": True,  # 跨层共享 DeepEmbed 表，显著降低参数/显存
    "dea_q_dim": 256,  # RWKV-8 风格 DEA: q 维度
    "dea_kv_dim": 32,  # RWKV-8 风格 DEA: k/v 低维缓存通道
    "dea_score_scale": 1024.0,
    "dea_cap_scale": 64.0,
    
    # ===== CaMoE 专家配置 (6R2T) =====
    "num_rwkv_experts": 6,
    "num_trans_experts": 2,
    "top_k": 2,  # Top-2 路由
    "prefix_len": 48,
    "low_rank_dim": 64,  # Bridge 低秩维度
    
    # ===== Market 参数 =====
    "total_capital": 10000.0,
    "min_capital_share": 0.05,
    "tax_threshold": 2.0,
    "tax_rate": 0.1,
    
    # ===== 训练参数 (RTX 5090 32GB；开启 DEA 时显存略增，可酌情微调 micro_batch_size) =====
    "micro_batch_size": 6,
    "ctx_len": 1024,
    "grad_accum": 8,  # 有效 batch = 48
    "total_steps": 100000,
    
    # ===== 阶段控制 (超长预热) =====
    "prewarm_steps": 4000,   # 冻结 RWKV，只训 Bridge/Trans
    "warmup_steps": 10000,   # 全参数热身
    
    # ===== 学习率 =====
    "lr_prewarm": 1e-4,
    "lr_warmup": 2e-4,
    "lr_normal": 3e-4,
    "use_gradient_checkpoint": True,
    "checkpoint_att_stage": True,
    "checkpoint_expert_stage": True,
    "route_no_grad": True,
    "lazy_prefix_union": True,
    
    # ===== 日志与评估 =====
    "log_interval": 10,
    "eval_interval": 1000,
    "eval_iters": 50,
    
    # ===== 路径 (f-string 自动生成) =====
    # 单数据集时使用 data_path；混合时使用 data_roots + mix（课程学习手动 Resume 换阶段）
    "data_path": "./data/camoe_mix_v19_edu_chat_cosmo",
    "data_roots": {
        "fineweb_edu": "./data/fineweb_edu_sample10bt_rwkv_processed",
        "ultrachat_200k": "./data/ultrachat_200k_rwkv_processed",
        "cosmopedia_100k": "./data/cosmopedia_100k_rwkv_processed",
    },
    # 混合比例：配置好后训练，到下一阶段改比例并 Resume 即可
    # 不配置 mix 则使用 data_path 单数据集
    "mix": {
        "fineweb_edu": 1.0 / 3.0,
        "ultrachat_200k": 1.0 / 3.0,
        "cosmopedia_100k": 1.0 / 3.0,
    },
    "weights_path": f"model/rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    "vocab_file": "tokenizer/rwkv_vocab_v20230424.txt",
    "save_dir": f"checkpoints/{VERSION}_{SCALE}",
}

# ==========================================
# 0.4B Toy 配置（快速验证流程）
# ==========================================
CONFIG_04B_TOY = {
    **CONFIG_04B,
    "scale": "0.4b_toy",
    "run_name": f"Toy-0.4b-{VARIANT}-{VERSION}",
    "micro_batch_size": 1,
    "ctx_len": 512,
    "grad_accum": 24,
    "total_steps": 6000,
    "prewarm_steps": 500,
    "warmup_steps": 1500,
    "eval_interval": 200,
    "eval_iters": 20,
    "lr_prewarm": 3e-5,
    "lr_warmup": 8e-5,
    "lr_normal": 1e-4,
    # Diagnostic/runtime switches (no structure change)
    "train_use_amp": True,
    "amp_dtype": "bfloat16",
    "cuda_use_fast_math": True,
    "cuda_force_fp32_kernel": False,
    "use_gradient_checkpoint": False,
    "checkpoint_att_stage": False,
    "checkpoint_expert_stage": False,
    "route_no_grad": True,
    "lazy_prefix_union": True,
    "data_path": "./data/camoe_mix_v1",
    "mix": None,
    "save_dir": f"checkpoints/{VERSION}_0.4b_toy",
}

# ==========================================
# 0.1B 规模配置 (测试/对照组)
# ==========================================
CONFIG_01B = {
    # ===== 元信息 =====
    "version": VERSION,
    "scale": "0.1b",
    "variant": "3R1T-Top2",
    "project": f"CaMoE-{VERSION}",
    "run_name": f"MiniPile-0.1b-3R1T-Top2-{VERSION}",
    
    # ===== 模型结构 =====
    "n_embd": 768,
    "n_layer": 12,
    "head_size": 64,
    "vocab_size": 32000,
    "tied_embeddings": True,
    "use_deep_embed_attention": False,
    "use_shared_deep_embed": True,
    "dea_q_dim": 256,
    "dea_kv_dim": 32,
    "dea_score_scale": 1024.0,
    "dea_cap_scale": 64.0,
    
    # ===== CaMoE 专家配置 (3R1T) =====
    "num_rwkv_experts": 3,
    "num_trans_experts": 1,
    "top_k": 2,
    "prefix_len": 64,
    "low_rank_dim": 64,
    
    # ===== Market 参数 =====
    "total_capital": 10000.0,
    "min_capital_share": 0.05,
    "tax_threshold": 2.0,
    "tax_rate": 0.1,
    
    # ===== 训练参数 =====
    "micro_batch_size": 4,
    "ctx_len": 768,
    "grad_accum": 12,
    "total_steps": 40000,
    
    # ===== 阶段控制 =====
    "prewarm_steps": 2000,
    "warmup_steps": 6000,
    
    # ===== 学习率 =====
    "lr_prewarm": 1e-4,
    "lr_warmup": 2e-4,
    "lr_normal": 3e-4,
    "use_gradient_checkpoint": True,
    "checkpoint_att_stage": True,
    "checkpoint_expert_stage": True,
    "route_no_grad": True,
    "lazy_prefix_union": True,
    
    # ===== 日志与评估 =====
    "log_interval": 10,
    "eval_interval": 1000,
    "eval_iters": 50,
    
    # ===== 路径 =====
    "data_path": "./data/minipile_processed",
    "data_roots": {
        "tinystories": "./data/tinystories_rwkv_processed",
        "dialog": "./data/dialog_rwkv_processed",
        "minipile": "./data/minipile_rwkv_processed",
    },
    "mix": None,  # 0.1b 默认单数据集，需要时配置 mix
    "weights_path": "model/rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    "vocab_file": "tokenizer/rwkv_vocab_v20230424.txt",
    "save_dir": f"checkpoints/{VERSION}_0.1b",
}

# ==========================================
# 默认配置选择
# ==========================================
def get_config(scale: str = "0.4b"):
    if scale == "0.4b":
        return CONFIG_04B
    elif scale == "0.4b_toy":
        return CONFIG_04B_TOY
    elif scale == "0.1b":
        return CONFIG_PILOT
    else:
        raise ValueError(f"Unknown scale: {scale}")

# 向后兼容
CONFIG_MINIPILE = CONFIG_04B
