"""
CaMoE v18 配置文件
使用 version 字段自动生成路径
"""
from .config_pilot import CONFIG_PILOT
# ==========================================
# 版本控制
# ==========================================
VERSION = "v18.5-test"
SCALE = "0.4b"  # "0.1b" or "0.4b"
VARIANT = "6R2T-Top2-DEA"  # 架构变体：含 DeepEmbed + DeepEmbedAttention

# ==========================================
# 自动生成的标识符
# ==========================================
RUN_ID = f"MiniPile-{SCALE}-{VARIANT}-{VERSION}"

# ==========================================
# 0.4B 规模配置 (v18 主力)
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
    "use_deep_embed_attention": True,  # v18.5-test: DeepEmbed + DEA 分支
    "use_shared_deep_embed": True,  # 跨层共享 DeepEmbed 表，显著降低参数/显存
    "dea_q_dim": 256,  # RWKV-8 风格 DEA: q 维度
    "dea_kv_dim": 32,  # RWKV-8 风格 DEA: k/v 低维缓存通道
    "dea_score_scale": 1024.0,
    "dea_cap_scale": 64.0,
    # ROSA experimental branch (default off)
    "use_rosa": False,
    "rosa_num_streams": 32,
    "rosa_emb_dim": 64,
    
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
    
    # ===== 日志与评估 =====
    "log_interval": 10,
    "eval_interval": 1000,
    "eval_iters": 50,
    
    # ===== 路径 (f-string 自动生成) =====
    # 单数据集时使用 data_path；混合时使用 data_roots + mix（课程学习手动 Resume 换阶段）
    "data_path": "./data/minipile_rwkv_processed",
    "data_roots": {
        "fineweb_edu": "./data/fineweb_edu_rwkv_processed",
        "minipile": "./data/minipile_rwkv_processed",
        "cosmopedia": "./data/cosmopedia_rwkv_processed",
        "ultrachat": "./data/ultrachat_rwkv_processed",
        "tinystories": "./data/tinystories_rwkv_processed",
    },
    # 方案A（均衡版）：总和=1.0
    "mix": {
        "fineweb_edu": 0.4,
        "minipile": 0.2,
        "ultrachat": 0.2,
        "cosmopedia": 0.1,
        "tinystories": 0.1,
    },
    "weights_path": f"model/rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    "vocab_file": "tokenizer/rwkv_vocab_v20230424.txt",
    "save_dir": f"checkpoints/{VERSION}_{SCALE}",
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
    "use_rosa": False,
    "rosa_num_streams": 32,
    "rosa_emb_dim": 64,
    
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
    elif scale == "0.1b":
        return CONFIG_PILOT
    else:
        raise ValueError(f"Unknown scale: {scale}")

# 向后兼容
CONFIG_MINIPILE = CONFIG_04B
