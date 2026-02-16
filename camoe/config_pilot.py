VERSION = "v19-pilot-1"
SCALE = "0.1b" 
VARIANT = "3R1T-Top2-P32"

# ==========================================
# 自动生成的标识符
# ==========================================
RUN_ID = f"TinyStories-{SCALE}-{VARIANT}-{VERSION}"

CONFIG_PILOT = {
    # ===== 元信息 =====
    "version": VERSION,
    "scale": SCALE,
    "variant": VARIANT,
    "project": "CaMoE-Local", # SwanLab 项目名
    "run_name": RUN_ID,
    
    # ===== 模型结构 (0.1B 迷你版) =====
    "n_embd": 768,
    "n_layer": 12,
    "head_size": 64,
    # 如果本地没有 32k tokenizer 也没事，用 RWKV 65k + Tied Embedding 一样跑
    # 这里假设你还没练好 32k，先用 RWKV 原生配置验证代码逻辑
    "vocab_size": 65536, 
    "tied_embeddings": True,
    # "tokenizer_path": ... (如果用 RWKV tokenizer 就不需要这个)
    
    # ===== CaMoE 专家配置 (3R + 1T) =====
    # 这是一个非常有趣的组合：
    # 总共 4 个专家，选 Top-2。
    # 这意味着 Transformer (1T) 只要不垫底 (Top 3/4)，就有很大概率被选中！
    # 这能强迫 Bridge 快速学会输出有效信息。
    "num_rwkv_experts": 3,
    "num_trans_experts": 1,
    "top_k": 2,
    
    "prefix_len": 32, # 显存够用，给它一点视野
    "low_rank_dim": 32,
    
    # ===== Market 参数 (测试激进策略) =====
    "total_capital": 1000.0, # 资本池小一点
    "min_capital_share": 0.05,
    "tax_threshold": 1.5, # 税收严一点，防止垄断
    "tax_rate": 0.2,
    
    # ===== 训练参数 (保守显存版，优先跑通) =====
    "micro_batch_size": 1,
    "ctx_len": 512,
    "grad_accum": 16, # 有效 batch = 16
    "total_steps": 5000,
    
    # ===== 阶段控制 =====
    "prewarm_steps": 500,  # 快速预热
    "warmup_steps": 1000,
    
    # ===== 学习率 =====
    "lr_prewarm": 3e-4,
    "lr_warmup": 5e-4,
    "lr_normal": 3e-4, # 0.1B 可以稍微大一点
    "use_gradient_checkpoint": False,
    "checkpoint_att_stage": False,
    "checkpoint_expert_stage": False,
    "route_no_grad": True,
    "lazy_prefix_union": True,
    # Diagnostic/runtime switches (no structure change)
    "train_use_amp": False,
    "amp_dtype": "bfloat16",
    "cuda_use_fast_math": False,
    "cuda_force_fp32_kernel": False,
    "stabilize_logits": True,
    "nan_debug": True,
    "sanitize_timemix_output": True,
    "force_timemix_fallback": False,
    
    # ===== 日志与评估 =====
    "log_interval": 10,
    "eval_interval": 500,
    "eval_iters": 20,
    
    # ===== 路径 (本地路径) =====
    # 假设你有 TinyStories 数据，如果没有，可以用 HuggingFace 在线流式
    # 这里指向你的本地数据目录
    "data_path": "./data/camoe_toy_mix",
    "weights_path": "model/rwkv7-g1d-0.1b-20260129-ctx8192.pth", # 如果没有底模，代码会自动随机初始化
    "vocab_file": "tokenizer/rwkv_vocab_v20230424.txt",
    "save_dir": f"checkpoints/{VERSION}",
}

# 导出给 get_config 用
def get_pilot_config():
    return CONFIG_PILOT
