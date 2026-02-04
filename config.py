CONFIG_BABYLM = {
    "project": "CaMoE-v12",
    "run_name": "babylm-0.1b-3R1T-final",
    
    # ===== 模型结构 (和 TinyStories 一样) =====
    "n_embd": 768,
    "n_layer": 12,
    "head_size": 64,
    "vocab_size": 65536,
    
    # ===== CaMoE 专家配置 =====
    "num_rwkv_experts": 3,
    "num_trans_experts": 1,
    "prefix_len": 16,
    
    # ===== Market 参数 =====
    "total_capital": 10000.0,
    "min_capital_share": 0.05,
    "tax_threshold": 1.5,
    "tax_rate": 0.15,
    
    # ===== 训练参数 (5080 16GB) =====
    "micro_batch_size": 4,
    "ctx_len": 512,      # BabyLM 句子较短
    "grad_accum": 12,     # 有效 batch = 48
    "total_steps": 25000,  # 见下方计算
    
    # ===== 阶段控制 =====
    "prewarm_steps": 500,
    "warmup_steps": 2500,
    
    # ===== 学习率 =====
    "lr_prewarm": 3e-4,
    "lr_warmup": 1.5e-4,
    "lr_normal": 5e-5,
    
    # ===== 路径 =====
    "data_path": "./data/babylm_100M_processed",
    "weights_path": "model/rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    "vocab_file": "tokenizer/rwkv_vocab_v20230424.txt",
    "save_dir": "checkpoints/babylm",
}

CONFIG_04B = {
    "project": "CaMoE-v10",
    "run_name": "v10-0.4b-g1a",
    
    "n_embd": 1024,
    "n_layer": 24,
    "head_size": 64,
    "vocab_size": 65536,
    
    "prefix_len": 16,
    "num_rwkv_experts": 2,
    
    "total_capital": 10000.0,
    "min_capital_share": 0.05,
    "tax_threshold": 1.5,
    "tax_rate": 0.15,
    
    "batch_size": 4,
    "ctx_len": 1024,
    "grad_accum": 8,
    "total_steps": 30000,
    
    "prewarm_steps": 500,
    "warmup_steps": 2000,
    "rewarm_interval": 5000,
    "rewarm_duration": 500,
    
    "lr_prewarm": 5e-4,
    "lr_warmup": 2e-4,
    "lr_normal": 5e-5,
    
    "weights_path": "model/rwkv7-g1a-0.4b-20250905-ctx4096.pth",
    "vocab_file": "tokenizer/rwkv_vocab_v20230424.txt",
    "save_dir": "checkpoints",
}