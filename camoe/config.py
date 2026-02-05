CONFIG_MINIPILE = {
    "project": "CaMoE-v16",
    "run_name": "MiniPile-0.1b-3R1T-first",
    
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
    "tax_threshold": 2.0,
    "tax_rate": 0.1,
    
    # ===== 训练参数 (5080 16GB) =====
    "micro_batch_size": 3,
    "ctx_len": 768,      
    "grad_accum": 16,     # 有效 batch = 48
    "total_steps": 20000,  # 见下方计算
    
    # ===== 阶段控制 =====
    "prewarm_steps": 0,
    "warmup_steps": 1000,
    
    # ===== 学习率 =====
    "lr_prewarm": 1e-4,
    "lr_warmup": 2e-4,
    "lr_normal": 1.5e-4,
    
    # ===== 路径 =====
    "data_path": "./data/minipile_processed",
    "weights_path": "model/rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    "vocab_file": "tokenizer/rwkv_vocab_v20230424.txt",
    "save_dir": "checkpoints/minipile",
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