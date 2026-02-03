CONFIG_01B = {
    "project": "CaMoE-v10",
    "run_name": "v12-0.1b-tinystories-3R-1T-fix",
    
    # 模型 (需要check_weights确认)
    "n_embd": 768,
    "n_layer": 12,
    "head_size": 64,
    "vocab_size": 65536,
    
    # CaMoE
    "prefix_len": 16,
    'num_trans_experts': 1,
    "num_rwkv_experts": 3, 
    
    # Market
    "total_capital": 10000.0,
    "min_capital_share": 0.05,
    "tax_threshold": 1.2,
    "tax_rate": 0.20,
    
    # 训练
    "micro_batch_size": 6,
    "ctx_len": 1024,
    "grad_accum": 10,
    "total_steps": 15000,
    
    # 阶段
    "prewarm_steps": 300,
    "warmup_steps": 1500,
    #"rewarm_interval": 3500,
    #"rewarm_duration": 400,
    
    # 学习率
    "lr_prewarm": 5e-4,
    "lr_warmup": 3e-4,
    "lr_normal": 1e-4,
    
    # 路径
    "weights_path": "model/rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    "vocab_file": "tokenizer/rwkv_vocab_v20230424.txt",
    "save_dir": "checkpoints",
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