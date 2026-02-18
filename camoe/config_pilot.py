# camoe/config_pilot.py
VERSION = "v20-pilot-min"
SCALE = "0.1b"
VARIANT = "3R1T-Top2-Min"

CONFIG_PILOT = {
    # meta
    "version": VERSION,
    "scale": SCALE,
    "variant": VARIANT,
    "project": "CaMoE-Local",
    "run_name": f"PilotMin-{SCALE}-{VERSION}",

    # model (0.1B)
    "n_embd": 768,
    "n_layer": 12,
    "head_size": 64,
    "vocab_size": 65536,
    "tied_embeddings": True,
    "use_deep_embed_attention": False,
    "use_shared_deep_embed": True,

    # experts
    "num_rwkv_experts": 3,
    "num_trans_experts": 1,
    "top_k": 2,
    "prefix_len": 32,
    "low_rank_dim": 32,

    # optimizer / phase
    "base_lr": 8e-5,
    "param_groups": [
        "rwkv_backbone", "rwkv_experts", "trans_experts",
        "bridge", "critic", "emb_head"
    ],
    "phase_schedule": [
        {"name": "prewarm",   "steps": 200,  "data_profile": "default", "train_groups": ["trans_experts", "bridge"],
         "lr_mult": {"rwkv_backbone":0.0,"rwkv_experts":0.0,"trans_experts":1.0,"bridge":1.0,"critic":0.0,"emb_head":0.0},
         "market_update": False},
        {"name": "warm",      "steps": 300,  "data_profile": "default", "train_groups": ["all"],
         "lr_mult": {"rwkv_backbone":0.35,"rwkv_experts":0.35,"trans_experts":0.35,"bridge":0.35,"critic":0.35,"emb_head":0.35},
         "market_update": False},
        {"name": "criticwarm","steps": 300,  "data_profile": "default", "train_groups": ["critic"],
         "lr_mult": {"rwkv_backbone":0.0,"rwkv_experts":0.0,"trans_experts":0.0,"bridge":0.0,"critic":2.0,"emb_head":0.0},
         "market_update": True},
        {"name": "prenormal", "steps": 200,  "data_profile": "default", "train_groups": ["all"],
         "lr_mult": {"rwkv_backbone":0.35,"rwkv_experts":0.6,"trans_experts":0.8,"bridge":0.8,"critic":1.6,"emb_head":0.5},
         "market_update": True},
        {"name": "normal",    "steps": 1000, "data_profile": "default", "train_groups": ["all"],
         "lr_mult": {"rwkv_backbone":0.7,"rwkv_experts":1.0,"trans_experts":1.0,"bridge":1.0,"critic":1.0,"emb_head":0.8},
         "market_update": True},
        {"name": "sft",       "steps": 0,   "data_profile": "default", "train_groups": ["all"], "lr_mult": {}, "market_update": True},
        {"name": "rlhf",      "steps": 0,   "data_profile": "default", "train_groups": ["all"], "lr_mult": {}, "market_update": True},
    ],
    "total_steps": 2000,

    # training
    "micro_batch_size": 1,
    "ctx_len": 256,
    "grad_accum": 8,
    "use_gradient_checkpoint": False,
    "checkpoint_att_stage": False,
    "checkpoint_expert_stage": False,
    "route_no_grad": True,
    "lazy_prefix_union": True,

    # market/economy (minimal)
    "total_capital": 1000.0,
    "min_capital_share": 0.05,
    "tax_threshold": 1.5,
    "tax_rate": 0.2,

    # stability
    "train_use_amp": False,
    "amp_dtype": "bfloat16",
    "cuda_use_fast_math": False,
    "cuda_force_fp32_kernel": False,
    "stabilize_logits": True,
    "nan_debug": False,

    # io
    "log_interval": 10,
    "eval_interval": 50,
    "eval_iters": 10,
    "data_path": "./data/camoe_toy_mix",
    "weights_path": "model/rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    "vocab_file": "tokenizer/rwkv_vocab_v20230424.txt",
    "save_dir": f"checkpoints/{VERSION}",
}

def get_pilot_config():
    return CONFIG_PILOT
