import torch
import torch.nn as nn
import sys
import os

# 确保能导入 CaMoE 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CaMoE.system import CaMoE_System
from CaMoE.config import CONFIG_04B, CONFIG_01B # 导入 v18 配置

def profile_vram(scale="0.4b"):
    print(f"\n🚀 Profiling CaMoE v18 [{scale.upper()}] ...")
    
    # 选择配置
    base_config = CONFIG_04B if scale == "0.4b" else CONFIG_01B
    config = base_config.copy()
    
    # 强制修改一些可能影响显存的参数以进行压力测试
    # config['micro_batch_size'] = 6  # 可以在这里覆盖测试
    # config['ctx_len'] = 1024
    
    device = "cuda"
    if not torch.cuda.is_available():
        print("❌ No CUDA device found!")
        return

    print(f"⚙️ Config: Batch={config['micro_batch_size']}, Ctx={config['ctx_len']}, "
          f"Experts={config['num_rwkv_experts']}R+{config['num_trans_experts']}T (Top-{config['top_k']})")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated() / 1024**2

    # 1. 模型加载
    print("📦 Loading Model...")
    try:
        model = CaMoE_System(config).to(device)
    except Exception as e:
        print(f"❌ Model Init Failed: {e}")
        return

    model_mem = torch.cuda.memory_allocated() / 1024**2 - base_mem
    print(f"  - Model Weights: {model_mem:.2f} MB ({model_mem/1024:.2f} GB)")

    # 2. Forward Pass (含 Activation)
    print("🔄 Running Forward Pass...")
    # 构造假数据
    x = torch.randint(0, config['vocab_size'], (config['micro_batch_size'], config['ctx_len'])).to(device)
    target = torch.randint(0, config['vocab_size'], (config['micro_batch_size'], config['ctx_len'])).to(device)
    
    # 开启混合精度以模拟真实训练显存
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, info = model(x, step=1000, phase="normal") # phase="normal" 启用 Market
        
        forward_mem = torch.cuda.memory_allocated() / 1024**2 - (base_mem + model_mem)
        print(f"  - Activations (Static): {forward_mem:.2f} MB")
        
        # 3. Compute Loss
        total_loss, _, _, _, _ = model.compute_losses(logits, target, info)
    
    # 4. Backward Pass (Peak Memory)
    print("🔙 Running Backward Pass...")
    try:
        total_loss.backward()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ OOM during Backward!")
            return
        else:
            raise e

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"📈 Peak Memory: {peak_mem:.2f} MB ({peak_mem/1024:.2f} GB)")
    
    # 5. Optimizer Step Simulation (Optional)
    # 8-bit AdamW 状态占用
    # param_bytes = 4 (float32 master) + 2 (bf16 weight) + 1 (state1) + 1 (state2) ? 
    # bnb 8bit: state is compressed. 
    # 估算: Model Weights * 0.5 (optim states)
    optim_est = model_mem * 0.25 # Rough estimate for 8-bit Adam
    total_est = peak_mem + optim_est
    print(f"🔮 Estimated Total Training VRAM (w/ Opt): ~{total_est/1024:.2f} GB")
    
    if total_est > 32000: # 32GB (RTX 5090)
        print("⚠️ WARNING: Might exceed 32GB VRAM!")
    else:
        print("✅ Safe for RTX 5090 (32GB)")

if __name__ == "__main__":
    profile_vram("0.4b")
    # profile_vram("0.1b") 